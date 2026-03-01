from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans

from common.data_classes.documents import Document
from common.data_classes.knowledge_triplets import ExtractedKnowledgeTriplet, StructuredDocument
from common.data_classes.rag_system import Chunk
from common.neo4j.data_classes import Executor
from common.neo4j.neo4j_environment import Neo4JEnvironment
from common.strategies.encoding import EncodingStrategy


class StandardExecutor(Executor):
    def __init__(
        self,
        *,
        env: Neo4JEnvironment,
        encoder: EncodingStrategy,
        similarity: str = "cosine",
        doc_centroids: int = 5,
    ):
        self.encoder = encoder
        self.embedding_dimension = encoder.get_embedding_dims()
        self.similarity = similarity
        self.env = env
        self.doc_centroids = doc_centroids

    def persist(self, structured: StructuredDocument, chunks: List[Chunk]):
        """
        Persist a StructuredDocument and its chunks into Neo4j following the GraphRAG model.
        """
        import hashlib, math

        # ---- Prepare document props ----
        doc = structured.document
        doc_props = {
            "id": doc.id,
            "title": getattr(doc, "title", None),
            "text": getattr(doc, "text", ""),  # Store full text on document node
            "author": getattr(doc, "author", None),
            "source": getattr(doc, "source", None),  # include source to match FTS index
            "publication_date": (
                doc.publication_date.isoformat() if getattr(doc, "publication_date", None) else None
            ),
            "references": getattr(doc, "references", None) or [],
        }

        # ---- Prepare chunk records (with embeddings) ----
        def _to_float_list(v):
            v = v.tolist() if hasattr(v, "tolist") else list(v)
            if self.embedding_dimension and len(v) != self.embedding_dimension:
                raise ValueError(
                    f"Embedding has dim {len(v)} != expected {self.embedding_dimension}"
                )
            # guard against NaN/Inf which break vector indexes
            for x in v:
                if not (isinstance(x, (int, float)) and math.isfinite(float(x))):
                    raise ValueError("Embedding contains non-finite values (NaN/Inf).")
            return [float(x) for x in v]

        chunk_records = []
        for ch in chunks or []:
            emb = self.encoder.encode(ch.text, query=False)  # encoding document chunks
            emb = _to_float_list(emb)
            chunk_records.append(
                {
                    "chunk_name": ch.chunk_id,
                    "text": ch.text,
                    "embedding": emb,
                }
            )

        # ---- Document centroid embeddings via k-means ----
        centroid_records: List[dict] = []
        if chunk_records and self.doc_centroids > 0:
            embeddings = np.array([c["embedding"] for c in chunk_records])
            k = min(self.doc_centroids, len(embeddings))
            kmeans = KMeans(n_clusters=k, n_init="auto", random_state=0)
            kmeans.fit(embeddings)
            centroids = kmeans.cluster_centers_.tolist()
            centroid_records = [
                {"id": f"{doc_props['id']}_c{i}", "embedding": _to_float_list(c)}
                for i, c in enumerate(centroids)
            ]

        # ---- Prepare entity records (dedupe by canonical name; union aliases) ----
        entity_map = {}
        for e in structured.entities or []:
            name = e.name
            if not name:
                continue
            rec = entity_map.get(name)
            if rec is None:
                entity_map[name] = {
                    "name": name,
                    "type": getattr(e, "type", None),
                    "aliases": list(dict.fromkeys(getattr(e, "aliases", None) or [])),
                }
            else:
                if not rec["type"] and getattr(e, "type", None):
                    rec["type"] = e.type
                existing = rec["aliases"]
                for alias in getattr(e, "aliases", None) or []:
                    if alias not in existing:
                        existing.append(alias)
        entity_records = list(entity_map.values())

        # ---- Prepare assertion records (dedupe subject|predicate|object; accumulate rank; collect chunk_ids) ----
        def assertion_id(subject: str, predicate: str, obj: str) -> str:
            key = f"{(subject or '').strip()}|{(predicate or '').strip()}|{(obj or '').strip()}".lower()
            return hashlib.sha256(key.encode("utf-8")).hexdigest()

        assertion_agg = {}
        for t in structured.triplets or []:
            aid = assertion_id(t.subject, t.relationship, t.object)
            if aid not in assertion_agg:
                triplet_text = f"{t.subject} {t.relationship} {t.object}"
                aemb = _to_float_list(self.encoder.encode(triplet_text, query=False))  # encoding knowledge triplets
                assertion_agg[aid] = {
                    "id": aid,
                    "predicate": t.relationship,
                    "embedding": aemb,
                    "rank": int(t.rank) if getattr(t, "rank", None) is not None else 1,
                    "subject": t.subject,
                    "object": t.object,
                    "chunk_ids": set([t.chunk_id] if getattr(t, "chunk_id", None) else []),
                }
            else:
                assertion_agg[aid]["rank"] += int(t.rank) if getattr(t, "rank", None) is not None else 1
                if getattr(t, "chunk_id", None):
                    assertion_agg[aid]["chunk_ids"].add(t.chunk_id)

        assertion_records = []
        for a in assertion_agg.values():
            a["chunk_ids"] = list(a["chunk_ids"])
            assertion_records.append(a)

        # ---- Persist to Neo4j in a single write transaction ----
        driver = self.env.get_driver()

        def _write_all(tx, doc_props, chunk_records, entity_records, assertion_records, centroid_records):
            tx.run(
                """
                MERGE (d:Document {id: $doc.id})
                SET d.title  = $doc.title,
                    d.text   = $doc.text,
                    d.author = $doc.author,
                    d.source = $doc.source,
                    d.references = $doc.references
                FOREACH (_ IN CASE WHEN $doc.publication_date IS NULL THEN [] ELSE [1] END |
                  SET d.publication_date = date(datetime($doc.publication_date)))
                """,
                doc=doc_props,
            )

            # Upsert Document centroids
            if centroid_records:
                tx.run(
                    """
                    MATCH (d:Document {id: $doc_id})
                    OPTIONAL MATCH (d)-[r:HAS_CENTROID]->(old:DocCentroid)
                    DELETE r, old
                    WITH d
                    UNWIND $centroids AS c
                    MERGE (dc:DocCentroid {id: c.id})
                    SET dc.embedding = c.embedding
                    MERGE (d)-[:HAS_CENTROID]->(dc)
                    """,
                    doc_id=doc_props["id"],
                    centroids=centroid_records,
                )

            # Upsert Chunks + HAS_CHUNK
            if chunk_records:
                tx.run(
                    """
                    MATCH (d:Document {id: $doc_id})
                    UNWIND $chunks AS c
                    MERGE (ch:Chunk {chunk_name: c.chunk_name})
                    SET  ch.text = c.text,
                         ch.embedding = c.embedding
                    MERGE (d)-[:HAS_CHUNK]->(ch)
                    """,
                    doc_id=doc_props["id"],
                    chunks=chunk_records,
                )

            # Upsert Entities (aliases additive, no dupes)
            if entity_records:
                tx.run(
                    """
                    UNWIND $entities AS e
                    MERGE (en:Entity {name: e.name})
                    SET en.type = e.type,
                        en.aliases =
                          CASE
                            WHEN en.aliases IS NULL THEN e.aliases
                            ELSE en.aliases + [x IN e.aliases WHERE NOT x IN en.aliases]
                          END
                    """,
                    entities=entity_records,
                )

            # Upsert Assertions (+ rank accumulation) and wire SUBJECT/OBJECT/DERIVED_FROM
            if assertion_records:
                tx.run(
                    """
                    UNWIND $assertions AS a
                    MERGE (as:Assertion {id: a.id})
                    ON CREATE SET as.predicate = a.predicate,
                                  as.embedding = a.embedding,
                                  as.rank = a.rank
                    ON MATCH  SET as.predicate = a.predicate,
                                  as.embedding = a.embedding,
                                  as.rank = coalesce(as.rank, 0) + a.rank

                    // Link SUBJECT and OBJECT
                    MERGE (sub:Entity {name: a.subject})
                    MERGE (obj:Entity {name: a.object})
                    MERGE (as)-[:SUBJECT]->(sub)
                    MERGE (as)-[:OBJECT]->(obj)

                    // Link provenance to all contributing chunks
                    WITH as, a
                    UNWIND a.chunk_ids AS cid
                    MATCH (ch:Chunk {chunk_name: cid})
                    MERGE (as)-[:DERIVED_FROM]->(ch)
                    """,
                    assertions=assertion_records,
                )

        with driver.session() as session:
            session.execute_write(
                _write_all,
                doc_props,
                chunk_records,
                entity_records,
                assertion_records,
                centroid_records,
            )

    def get_installation_schema(self) -> List[str]:
        dim = max(1, int(self.embedding_dimension))  # guardrail
        sim = self.similarity

        # We proactively drop vector indexes so we can guarantee the correct dimension/similarity.
        # Also align Entity uniqueness with persistence (by NAME, not by id).
        statements: List[str] = [
        "DROP INDEX chunk_embedding_ix IF EXISTS",
        "DROP INDEX assertion_embedding_ix IF EXISTS",
        "DROP INDEX doc_centroid_embedding_ix IF EXISTS",
        "DROP CONSTRAINT entity_id_unique IF EXISTS",
        "DROP CONSTRAINT doccentroid_id_unique IF EXISTS",
        ]

        # --- Constraints (Neo4j 5 syntax) ---
        statements += [
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS "
            "FOR (d:Document) REQUIRE d.id IS UNIQUE",

            "CREATE CONSTRAINT chunk_chunk_name_unique IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE c.chunk_name IS UNIQUE",

            # align with MERGE (en:Entity {name: ...})
            "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.name IS UNIQUE",

            "CREATE CONSTRAINT assertion_id_unique IF NOT EXISTS "
            "FOR (a:Assertion) REQUIRE a.id IS UNIQUE",

            "CREATE CONSTRAINT doccentroid_id_unique IF NOT EXISTS "
            "FOR (dc:DocCentroid) REQUIRE dc.id IS UNIQUE",

            # Type constraints
            "CREATE CONSTRAINT chunk_embedding_type IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE c.embedding IS :: LIST<FLOAT NOT NULL>",

            "CREATE CONSTRAINT assertion_embedding_type IF NOT EXISTS "
            "FOR (a:Assertion) REQUIRE a.embedding IS :: LIST<FLOAT NOT NULL>",

            "CREATE CONSTRAINT doccentroid_embedding_type IF NOT EXISTS "
            "FOR (dc:DocCentroid) REQUIRE dc.embedding IS :: LIST<FLOAT NOT NULL>",
        ]

        # --- Vector Indexes on embeddings (ANN) ---
        statements += [
            f"CREATE VECTOR INDEX chunk_embedding_ix "
            f"FOR (c:Chunk) ON c.embedding "
            f"OPTIONS {{ indexConfig: {{ `vector.dimensions`: {dim}, "
            f"`vector.similarity_function`: '{sim}' }} }}",

            f"CREATE VECTOR INDEX assertion_embedding_ix "
            f"FOR (a:Assertion) ON a.embedding "
            f"OPTIONS {{ indexConfig: {{ `vector.dimensions`: {dim}, "
            f"`vector.similarity_function`: '{sim}' }} }}",

            f"CREATE VECTOR INDEX doc_centroid_embedding_ix "
            f"FOR (dc:DocCentroid) ON dc.embedding "
            f"OPTIONS {{ indexConfig: {{ `vector.dimensions`: {dim}, "
            f"`vector.similarity_function`: '{sim}' }} }}",
        ]

        # --- Full-text indexes (STRING or LIST<STRING> props) ---
        statements += [
            "CREATE FULLTEXT INDEX fts_chunks IF NOT EXISTS "
            "FOR (c:Chunk) ON EACH [c.text]",

            # includes d.source which we now set in persist()
            "CREATE FULLTEXT INDEX fts_documents IF NOT EXISTS "
            "FOR (d:Document) ON EACH [d.title, d.source, d.author, d.text]",

            "CREATE FULLTEXT INDEX fts_entities IF NOT EXISTS "
            "FOR (e:Entity) ON EACH [e.name, e.aliases]",

            "CREATE FULLTEXT INDEX fts_assertions IF NOT EXISTS "
            "FOR (a:Assertion) ON EACH [a.predicate]",
        ]

        # Wait for all indexes/constraints
        statements += ["CALL db.awaitIndexes()"]

        return statements

    def cleandb(self):
        """
        Remove all GraphRAG data while preserving the schema (constraints/indexes).
        After this, the database is effectively empty for our labels and ready for `persist`.
        """
        driver = self.env.get_driver()

        def _wipe(tx):
            # Delete in an order that typically reduces detach overhead.
            tx.run("MATCH (a:Assertion) DETACH DELETE a")
            tx.run("MATCH (c:Chunk) DETACH DELETE c")
            tx.run("MATCH (d:Document) DETACH DELETE d")
            tx.run("MATCH (dc:DocCentroid) DETACH DELETE dc")
            tx.run("MATCH (e:Entity) DETACH DELETE e")

        with driver.session() as session:
            session.execute_write(_wipe)

