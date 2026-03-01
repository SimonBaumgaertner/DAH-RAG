from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Tuple
from common.data_classes.rag_system import Chunk
from common.logging.run_logger import RunLogger
from common.neo4j.data_classes import Executor
from common.neo4j.standard_executor import StandardExecutor
from common.strategies.graph_search.abstract_graph_search import GraphSearch
from common.strategies.graph_search.vector_graph_search import VectorGraphSearch
from common.strategies.named_entity_recognition import DistilBertNER, NERStrategy


class DocumentAwareDenseGraphSearch(GraphSearch):
    """Graph search strategy that combines multiple scoring components.

    The retrieval process builds a subgraph using dense and entity search and
    then ranks candidate chunks.  The final ranking is a weighted combination of
    lexical matching, embedding similarity and entity overlap.  The weights can
    be adjusted to emphasise different aspects of the ranking.  For example,
    increasing ``lexical_weight`` favours literal keyword matches while a higher
    ``dense_weight`` prioritises semantic similarity.

    Document-Aware Hybrid RAG works in multiple stages
    1. Graph-Building. The idea is to first have a subgraph of controllable size that we then can do more expensive search on
    1.1 Dense Search
    1.1.1 First we embed the query using our encoder into vector space
    1.1.2 We then match this vector with the embedding Vectors of the Documents (centroids)
    1.1.3 High similarity with a centroid signifies high relevance of that document
    1.2 Entity Search
    1.2.1 We extract Named-Entities from the query using a NER strategy
    1.2.2 We then add every Assertion + Chunk linked with that Entities for our subgraph
    1.2.3 We can then use the candidate list and the lexical search (with Query Entities)to find the most relevant documents (high preference on lexical, but taking into consideration the amount of Entities in a Document to not advantage large docs)
    1.3 Build the Subgraph using
        - 1. The Chunks and Assertions connected to Entities from the query
        - 2. the Chunks, Entities and Assertions from the top chosen documents
        We will add all from 1. and at least one document and add documents until a threshold of Chunks (for now 300) is reached (add no more docs after)
    2. Graph Search
    2.1 For now just do a Dense Search on the subgraph
    """

    def __init__(
        self,
        *,
        executor: StandardExecutor,
        log: RunLogger,
        ner: NERStrategy | None = None,
        chunk_limit: int = 300,
        lexical_weight: float = 1.0,
        dense_weight: float = 0.1,
        entity_weight: float = 0.3,
    ) -> None:
        self.executor = executor
        self.ner = ner or DistilBertNER()
        self.chunk_limit = chunk_limit
        self.log = log
        self.lexical_weight = lexical_weight
        self.dense_weight = dense_weight
        self.entity_weight = entity_weight

    # ------------------------------------------------------------------
    def _encode(self, text: str) -> List[float]:
        emb = self.executor.encoder.encode(text, query=True)
        return emb.tolist() if hasattr(emb, "tolist") else list(emb)

    # ------------------------------------------------------------------
    def _is_valid_entity(self, entity: str) -> bool:
        """Check if an entity is valid for use in Neo4j fulltext search queries.
        
        Filters out:
        - Single punctuation characters
        - Very short fragments (< 2 alphanumeric characters)
        - Empty or whitespace-only strings
        - Strings that are only punctuation
        """
        if not entity or not entity.strip():
            return False
        
        # Exclude Lucene boolean operators which can cause syntax errors if treated as terms
        if entity.upper() in {"AND", "OR", "NOT"}:
            return False
        
        alnum_chars = ''.join(c for c in entity if c.isalnum())
        if len(alnum_chars) < 2:
            return False
        
        stripped = entity.strip()
        if all(not c.isalnum() for c in stripped):
            return False
        
        return True

    # ------------------------------------------------------------------
    def _escape_lucene_query(self, query: str) -> str:
        """Escape special Lucene characters in a query string.
        
        Lucene special characters that need escaping:
        + - && || ! ( ) { } [ ] ^ " ~ * ? : \ /
        """
        # Characters that need to be escaped with backslash in Lucene
        lucene_special_chars = r'\+-&|!(){}[]^"~*?:\\/'
        # Escape each special character
        escaped = re.sub(rf'([{re.escape(lucene_special_chars)}])', r'\\\1', query)
        return escaped

    # ------------------------------------------------------------------
    def search(self, query: str, k: int) -> List[Chunk]:
        driver = self.executor.env.get_driver()
        embedding = self._encode(query)

        self.log.info(f"🔍 Starting search for query: {query}")

        # --------------------------------------------------------------
        # 1.1 Dense search over document centroids to get candidate docs
        # --------------------------------------------------------------
        doc_dense_scores: Dict[str, float] = {}
        cypher_dense = (
            "CALL db.index.vector.queryNodes('doc_centroid_embedding_ix', $limit, $embedding) "
            "YIELD node, score "
            "MATCH (d:Document)-[:HAS_CENTROID]->(node) "
            "RETURN d.id AS doc_id, score "
            "ORDER BY score DESC"
        )
        with driver.session() as session:
            for record in session.run(cypher_dense, limit=20, embedding=embedding):
                doc_id = record["doc_id"]
                score = float(record["score"])
                if score > doc_dense_scores.get(doc_id, float("-inf")):
                    doc_dense_scores[doc_id] = score

            self.log.info(f"📊 Dense candidate docs: {doc_dense_scores}")

            # ----------------------------------------------------------
            # 1.2 Entity search
            # ----------------------------------------------------------
            all_entities = [e for e, _ in self.ner.extract_entities(query)]
            entities = [e for e in all_entities if self._is_valid_entity(e)]
            
            if len(entities) < len(all_entities):
                filtered = set(all_entities) - set(entities)
                self.log.info(f"🚫 Filtered out invalid entities: {filtered}")
            
            self.log.info(f"🧩 Extracted entities: {entities}")

            entity_chunks: Dict[str, Tuple[str, str]] = {}
            doc_lexical_scores: Dict[str, float] = defaultdict(float)
            doc_entity_hits: Dict[str, set[str]] = defaultdict(set)

            for ent in entities:
                self.log.info(f"🔗 Searching for entity: {ent}")

                # Gather chunks and assertions linked with entity
                cypher_chunks = """
                MATCH (e:Entity)
                WHERE toLower(e.name) = toLower($name)
                MATCH (a:Assertion)-[:SUBJECT|OBJECT]->(e)
                MATCH (a)-[:DERIVED_FROM]->(c:Chunk)<-[:HAS_CHUNK]-(d:Document)
                RETURN DISTINCT c.chunk_name AS chunk_id, c.text AS text, d.id AS doc_id
                """
                for r in session.run(cypher_chunks, name=ent):
                    cid = r["chunk_id"]
                    entity_chunks[cid] = (r["text"], r["doc_id"])
                    doc_entity_hits[r["doc_id"]].add(ent.lower())

                # Lexical search for documents mentioning entity
                cypher_lex = """
                CALL db.index.fulltext.queryNodes('fts_chunks', $q) YIELD node, score
                MATCH (d:Document)-[:HAS_CHUNK]->(node)
                RETURN d.id AS doc_id, score
                ORDER BY score DESC
                LIMIT 20
                """
                escaped_ent = self._escape_lucene_query(ent)
                for r in session.run(cypher_lex, q=escaped_ent):
                    doc_lexical_scores[r["doc_id"]] += float(r["score"])
                    doc_entity_hits[r["doc_id"]].add(ent.lower())

            self.log.info(f"📚 Lexical scores: {dict(doc_lexical_scores)}")
            self.log.info(f"🧬 Entity hits per doc: {dict(doc_entity_hits)}")

        # --------------------------------------------------------------
        # 1.3 Build the subgraph
        # --------------------------------------------------------------
        candidate_docs = set(doc_dense_scores) | set(doc_lexical_scores)
        doc_rank: List[Tuple[str, float]] = []
        for doc in candidate_docs:
            score = self._combine_scores(
                doc_lexical_scores.get(doc, 0.0),
                doc_dense_scores.get(doc, 0.0),
                len(doc_entity_hits.get(doc, set())),
            )
            doc_rank.append((doc, score))
        doc_rank.sort(key=lambda x: x[1], reverse=True)

        self.log.info(f"🏆 Ranked candidate docs: {doc_rank}")

        selected_chunks: Dict[str, Tuple[str, str]] = dict(entity_chunks)
        total_chunks = len(selected_chunks)

        with driver.session() as session:
            for idx, (doc_id, score) in enumerate(doc_rank):
                self.log.info(f"📄 Adding chunks from doc {doc_id} (score={score:.3f})")
                cypher_doc_chunks = """
                MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
                RETURN c.chunk_name AS chunk_id, c.text AS text, d.id AS doc_id
                """
                rows = list(session.run(cypher_doc_chunks, doc_id=doc_id))
                if not rows:
                    self.log.info(f"⚠️ No chunks found for doc {doc_id}")
                    continue
                # ensure at least one document is considered
                if total_chunks >= self.chunk_limit and idx > 0:
                    break
                for row in rows:
                    cid = row["chunk_id"]
                    if cid in selected_chunks:
                        continue
                    selected_chunks[cid] = (row["text"], row["doc_id"])
                    total_chunks += 1
                    if total_chunks >= self.chunk_limit and idx > 0:
                        break
                if total_chunks >= self.chunk_limit:
                    break

        self.log.info(f"🗂️ Total chunks selected: {len(selected_chunks)}")

        if not selected_chunks:
            self.log.info("⚡ No chunks selected, falling back to global vector search")
            return VectorGraphSearch(self.executor).search(query, k)

        chunk_ids = list(selected_chunks.keys())

        # --------------------------------------------------------------
        # 2. Dense search on the collected subgraph chunks
        # --------------------------------------------------------------
        self.log.info(f"🔎 Running final dense search on {len(chunk_ids)} chunks")

        with driver.session() as session:
            cypher_final = """
            MATCH (c:Chunk)
            WHERE c.chunk_name IN $chunk_ids
            MATCH (d:Document)-[:HAS_CHUNK]->(c)
            WITH c, d, gds.similarity.cosine(c.embedding, $embedding) AS score
            RETURN c.chunk_name AS chunk_id, c.text AS text, score AS score, d.id AS doc_id
            ORDER BY score DESC
            LIMIT $k
            """
            records = session.run(cypher_final, chunk_ids=chunk_ids, embedding=embedding, k=k)
            results = [
                Chunk(
                    chunk_id=r["chunk_id"],
                    text=r["text"],
                    score=r["score"],
                    doc_id=r["doc_id"],
                )
                for r in records
            ]

        self.log.info(f"✅ Returning {len(results)} chunks for query '{query}'")
        return results

    def _combine_scores(
        self, lexical_score: float, dense_score: float, entity_score: float
    ) -> float:
        """Combine scoring components using the configured weights."""

        return (
            self.lexical_weight * lexical_score
            + self.dense_weight * dense_score
            + self.entity_weight * entity_score
        )
