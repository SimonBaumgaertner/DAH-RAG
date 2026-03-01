from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections import defaultdict
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
from common.data_classes.qa import QuestionAnswerPair
from common.data_classes.rag_system import Chunk
from common.logging.run_logger import RunLogger
from common.neo4j.data_classes import Executor
from common.neo4j.standard_executor import StandardExecutor
from common.strategies.graph_search.abstract_graph_search import GraphSearch
from common.strategies.graph_search.chunk_rankings import ChunkRankings, ChunkScore
from common.strategies.graph_search.document_rankings import DocumentRankings
from common.strategies.graph_search.vector_graph_search import VectorGraphSearch
from common.strategies.named_entity_recognition import DistilBertNER, NERStrategy


class DocumentAwareExperimentalGraphSearch(GraphSearch):
    """Graph search strategy that utilizes a two-stage "Filter & Rank" architecture.

    The retrieval process first builds a targeted subgraph by filtering for high-relevance
    documents, then performs a graph-based ranking on the specific chunks within that
    subgraph. The scoring logic is decoupled, allowing for distinct weights during
    document selection (Filtering) versus final chunk scoring (Ranking).

    Document-Aware Hybrid RAG works in two distinct stages:

    1. Graph-Building (Filtering). The goal is to identify and load the most relevant
       documents to construct a subgraph of controllable size. Documents are scored
       and selected using a weighted combination of:
       1.1 Global Dense Search: Matches the query vector against Document Centroids.
       1.2 Global Lexical Search: Matches query keywords against the full-text index
           to capture specific terms or IDs.
       1.3 Entity Overlap: Boosts documents that explicitly contain Named Entities
           extracted from the user query.

    2. Subgraph Ranking. Once the subgraph (Chunks + Assertions + Entities) is projected,
       we identify the top-k specific chunks using a weighted combination of:
       2.1 Personalized PageRank (PPR): Uses Entities and high-similarity chunks as
           seeds to propagate importance through the graph topology.
       2.2 Local Dense Score: Semantic similarity between the query and the specific chunk.
       2.3 Local Lexical Score: Keyword relevance of the specific chunk text.
       2.4 Return the top-k chunks based on the scores and weights

    Parameters
    ----------
    executor : StandardExecutor
        Neo4j executor for database operations.
    log : RunLogger
        Logger for tracking operations.
    ner : NERStrategy, optional
        Named entity recognition strategy. Defaults to DistilBertNER().
    chunk_limit : int, default=1000
        Maximum number of chunks to include in the projected subgraph.

    Filtering Weights (Stage 1 - Document Selection)
    ------------------------------------------------
    filter_lexical_weight : float
        Weight for global keyword matching (BM25) when selecting candidate documents.
    filter_dense_weight : float
        Weight for vector similarity between query and Document Centroids.
    filter_entity_weight : float
        Weight for the count of query entities found within a document.

    Ranking Weights (Stage 2 - Chunk Scoring)
    -----------------------------------------
    rank_ppr_weight : float
        Weight for the Graph Topology score (PageRank). This is typically set higher
        to normalize against raw probability values.
    rank_dense_weight : float
        Weight for vector similarity between the query and specific Chunks.
    rank_lexical_weight : float
        Weight for keyword matching on the specific chunk text.
    """

    def __init__(
            self,
            *,
            executor: StandardExecutor,
            log: RunLogger,
            ner: NERStrategy | None = None,
            chunk_limit: int = 1000,

            filter_lexical_weight: float =   0.0545,  # BM25 / Keyword
            filter_dense_weight: float = 10.0,  # Vector / Centroid
            filter_entity_weight: float = 0.6871,  # Count of Entity Hits
            rank_ppr_weight: float = 0.2893,  # How much we trust the Graph Topology
            rank_dense_weight: float = 10.0000,  # How much we trust Vector Similarity
            rank_lexical_weight: float = 0.2369  # BM25 / Keyword
    ) -> None:
        self.executor = executor
        self.ner = ner or DistilBertNER()
        self.chunk_limit = chunk_limit
        self.log = log

        # Filtering Stage
        self.filter_lexical_weight = filter_lexical_weight
        self.filter_dense_weight = filter_dense_weight
        self.filter_entity_weight = filter_entity_weight

        # Ranking Stage
        self.rank_ppr_weight = rank_ppr_weight
        self.rank_dense_weight = rank_dense_weight
        self.rank_lexical_weight = rank_lexical_weight

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
        # Characters that need to be escaped with backslash in Lucene
        lucene_special_chars = r'\+-&|!(){}[]^"~*?:\\/'
        # Escape each special character
        escaped = re.sub(rf'([{re.escape(lucene_special_chars)}])', r'\\\1', query)
        return escaped

    # ------------------------------------------------------------------
    def _combine_filter_scores(
            self, lexical_filter_score: float, dense_filter_score: float, entity_filter_score: int
    ) -> float:
        """Combine scoring components for the Document Filtering stage (Stage 1)
        using the configured filter weights."""
        return (
                self.filter_lexical_weight * lexical_filter_score
                + self.filter_dense_weight * dense_filter_score
                + self.filter_entity_weight * entity_filter_score
        )

    # ------------------------------------------------------------------
    def _combine_rank_scores(
            self, ppr_rank_score: float, dense_rank_score: float, lexical_rank_score: float
    ) -> float:
        """Combine scoring components for the Subgraph Ranking stage (Stage 2)
        using the configured ranking weights."""
        return (
                self.rank_ppr_weight * ppr_rank_score
                + self.rank_dense_weight * dense_rank_score
                + self.rank_lexical_weight * lexical_rank_score
        )

    # ------------------------------------------------------------------
    def _save_rankings_json(
        self,
        query: str,
        qa_pair: QuestionAnswerPair,
        document_rankings: DocumentRankings,
        chunk_rankings: ChunkRankings
    ) -> None:
        """Save rankings data to JSON file for weight optimization.
        
        Creates a JSON file in the rankings folder with:
        - The question
        - Correct document IDs from the QA pair proofs
        - All filter and rank weights
        - Document scores (from DocumentScore.get_json())
        - Chunk scores (from ChunkScore.get_json())
        """
        # Create rankings folder if it doesn't exist
        rankings_dir = Path(__file__).parent / "rankings"
        rankings_dir.mkdir(exist_ok=True)

        # Create subfolder for the experiment (run_id)
        run_id = self.log.run_id if hasattr(self.log, "run_id") else "unknown_run"
        experiment_dir = rankings_dir / run_id
        experiment_dir.mkdir(exist_ok=True)
        
        # Hash the question to create a unique filename
        question_hash = hashlib.sha256(query.encode('utf-8')).hexdigest()
        json_path = experiment_dir / f"{question_hash}.json"
        
        # Get correct document IDs from proofs
        correct_document_ids = [proof.document_id for proof in qa_pair.proofs]
        
        # Get document scores as JSON
        document_scores = [doc_score.get_json() for doc_score in document_rankings.get_document_rankings()]
        
        # Get chunk scores as JSON
        chunk_scores = [chunk_score.get_json() for chunk_score in chunk_rankings.get_chunk_rankings()]
        
        # Create the JSON data structure
        rankings_data = {
            "question": query,
            "correct_documents": correct_document_ids,
            "weights": {
                "filter_lexical_weight": self.filter_lexical_weight,
                "filter_dense_weight": self.filter_dense_weight,
                "filter_entity_weight": self.filter_entity_weight,
                "rank_ppr_weight": self.rank_ppr_weight,
                "rank_dense_weight": self.rank_dense_weight,
                "rank_lexical_weight": self.rank_lexical_weight
            },
            "document_scores": document_scores,
            "chunk_scores": chunk_scores
        }
        
        # Write JSON file (overwrite if exists)
        with json_path.open('w', encoding='utf-8') as f:
            json.dump(rankings_data, f, indent=2, ensure_ascii=False)
        
        self.log.info(f"💾 Saved rankings data to {json_path}")

    # ------------------------------------------------------------------
    def search(self, query: str, k: int, qa_pair: Optional[QuestionAnswerPair] = None) -> List[Chunk]:
        timings: Dict[str, float] = {}
        t_start_total = time.perf_counter()

        driver = self.executor.env.get_driver()
        embedding = self._encode(query)

        self.log.info(f"🔍 Starting search for query: {query}")

        # --------------------------------------------------------------
        # 1.1 Dense search over document centroids to get candidate docs
        # --------------------------------------------------------------
        t0 = time.perf_counter()
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
            timings["document_filter_dense_search"] = time.perf_counter() - t0

            # ----------------------------------------------------------
            # 1.2 Entity search and global lexical search
            # ----------------------------------------------------------
            t0 = time.perf_counter()
            all_entities = [e for e, _ in self.ner.extract_entities(query)]
            entities = [e for e in all_entities if self._is_valid_entity(e)]

            if len(entities) < len(all_entities):
                filtered = set(all_entities) - set(entities)
                self.log.info(f"🚫 Filtered out invalid entities: {filtered}")

            self.log.info(f"🧩 Extracted entities: {entities}")

            entity_chunks: Dict[str, Tuple[str, str]] = {}
            doc_lexical_scores: Dict[str, float] = defaultdict(float)
            doc_entity_hits: Dict[str, set[str]] = defaultdict(set)
            
            # Use query terms for a general lexical search to augment entity-based search
            query_terms = [t for t in re.split(r'\s+', query) if self._is_valid_entity(t)]
            search_terms = list(dict.fromkeys(entities + query_terms)) # keep order, de-duplicate
            
            # OPTIMIZATION: Batch entity chunk lookup instead of querying per term
            if entities:
                cypher_chunks_batch = """
                MATCH (e:Entity)
                WHERE toLower(e.name) IN $entity_names
                MATCH (a:Assertion)-[:SUBJECT|OBJECT]->(e)
                MATCH (a)-[:DERIVED_FROM]->(c:Chunk)<-[:HAS_CHUNK]-(d:Document)
                RETURN DISTINCT c.chunk_name AS chunk_id, c.text AS text, d.id AS doc_id, toLower(e.name) AS entity_name
                """
                for r in session.run(cypher_chunks_batch, entity_names=[e.lower() for e in entities]):
                    cid = r["chunk_id"]
                    entity_chunks[cid] = (r["text"], r["doc_id"])
                    doc_entity_hits[r["doc_id"]].add(r["entity_name"])

            # OPTIMIZATION: Batch lexical search for all terms at once
            for term in search_terms:
                escaped_term = self._escape_lucene_query(term)
                cypher_lex = """
                CALL db.index.fulltext.queryNodes('fts_documents', $q) YIELD node, score
                RETURN node.id AS doc_id, score
                ORDER BY score DESC
                LIMIT 100
                """
                for r in session.run(cypher_lex, q=escaped_term):
                    doc_id = r["doc_id"]
                    # Sum scores from multiple query terms/entities
                    doc_lexical_scores[doc_id] += float(r["score"])
                    
                    # Track entity hits for the entity boost score
                    if term in entities:
                        doc_entity_hits[doc_id].add(term.lower())

            # self.log.info(f"📚 Global Lexical scores: {dict(doc_lexical_scores)}")
            # self.log.info(f"🧬 Entity hits per doc: {dict(doc_entity_hits)}")

        # --------------------------------------------------------------
        # 1.3 Combine scores and build the subgraph using DocumentRankings
        # --------------------------------------------------------------
        timings["document_filter_lexical_entity_search"] = time.perf_counter() - t0
        
        document_rankings = DocumentRankings(query)
        candidate_docs = set(doc_dense_scores) | set(doc_lexical_scores)
        
        # Add all candidate documents with their scores
        for doc_id in candidate_docs:
            entity_hit_count = len(doc_entity_hits.get(doc_id, set()))
            document_rankings.add_document_ranking(
                document_id=doc_id,
                lexical_score=doc_lexical_scores.get(doc_id, 0.0),
                dense_score=doc_dense_scores.get(doc_id, 0.0),
                entity_score=entity_hit_count
            )

        # Calculate final document scores
        document_rankings.calculate_document_rankings(
            self.filter_lexical_weight,
            self.filter_dense_weight,
            self.filter_entity_weight
        )
        
        ranked_docs = document_rankings.get_document_rankings()
        # self.log.info(f"🏆 Final ranked candidate docs: {[(doc.document_id, doc.score) for doc in ranked_docs]}")

        selected_chunks: Dict[str, Tuple[str, str]] = dict(entity_chunks)
        total_chunks = len(selected_chunks)

        # OPTIMIZATION: Batch fetch all chunks for ranked documents in one query
        with driver.session() as session:
            # Determine which documents to fetch based on chunk limit
            docs_to_fetch = []
            for idx, doc_score in enumerate(ranked_docs):
                if total_chunks >= self.chunk_limit and idx > 0:
                    break
                docs_to_fetch.append(doc_score.document_id)
            
            if docs_to_fetch:
                # Single query to fetch all chunks from all documents
                # UNWIND preserves the order of the input list, ensuring chunks are fetched doc-by-doc in order
                cypher_all_chunks = """
                UNWIND $doc_ids AS doc_id
                MATCH (d:Document {id: doc_id})-[:HAS_CHUNK]->(c:Chunk)
                RETURN c.chunk_name AS chunk_id, c.text AS text, d.id AS doc_id
                """
                
                for row in session.run(cypher_all_chunks, doc_ids=docs_to_fetch):
                    cid = row["chunk_id"]
                    if cid not in selected_chunks:
                        selected_chunks[cid] = (row["text"], row["doc_id"])
                        total_chunks += 1
                        if total_chunks >= self.chunk_limit:
                            break

        self.log.info(f"🗂️ Total chunks selected: {len(selected_chunks)}")

        if not selected_chunks:
            self.log.warning("⚠️ No chunks selected - no documents matched the query. This should pretty much NEVER happen!")
            return []

        chunk_ids = list(selected_chunks.keys())

        # --------------------------------------------------------------
        # 2. Local Dense and Lexical Scores for Ranking
        # --------------------------------------------------------------
        # Create ChunkRankings early and populate with dense/lexical scores
        chunk_rankings = ChunkRankings(query)
        chunk_el_ids: Dict[str, str] = {}
        
        with driver.session() as session:
            t0 = time.perf_counter()
            # Local Dense scores for chunks
            cypher_chunk_dense = """
            MATCH (c:Chunk)
            WHERE c.chunk_name IN $chunk_ids
            RETURN c.chunk_name AS chunk_id,
                   elementId(c) AS el_id,
                   gds.similarity.cosine(c.embedding, $embedding) AS score
            """
            for r in session.run(cypher_chunk_dense, chunk_ids=chunk_ids, embedding=embedding):
                cid = r["chunk_id"]
                chunk_el_ids[cid] = r["el_id"]
                # Add chunk with dense score, lexical and PPR will be set later
                chunk_rankings.add_chunk_ranking(
                    chunk_id=cid,
                    ppr_score=0.0,
                    dense_score=float(r["score"]),
                    lexical_score=0.0
                )

            # Local Lexical search for chunks
            escaped_query = self._escape_lucene_query(query)
            cypher_local_lex = """
            CALL db.index.fulltext.queryNodes('fts_chunks', $q) YIELD node, score
            WHERE node.chunk_name IN $chunk_ids
            RETURN node.chunk_name AS chunk_id, score
            ORDER BY score DESC
            """
            # Update lexical scores in existing ChunkScore objects
            for r in session.run(cypher_local_lex, q=escaped_query, chunk_ids=chunk_ids):
                cid = r["chunk_id"]
                # Find existing ChunkScore and update lexical score
                for chunk_score in chunk_rankings.get_chunk_rankings():
                    if chunk_score.chunk_id == cid:
                        chunk_score.lexical_score = float(r["score"])
                        break
                else:
                    # Chunk not in dense results, add it now
                    chunk_rankings.add_chunk_ranking(
                        chunk_id=cid,
                        ppr_score=0.0,
                        dense_score=0.0,
                        lexical_score=float(r["score"])
                    )

            # Ensure all chunks from chunk_ids have ChunkScore objects
            existing_chunk_ids = {cs.chunk_id for cs in chunk_rankings.get_chunk_rankings()}
            for cid in chunk_ids:
                if cid not in existing_chunk_ids:
                    chunk_rankings.add_chunk_ranking(
                        chunk_id=cid,
                        ppr_score=0.0,
                        dense_score=0.0,
                        lexical_score=0.0
                    )

            lexical_scores_log = {
                cs.chunk_id: cs.lexical_score 
                for cs in chunk_rankings.get_chunk_rankings() 
                if cs.lexical_score > 0.0
            }
            #self.log.info(f"📝 Local Lexical scores for chunks: {lexical_scores_log}")
            timings["ranking_dense_lexical_scoring"] = time.perf_counter() - t0

        # --------------------------------------------------------------
        # 3. Seed Selection & PPR Execution (New Session Block)
        # --------------------------------------------------------------
        
        with driver.session() as session:
            t0 = time.perf_counter()
            
            # --- 3.1 Entity Seeds ---
            cypher_entity_seeds = """
            MATCH (e:Entity)<-[:SUBJECT|OBJECT]-(a:Assertion)-[:DERIVED_FROM]->(c:Chunk)
            WHERE c.chunk_name IN $chunk_ids AND toLower(e.name) IN $entities
            RETURN elementId(e) AS el_id
            """
            entity_seed_ids = [
                r["el_id"]
                for r in session.run(
                    cypher_entity_seeds,
                    chunk_ids=chunk_ids,
                    entities=[e.lower() for e in entities],
                )
            ]

            # --- 3.2 Fallback Seeds ---
            seed_el_ids = entity_seed_ids
            
            if not seed_el_ids:
                self.log.info("⚠️ No entity seeds found. Falling back to Vector Chunks for PPR seeds.")
                
                chunk_score_lookup = {
                    cs.chunk_id: cs.dense_score 
                    for cs in chunk_rankings.get_chunk_rankings()
                }
                
                # Fetch element IDs for the chunks we already found in stage 1
                cypher_fallback = """
                MATCH (c:Chunk) WHERE c.chunk_name IN $chunk_ids
                RETURN elementId(c) as el_id, c.chunk_name as cid
                """
                fallback_rows = list(session.run(cypher_fallback, chunk_ids=chunk_ids))
                
                # Sort by the dense score
                fallback_rows.sort(
                    key=lambda row: chunk_score_lookup.get(row["cid"], 0.0), 
                    reverse=True
                )
                seed_el_ids = [row["el_id"] for row in fallback_rows[:k]]
            else:
                self.log.info(f"🌱 PPR Seeding (Using {len(seed_el_ids)} Entities)")

            # --- 3.3 Project Graph (Entity-Centric) ---
            graph_name = f"ppr_subgraph_{uuid.uuid4().hex[:8]}"

            # Flattened Topology: (Chunk)-[MENTIONS]-(Entity)
            node_query = """
            MATCH (c:Chunk) WHERE c.chunk_name IN $chunk_ids 
            RETURN id(c) AS id, labels(c) AS labels
            UNION
            MATCH (e:Entity)
            WHERE EXISTS {
                MATCH (e)<-[:SUBJECT|OBJECT]-(:Assertion)-[:DERIVED_FROM]->(c:Chunk)
                WHERE c.chunk_name IN $chunk_ids
            }
            RETURN id(e) AS id, labels(e) AS labels
            """

            rel_query = """
            MATCH (c:Chunk)<-[:DERIVED_FROM]-(:Assertion)-[:SUBJECT|OBJECT]->(e:Entity)
            WHERE c.chunk_name IN $chunk_ids
            RETURN id(c) AS source, id(e) AS target, 'MENTIONS' AS type, 'UNDIRECTED' AS orientation
            """

            session.run(
                """
                CALL gds.graph.project.cypher(
                    $graph_name,
                    $node_query,
                    $rel_query,
                    {parameters: {chunk_ids: $chunk_ids}}
                )
                YIELD graphName
                """,
                graph_name=graph_name,
                node_query=node_query,
                rel_query=rel_query,
                chunk_ids=chunk_ids,
            ).consume()

            # --- 3.4 Run PPR & AGGREGATE SCORES ---
            # Calculates the importance of chunks based on the sum of their contained entities' PPR scores.
            cypher_ppr_agg = """
            MATCH (n) WHERE elementId(n) IN $seed_ids
            WITH collect(n) AS seeds
            CALL gds.pageRank.stream($graph_name, {
                sourceNodes: seeds,
                dampingFactor: 0.85,
                maxIterations: 20
            })
            YIELD nodeId, score
            WITH gds.util.asNode(nodeId) AS n, score
            WHERE n:Entity AND score > 0
            
            MATCH (n)<-[:SUBJECT|OBJECT]-(:Assertion)-[:DERIVED_FROM]->(c:Chunk)
            WHERE c.chunk_name IN $chunk_ids
            
            RETURN c.chunk_name AS chunk_id, sum(score) AS ppr_score
            ORDER BY ppr_score DESC
            """

            ppr_records = list(
                session.run(
                    cypher_ppr_agg,
                    graph_name=graph_name,
                    seed_ids=seed_el_ids,
                    chunk_ids=chunk_ids  # <--- THIS WAS MISSING
                )
            )

            # --- 3.5 Cleanup ---
            session.run(
                "CALL gds.graph.drop($graph_name)", graph_name=graph_name
            ).consume()
            timings["ranking_ppr_scoring"] = time.perf_counter() - t0

        # --------------------------------------------------------------
        # 4. Combine PPR score with dense and lexical scores
        # --------------------------------------------------------------
        chunk_score_map: Dict[str, ChunkScore] = {
            cs.chunk_id: cs for cs in chunk_rankings.get_chunk_rankings()
        }
        
        t0 = time.perf_counter()
        for r in ppr_records:
            cid = r["chunk_id"]
            ppr_score = float(r["ppr_score"])
            
            if cid in chunk_score_map:
                chunk_score_map[cid].ppr_score = ppr_score
            else:
                chunk_rankings.add_chunk_ranking(
                    chunk_id=cid,
                    ppr_score=ppr_score,
                    dense_score=0.0,
                    lexical_score=0.0
                )
                chunk_score_map[cid] = chunk_rankings.get_chunk_rankings()[-1]
        
        ranked_chunk_scores = chunk_rankings.calculate_chunk_rankings(
            self.rank_ppr_weight,
            self.rank_dense_weight,
            self.rank_lexical_weight
        )

        # Create Chunk objects...
        results = []
        for chunk_score in ranked_chunk_scores[:k]:
            cid = chunk_score.chunk_id
            if cid in selected_chunks:
                text, doc_id = selected_chunks[cid]
            else:
                # Need to fetch text for purely graph-discovered chunks
                continue 
            
            results.append(
                Chunk(
                    chunk_id=cid,
                    text=text,
                    score=chunk_score.score,
                    doc_id=doc_id,
                    chunk_scores=chunk_score
                )
            )

        # Save rankings JSON for weight optimization
        if qa_pair is not None:
            self._save_rankings_json(
                query=query,
                qa_pair=qa_pair,
                document_rankings=document_rankings,
                chunk_rankings=chunk_rankings
            )

        self.log.info(f"✅ Returning {len(results)} chunks for query '{query}'")
        timings["ranking_final_selection"] = time.perf_counter() - t0
        timings["total_search_time"] = time.perf_counter() - t_start_total

        # Save timings to JSON
        try:
            timing_record = {
                "query": query,
                "timings": timings
            }
            run_id = self.log.run_id if hasattr(self.log, "run_id") else "unknown_run"
            timing_dir = Path(__file__).parent / "timing"
            timing_dir.mkdir(parents=True, exist_ok=True)
            timing_file = timing_dir / f"{run_id}.json"

            existing_data = []
            if timing_file.exists():
                with timing_file.open("r", encoding="utf-8") as f:
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        pass
            
            if not isinstance(existing_data, list):
                existing_data = []
            
            existing_data.append(timing_record)
            
            with timing_file.open("w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.log.error(f"Failed to save timing data: {e}")

        # Log timing summary in one line
        self.log.info(
            "⏱️  GRAPH_SEARCH TIMING: TOTAL=%.3fs | Doc_Dense=%.3fs (%.1f%%) | Doc_Lex_Entity=%.3fs (%.1f%%) | "
            "Rank_Dense_Lex=%.3fs (%.1f%%) | Rank_PPR=%.3fs (%.1f%%) | Rank_Final=%.3fs (%.1f%%)",
            timings['total_search_time'],
            timings.get('document_filter_dense_search', 0.0),
            100 * timings.get('document_filter_dense_search', 0.0) / timings['total_search_time'] if timings['total_search_time'] > 0 else 0,
            timings.get('document_filter_lexical_entity_search', 0.0),
            100 * timings.get('document_filter_lexical_entity_search', 0.0) / timings['total_search_time'] if timings['total_search_time'] > 0 else 0,
            timings.get('ranking_dense_lexical_scoring', 0.0),
            100 * timings.get('ranking_dense_lexical_scoring', 0.0) / timings['total_search_time'] if timings['total_search_time'] > 0 else 0,
            timings.get('ranking_ppr_scoring', 0.0),
            100 * timings.get('ranking_ppr_scoring', 0.0) / timings['total_search_time'] if timings['total_search_time'] > 0 else 0,
            timings.get('ranking_final_selection', 0.0),
            100 * timings.get('ranking_final_selection', 0.0) / timings['total_search_time'] if timings['total_search_time'] > 0 else 0
        )

        return results