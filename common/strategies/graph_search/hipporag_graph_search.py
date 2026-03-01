from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional, Tuple

# Assuming these exist in your project based on previous code
from common.data_classes.qa import QuestionAnswerPair
from common.data_classes.rag_system import Chunk
from common.logging.run_logger import RunLogger
from common.neo4j.standard_executor import StandardExecutor
from common.strategies.graph_search.abstract_graph_search import GraphSearch
from common.strategies.graph_search.chunk_rankings import ChunkRankings, ChunkScore
from common.strategies.graph_search.document_rankings import DocumentRankings
from common.strategies.named_entity_recognition import DistilBertNER, NERStrategy
from .document_aware_advanced_graph_search import DocumentAwareAdvancedGraphSearch

class HippoRAGGraphSearch(GraphSearch):
    """
    An adaptation of HippoRAG 2.0's retrieval mechanism.
    
    This strategy projects the full Triplet Knowledge Graph:
    (Subject)-[RELATION]->(Object)
    
    The PPR algorithm propagates probability through these semantic relations, 
    allowing the system to discover chunks that are "multi-hop" related to the 
    query even if they don't share exact keywords.
    """

    def __init__(
            self,
            *,
            executor: StandardExecutor,
            log: RunLogger,
            ner: NERStrategy | None = None,
            chunk_limit: int = 1000,
            
            # Adjusted weights for HippoRAG style (heavy on graph, low on dense)
            filter_lexical_weight: float =   0.0545,  # BM25 / Keyword
            filter_dense_weight: float = 10.0,  # Vector / Centroid
            filter_entity_weight: float = 0.6871,  # Count of Entity Hits
            rank_ppr_weight: float = 0.2893,  # How much we trust the Graph Topology
            rank_dense_weight: float = 10.0000,  # How much we trust Vector Similarity
            rank_lexical_weight: float = 0.2369  # BM25 / Keyword
    ) -> None:
        self.executor = executor
        self.log = log
        self.ner = ner or DistilBertNER()
        self.chunk_limit = chunk_limit
        
        # Filtering Stage
        self.filter_lexical_weight = filter_lexical_weight
        self.filter_dense_weight = filter_dense_weight
        self.filter_entity_weight = filter_entity_weight
        
        # Ranking Stage
        self.rank_ppr_weight = rank_ppr_weight
        self.rank_dense_weight = rank_dense_weight
        self.rank_lexical_weight = rank_lexical_weight

    def _encode(self, text: str) -> List[float]:
        """Encode text to embedding vector."""
        emb = self.executor.encoder.encode(text, query=True)
        return emb.tolist() if hasattr(emb, "tolist") else list(emb)

    def _is_valid_entity(self, entity: str) -> bool:
        """Check if an entity is valid for use in Neo4j queries."""
        if not entity or not entity.strip():
            return False
        if entity.upper() in {"AND", "OR", "NOT"}:
            return False
        alnum_chars = ''.join(c for c in entity if c.isalnum())
        if len(alnum_chars) < 2:
            return False
        stripped = entity.strip()
        if all(not c.isalnum() for c in stripped):
            return False
        return True

    def search(self, query: str, k: int, qa_pair: Optional[QuestionAnswerPair] = None) -> List[Chunk]:
        timings: Dict[str, float] = {}
        t_start_total = time.perf_counter()

        self.log.info(f"🦛 Starting HippoRAG Search for: {query}")
        driver = self.executor.env.get_driver()
        embedding = self._encode(query)

        # --------------------------------------------------------------
        # Phase 1: Contextual Subgraph Filtering
        # --------------------------------------------------------------
        # Reuse existing DocumentAware filtering logic to limit search space
        # --------------------------------------------------------------
        t0 = time.perf_counter()
        
        # 1.1 Dense Document Search
        doc_dense_scores: Dict[str, float] = {}
        with driver.session() as session:
            cypher_dense = (
                "CALL db.index.vector.queryNodes('doc_centroid_embedding_ix', 20, $embedding) "
                "YIELD node, score RETURN node.id AS doc_id, score"
            )
            for r in session.run(cypher_dense, embedding=embedding):
                doc_dense_scores[r["doc_id"]] = float(r["score"])

        # 1.2 Entity Extraction
        all_entities = [e for e, _ in self.ner.extract_entities(query)]
        entities = [e for e in all_entities if self._is_valid_entity(e)]
        
        # 1.3 Select Top Documents for Subgraph
        # Simple selection: Top 50 documents by dense score
        # (In a full implementation, you'd use the full weighted combination from parent)
        top_docs = sorted(doc_dense_scores.keys(), key=lambda x: doc_dense_scores[x], reverse=True)[:50]
        
        # 1.4 Fetch Chunks for Subgraph
        selected_chunks = {}  # Map chunk_id -> (text, doc_id)
        
        with driver.session() as session:
            if top_docs:
                cypher_get_chunks = """
                UNWIND $doc_ids AS doc_id
                MATCH (d:Document {id: doc_id})-[:HAS_CHUNK]->(c:Chunk)
                RETURN c.chunk_name as chunk_id, c.text as text, d.id as doc_id
                """
                for r in session.run(cypher_get_chunks, doc_ids=top_docs):
                    selected_chunks[r["chunk_id"]] = (r["text"], r["doc_id"])
        
        chunk_ids = list(selected_chunks.keys())
        timings["phase_1_filtering"] = time.perf_counter() - t0
        self.log.info(f"Construction Subgraph with {len(chunk_ids)} chunks from {len(top_docs)} docs")

        if not chunk_ids:
            self.log.info("⚠️ No chunks found in subgraph. Returning empty results.")
            return []

        # --------------------------------------------------------------
        # Phase 2: HippoRAG PPR (Projecting Triplet Graph)
        # --------------------------------------------------------------
        # Projects: (Entity)<-(Assertion)->(Entity)
        # --------------------------------------------------------------
        chunk_rankings = ChunkRankings(query)
        
        with driver.session() as session:
            t0 = time.perf_counter()

            # 2.1 Identify Seeds
            # Find entity nodes in graph matching our query entities
            cypher_get_seeds = """
            MATCH (e:Entity)
            WHERE toLower(e.name) IN $entities
            RETURN elementId(e) as el_id
            """
            seed_ids = [r["el_id"] for r in session.run(cypher_get_seeds, entities=[e.lower() for e in entities])]
            
            if not seed_ids:
                self.log.info("⚠️ No graph seeds found for extracted entities. Returning empty results.")
                return []

            # 2.2 Project Graph
            graph_name = f"hipporag_{uuid.uuid4().hex[:8]}"
            
            # Nodes: Assertions and Entities related to our chunks
            node_query = """
            MATCH (c:Chunk)<-[:DERIVED_FROM]-(a:Assertion)
            WHERE c.chunk_name IN $chunk_ids
            RETURN id(a) as id, labels(a) as labels
            UNION
            MATCH (c:Chunk)<-[:DERIVED_FROM]-(a:Assertion)-[:SUBJECT|OBJECT]->(e:Entity)
            WHERE c.chunk_name IN $chunk_ids
            RETURN DISTINCT id(e) as id, labels(e) as labels
            """

            # Edges: Connect Entity <-> Assertion (Undirected for associative flow)
            rel_query = """
            MATCH (c:Chunk)<-[:DERIVED_FROM]-(a:Assertion)-[r:SUBJECT|OBJECT]->(e:Entity)
            WHERE c.chunk_name IN $chunk_ids
            RETURN id(a) as source, id(e) as target, type(r) as type, 'UNDIRECTED' as orientation
            """

            try:
                session.run(
                    """
                    CALL gds.graph.project.cypher(
                        $graph_name, $node_query, $rel_query, 
                        {parameters: {chunk_ids: $chunk_ids}}
                    )
                    """,
                    graph_name=graph_name, node_query=node_query, rel_query=rel_query, chunk_ids=chunk_ids
                ).consume()

                # 2.3 Run PPR
                # Flow: Seed Entity -> Assertion -> Entity -> Assertion ...
                # We capture the score on Assertions (facts)
                cypher_run_ppr = """
                MATCH (s) WHERE elementId(s) IN $seed_ids
                WITH collect(s) as seeds
                CALL gds.pageRank.stream($graph_name, {
                    sourceNodes: seeds,
                    dampingFactor: 0.85,
                    maxIterations: 20
                })
                YIELD nodeId, score
                WITH gds.util.asNode(nodeId) as n, score
                WHERE score > 0 AND n:Assertion
                
                // 2.4 Aggregate to Chunks
                // Chunk score = sum of contained Fact scores
                MATCH (n:Assertion)-[:DERIVED_FROM]->(c:Chunk)
                RETURN c.chunk_name as chunk_id, sum(score) as ppr_score
                ORDER BY ppr_score DESC
                LIMIT $limit
                """
                
                ppr_results = list(session.run(cypher_run_ppr, 
                                             graph_name=graph_name, 
                                             seed_ids=seed_ids, 
                                             limit=k*2))
            finally:
                # Always cleanup graph
                session.run("CALL gds.graph.drop($graph_name, false)", graph_name=graph_name).consume()
                
            timings["phase_2_ppr"] = time.perf_counter() - t0

        # --------------------------------------------------------------
        # Phase 3: Final Scoring & Selection
        # --------------------------------------------------------------
        t0 = time.perf_counter()
        
        # Get dense scores for top PPR candidates to allow hybrid ranking
        top_ppr_ids = [r["chunk_id"] for r in ppr_results]
        chunk_dense_scores = {}
        
        with driver.session() as session:
            if top_ppr_ids:
                cypher_local_dense = """
                MATCH (c:Chunk) WHERE c.chunk_name IN $cids
                RETURN c.chunk_name as cid, gds.similarity.cosine(c.embedding, $vec) as score
                """
                for r in session.run(cypher_local_dense, cids=top_ppr_ids, vec=embedding):
                    chunk_dense_scores[r["cid"]] = float(r["score"])

        # Create Ranking Objects
        for r in ppr_results:
            cid = r["chunk_id"]
            ppr_score = float(r["ppr_score"])
            dense_score = chunk_dense_scores.get(cid, 0.0)
            
            chunk_rankings.add_chunk_ranking(
                chunk_id=cid,
                ppr_score=ppr_score,
                dense_score=dense_score,
                lexical_score=0.0
            )

        # Calculate Final Scores
        ranked_scores = chunk_rankings.calculate_chunk_rankings(
            self.rank_ppr_weight,
            self.rank_dense_weight,
            self.rank_lexical_weight
        )

        # Build Final Result List
        results = []
        for cs in ranked_scores[:k]:
            if cs.chunk_id in selected_chunks:
                text, doc_id = selected_chunks[cs.chunk_id]
                results.append(Chunk(
                    chunk_id=cs.chunk_id,
                    text=text,
                    score=cs.score,
                    doc_id=doc_id,
                    chunk_scores=cs
                ))

        timings["phase_3_ranking"] = time.perf_counter() - t0
        timings["total"] = time.perf_counter() - t_start_total
        
        self.log.info(
            f"⏱️ HippoRAG Search: Total={timings['total']:.3f}s | "
            f"Filter={timings['phase_1_filtering']:.3f}s | "
            f"PPR={timings['phase_2_ppr']:.3f}s | "
            f"Rank={timings['phase_3_ranking']:.3f}s"
        )
        
        return results