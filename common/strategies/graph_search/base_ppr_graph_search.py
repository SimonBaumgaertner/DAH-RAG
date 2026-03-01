import uuid
from typing import List

from common.data_classes.rag_system import Chunk
from common.neo4j.standard_executor import StandardExecutor
from common.strategies.graph_search.abstract_graph_search import GraphSearch
from common.strategies.graph_search.vector_graph_search import VectorGraphSearch


class BasePersonalizedPageRankGraphSearch(GraphSearch):
    """
    Graph search over the RAG graph using Personalized PageRank (PPR).

    Steps:
      1) Encode the query and run a vector search over Assertion embeddings
         to get top-k seed Assertions.
      2) Build an ephemeral in-memory GDS graph over (Assertion|Entity|Chunk)
         with undirected edges between:
            (Assertion) <-> (Entity) via SUBJECT/OBJECT
            (Assertion) <-> (Chunk)  via DERIVED_FROM
      3) Run gds.pageRank.stream with `sourceNodes` set to the projected node
         ids for those seed Assertions.
      4) Return top-k Chunk nodes by PPR score (including doc_id).

    Notes:
      - If no assertion seeds are found, this class falls back to a plain
        vector chunk search for robustness.
      - Index names and GDS params are configurable.
    """

    def __init__(
        self,
        *,
        executor: StandardExecutor,
        assertion_index_name: str = "assertion_embedding_ix",
        damping_factor: float = 0.85,
        max_iterations: int = 20,
    ):
        super().__init__(executor)
        self.assertion_index_name = assertion_index_name
        self.damping_factor = damping_factor
        self.max_iterations = max_iterations

    def _encode_query(self, query: str) -> List[float]:
        embedding = self.executor.encoder.encode(query, query=True)
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)


    def _project_graph(self, driver, graph_name: str) -> None:
        cypher = """
        CALL gds.graph.project(
          $graphName,
          ['Assertion','Entity','Chunk'],
          {
            SUBJECT:      { orientation: 'UNDIRECTED' },
            OBJECT:       { orientation: 'UNDIRECTED' },
            DERIVED_FROM: { orientation: 'UNDIRECTED' }
          }
        )
        YIELD graphName
        """
        with driver.session() as s:
            s.run(cypher, graphName=graph_name).consume()

    def _run_ppr_and_fetch_chunks(self, driver, graph_name: str, seed_el_ids: List[str], k: int) -> List[Chunk]:
        if not seed_el_ids:
            return []
        cypher = """
        WITH $graphName AS graphName, $seedElIds AS seedElIds
        MATCH (a:Assertion) WHERE elementId(a) IN seedElIds
        WITH graphName, collect(a) AS seeds
        CALL gds.pageRank.stream(graphName, {
            sourceNodes: seeds,
            dampingFactor: $dampingFactor,
            maxIterations: $maxIterations
        })
        YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS n, score
        WHERE n:Chunk
        MATCH (d:Document)-[:HAS_CHUNK]->(n)
        RETURN n.chunk_name AS chunk_id, n.text AS text, score AS score, d.id AS doc_id
        ORDER BY score DESC
        LIMIT $k
        """
        with driver.session() as s:
            recs = s.run(cypher,
                         graphName=graph_name,
                         seedElIds=seed_el_ids,
                         dampingFactor=self.damping_factor,
                         maxIterations=self.max_iterations,
                         k=k)
            return [Chunk(chunk_id=r["chunk_id"], text=r["text"], score=r["score"], doc_id=r["doc_id"])
                    for r in recs]


    def _drop_graph(self, driver, graph_name: str) -> None:
        with driver.session() as session:
            try:
                session.run("CALL gds.graph.drop($graphName)", graphName=graph_name).consume()
            except Exception:
                # Best-effort cleanup; ignore if already dropped / not present
                pass

    def _get_seed_assertion_element_ids(self, driver, embedding: List[float], k: int) -> List[str]:
        cypher = """
        CALL db.index.vector.queryNodes($assertion_index, $k, $embedding)
        YIELD node, score
        RETURN elementId(node) AS elId
        ORDER BY score DESC
        """
        with driver.session() as s:
            return [r["elId"] for r in s.run(cypher,
                                             assertion_index=self.assertion_index_name,
                                             k=k, embedding=embedding)]

    def search(self, query: str, k: int) -> List[Chunk]:
        """
        Execute PPR-based graph search and return top-k chunks.

        Fallback: if no assertion seeds are found or PPR returns no chunks,
        fall back to the simple vector chunk search to avoid empty results.
        """
        driver = self.executor.env.get_driver()
        embedding = self._encode_query(query)

        # 1) Seed Assertions via vector search (return **elementIds**, not internal ids)
        seed_el_ids = self._get_seed_assertion_element_ids(driver, embedding, k)

        # If no assertion seeds, fall back to vector chunk search
        if not seed_el_ids:
            return VectorGraphSearch(self.executor).search(query, k)

        # 2) Project ephemeral GDS graph
        graph_name = f"ppr_graph_{uuid.uuid4().hex[:12]}"
        try:
            self._project_graph(driver, graph_name)

            # 3) Run PPR using the **nodes themselves** as sourceNodes
            results = self._run_ppr_and_fetch_chunks(driver, graph_name, seed_el_ids, k)

            # Fallback to vector chunk search if PPR yields nothing
            if not results:
                return VectorGraphSearch(self.executor).search(query, k)

            return results
        finally:
            # Always clean up the in-memory graph
            self._drop_graph(driver, graph_name)