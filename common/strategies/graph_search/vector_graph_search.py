from typing import List

from common.data_classes.rag_system import Chunk
from common.neo4j.standard_executor import StandardExecutor
from common.strategies.graph_search.abstract_graph_search import GraphSearch


class VectorGraphSearch(GraphSearch):
    """Simple vector-based search over chunk embeddings."""
    def __init__(self, executor: StandardExecutor):
        self.executor = executor

    def search(self, query: str, k: int) -> List[Chunk]:
        embedding = self.executor.encoder.encode(query, query=True)
        embedding = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        driver = self.executor.env.get_driver()
        cypher = (
            "CALL db.index.vector.queryNodes('chunk_embedding_ix', $k, $embedding) "
            "YIELD node, score "
            "MATCH (d:Document)-[:HAS_CHUNK]->(node) "
            "RETURN node.chunk_name AS chunk_id, node.text AS text, score AS score, d.id AS doc_id "
            "ORDER BY score DESC"
        )
        with driver.session() as session:
            records = session.run(cypher, k=k, embedding=embedding)
            results = [
                Chunk(
                    chunk_id=r["chunk_id"],
                    text=r["text"],
                    score=r["score"],
                    doc_id=r["doc_id"],
                )
                for r in records
            ]
        return results