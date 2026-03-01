from typing import List, Optional
import time

from common.data_classes.qa import QuestionAnswerPair
from common.data_classes.rag_system import Chunk, Retriever
from common.logging.run_logger import RunLogger
from common.strategies.encoding import EncodingStrategy
from common.neo4j.neo4j_environment import Neo4JEnvironment
from common.neo4j.standard_executor import StandardExecutor
from common.strategies.graph_search.abstract_graph_search import GraphSearch


class DocumentAwareRetriever(Retriever):
    def __init__(
        self,
        *,
        encoder: EncodingStrategy,
        log: RunLogger,
        env: Neo4JEnvironment,
        executor: StandardExecutor,
        search: GraphSearch,
    ):
        self.encoder = encoder
        self.log = log
        self.env = env
        self.executor = executor
        self.search = search

    def retrieve(self, question: str, return_chunk_amount: int = 50, qa_pair: Optional[QuestionAnswerPair] = None) -> List[Chunk]:
        # Initialize timing
        total_start = time.time()
        
        self.log.info("🔍 Retrieving for query: %s", question)

        # Time the graph search
        search_start = time.time()
        results = self.search.search(question, return_chunk_amount, qa_pair=qa_pair)
        search_time = time.time() - search_start
        
        total_time = time.time() - total_start

        self.log.info("🔍 Retrieved %d chunk(s).", len(results))
        
        # Print timing summary in one line
        self.log.info(
            "⏱️  RETRIEVAL TIMING: TOTAL=%.3fs | Graph_Search=%.3fs | Per_Chunk=%.3fms",
            total_time,
            search_time,
            (search_time * 1000) / len(results) if len(results) > 0 else 0
        )
        
        return results
