from __future__ import annotations

from typing import List, Optional

from common.data_classes.documents import Document
from common.data_classes.qa import QuestionAnswerPair
from common.data_classes.rag_system import Chunk, RAGSystem, Indexer, Retriever, Generator
from common.llm.base_llm_runner import BaseLLMRunner
from common.logging.run_logger import RunLogger
from common.strategies.generator import StandardMCAnswerGenerator


class _NullIndexer(Indexer):
    def __init__(self, *, log: RunLogger) -> None:
        self.log = log

    def index(self, document: Document) -> None:
        self.log.info("🚫 Skipping indexing for document %s", document.id)


class _NullRetriever(Retriever):
    def __init__(self, *, log: RunLogger) -> None:
        self.log = log

    def retrieve(self, question: str, k: int = 5, qa_pair: Optional[QuestionAnswerPair] = None) -> List[Chunk]:
        self.log.info("🚫 Skipping retrieval for question: %s", question)
        return []


class NoRAGGeneration(RAGSystem):
    """Simple RAGSystem that only performs answer generation."""

    def __init__(
        self,
        *,
        name: str = "no-rag-generation",
        log: RunLogger | None = None,
        llm: BaseLLMRunner,
        generator: Generator | None = None,
        generation_max_chunks: int = 5,
    ) -> None:
        log = log or RunLogger(run_id=name)

        indexer = _NullIndexer(log=log)
        retriever = _NullRetriever(log=log)
        generator = generator or StandardMCAnswerGenerator(llm=llm, log=log, max_chunks=generation_max_chunks)

        super().__init__(
            indexer=indexer,
            retriever=retriever,
            generator=generator,
            name=name,
            log=log,
        )

    def clear_memory(self) -> None:
        """No in-memory state to clear."""
        pass
