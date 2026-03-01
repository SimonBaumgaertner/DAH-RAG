from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, runtime_checkable

from common.data_classes.documents import Document
from common.data_classes.qa import QuestionAnswerPair
from common.logging.run_logger import RunLogger
from common.strategies.graph_search.chunk_rankings import ChunkScore

@dataclass
class Chunk:
    chunk_id: str
    text: str
    score: float | None = None
    doc_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    chunk_scores: Optional[ChunkScore] = None

    def to_json(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text
        }
@runtime_checkable
class Indexer(Protocol):
    @abstractmethod
    def index(self, document: Document) -> None:
        ...
    
    def finalize_indexing(self) -> None:
        """Optional method called after all documents have been indexed.
        
        Indexers can override this to perform batch operations or cleanup.
        Default implementation does nothing to maintain backward compatibility.
        """
        pass


@runtime_checkable
class Retriever(Protocol):

    @abstractmethod
    def retrieve(self, question: str, k: int = 5, qa_pair: Optional[QuestionAnswerPair] = None) -> List[Chunk]:
        ...


@runtime_checkable
class Generator(Protocol):
    @abstractmethod
    def generate(self, qa_pair: QuestionAnswerPair, context: List[Chunk]) -> str:
        ...

class RAGSystem(ABC):
    def __init__(
        self,
        *,
        indexer: Indexer,
        retriever: Retriever,
        generator: Generator,
        name: str,
        log: RunLogger,
    ):
        self._name = name
        self._indexer = indexer
        self._retriever = retriever
        self._generator = generator
        self.log = log

    def index_document(self, document: Document) -> None:
        self._indexer.index(document)

    @property
    def name(self) -> str:
        return self._name

    @property
    def indexer(self) -> Indexer:
        return self._indexer

    @property
    def retriever(self) -> Retriever:
        return self._retriever

    @property
    def generator(self) -> Generator:
        return self._generator