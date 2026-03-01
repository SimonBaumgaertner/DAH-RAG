from __future__ import annotations

import re
from typing import List, Optional, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from common.data_classes.documents import Document
from common.data_classes.evaluation import EntryType
from common.data_classes.qa import QuestionAnswerPair
from common.data_classes.rag_system import Chunk, Generator, Indexer, RAGSystem, Retriever
from common.llm.base_llm_runner import BaseLLMRunner
from common.logging.run_logger import RunLogger
from common.strategies.chunking import ChunkingStrategy, ContextualizedSentenceChunker
from common.strategies.generator import StandardMCAnswerGenerator


def _tokenize(text: str) -> List[str]:
    """Lightweight tokenizer for BM25 scoring."""
    return re.findall(r"\w+", text.lower())


class BM25Store:
    def __init__(self) -> None:
        self._tokenized_chunks: List[List[str]] = []
        self._chunks: List[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def add(self, tokens: List[str], chunk: Chunk) -> None:
        self._tokenized_chunks.append(tokens)
        self._chunks.append(chunk)
        self._bm25 = BM25Okapi(self._tokenized_chunks)

    def clear(self) -> None:
        self._tokenized_chunks.clear()
        self._chunks.clear()
        self._bm25 = None

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._chunks)

    @property
    def bm25(self) -> BM25Okapi | None:
        return self._bm25

    @property
    def chunks(self) -> List[Chunk]:
        return self._chunks


class BM25Indexer(Indexer):
    def __init__(
        self,
        *,
        store: BM25Store,
        chunker: ChunkingStrategy,
        log: RunLogger,
    ) -> None:
        self.store = store
        self.chunker = chunker
        self.log = log

    def index(self, document: Document) -> None:
        self.log.info("🔪 Chunking doc %s …", document.id)
        chunks = self.chunker.chunk(document)
        self.log.info("📑 → %d chunks", len(chunks))

        for chunk in chunks:
            tokens = _tokenize(chunk.text)
            self.store.add(tokens, chunk)

        self.log.track(
            entry_type=EntryType.CHUNK_COUNT_TRACK.value,
            identifier=document.id,
            value=str(len(chunks)),
        )

        self.log.info("📥 Stored %d chunk(s) for doc %s", len(chunks), document.id)


class BM25Retriever(Retriever):
    def __init__(
        self,
        *,
        store: BM25Store,
        log: RunLogger | None = None,
    ) -> None:
        self.store = store
        self.log = log or RunLogger(run_id="bm25-retriever")

    def _select_candidates(self, scores: np.ndarray, k: int) -> List[Chunk]:
        top_indices = np.argsort(-scores)[:k]
        candidates: List[Chunk] = []
        for idx in top_indices:
            chunk = self.store.chunks[idx]
            candidates.append(
                Chunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=float(scores[idx]),
                    doc_id=chunk.doc_id,
                    metadata=chunk.metadata,
                )
            )
        return candidates

    def retrieve(self, question: str, k: int = 5, qa_pair: Optional[QuestionAnswerPair] = None) -> List[Chunk]:
        self.log.info("🔍 Retrieving for query: %s", question)

        if self.store.bm25 is None:
            self.log.warning("⚠️ BM25 store empty – returning no results.")
            return []

        query_tokens = _tokenize(question)
        scores = self.store.bm25.get_scores(query_tokens)
        results = self._select_candidates(scores, min(k, len(scores)))
        return results


class BM25RAG(RAGSystem):
    def __init__(
        self,
        *,
        name: str = "bm25-rag",
        log: RunLogger | None = None,
        llm: BaseLLMRunner,
        generator: Generator | None = None,
        chunker: ChunkingStrategy | None = None,
        generation_max_chunks: int = 5,
    ) -> None:
        log = log or RunLogger(run_id=name)

        store = BM25Store()
        chunker = chunker or ContextualizedSentenceChunker()
        indexer = BM25Indexer(store=store, chunker=chunker, log=log)
        retriever = BM25Retriever(store=store, log=log)
        generator = generator or StandardMCAnswerGenerator(llm=llm, log=log, max_chunks=generation_max_chunks)

        super().__init__(
            indexer=indexer,
            retriever=retriever,
            generator=generator,
            name=name,
            log=log,
        )

        self.store = store

    def clear_memory(self) -> None:
        self.store.clear()
