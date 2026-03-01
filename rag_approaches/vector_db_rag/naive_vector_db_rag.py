from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple

from common.data_classes.documents import Document
from common.data_classes.evaluation import EntryType
from common.data_classes.qa import QuestionAnswerPair
from common.data_classes.rag_system import Chunk, RAGSystem, Retriever, Indexer, Generator
from common.llm.base_llm_runner import BaseLLMRunner
from common.logging.run_logger import RunLogger
from common.strategies.chunking import ChunkingStrategy, ContextualizedSentenceChunker
from common.strategies.encoding import EncodingStrategy, MiniLMMeanPoolingEncoder, QwenEncoder
from common.strategies.generator import StandardMCAnswerGenerator


class InMemoryVectorDB:
    """Simple in‑memory vector store using a Python list."""

    def __init__(self) -> None:
        self.store: List[Tuple[np.ndarray, Chunk]] = []

    def add(self, embedding: np.ndarray, chunk: Chunk) -> None:
        self.store.append((embedding.astype(np.float32), chunk))

    def clear(self) -> None:
        self.store.clear()

    def __len__(self) -> int:
        return len(self.store)

    # helpers for debugging / tests
    def chunks(self) -> List[Chunk]:
        return [c for _, c in self.store]

    def embeddings(self) -> np.ndarray:
        return np.vstack([e for e, _ in self.store]) if self.store else np.empty((0, 0))


class VectorIndexer(Indexer):
    def __init__(
            self,
            *,
            db: InMemoryVectorDB,
            encoder: EncodingStrategy,
            chunker: ChunkingStrategy,
            log: RunLogger,
    ):
        self.db, self.encoder, self.chunker, self.log = db, encoder, chunker, log

    def index(self, document: Document) -> None:
        self.log.info("🔪 Chunking doc %s …", document.id)
        chunks = self.chunker.chunk(document)
        self.log.info("📑 → %d chunks", len(chunks))

        for chunk in chunks:
            emb = self.encoder.encode(chunk.text, query=False)  # encoding document chunks
            self.db.add(emb, chunk)

        # Track chunk count for this document
        self.log.track(
            entry_type=EntryType.CHUNK_COUNT_TRACK.value,
            identifier=document.id,
            value=str(len(chunks))
        )

        self.log.info("📥 Stored %d vectors for doc %s", len(chunks), document.id)


class VectorRetriever(Retriever):
    """
    Brute‑force cosine‑similarity search over the in‑memory vector store.
    """

    def __init__(self, *, encoder: EncodingStrategy, vector_db: InMemoryVectorDB, log: RunLogger | None = None):
        self.encoder = encoder
        self.db = vector_db
        self.log = log

    @staticmethod
    def _cosine(vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
        # cosine similarity between a single vector and a matrix of vectors
        vec_norm = np.linalg.norm(vec)
        mat_norm = np.linalg.norm(mat, axis=1)
        return (mat @ vec) / (mat_norm * vec_norm + 1e-9)


    def retrieve(self, query: str, return_chunk_amount: int = 50, qa_pair: Optional[QuestionAnswerPair] = None) -> List[Chunk]:
        """Return the top‑`return_chunk_amount` chunks for *query*.

        A larger candidate pool (`self._pool_size`, default 25) is selected via
        fast nearest‑neighbour search before cross‑encoder re‑ranking. This
        pool size can be overridden per instance.
        """

        self.log.info("🔍 Retrieving for query: %s", query)

        if len(self.db) == 0:
            self.log.warning("⚠️ Vector store empty – returning no results.")
            return []

        query_vec = self.encoder.encode(query, query=True)  # encoding query
        matrix = self.db.embeddings()  # (N, D)
        sims = self._cosine(query_vec, matrix)  # (N,)

        # -------------------------------------------
        # 1️⃣ Grab the requested amount of chunks
        # -------------------------------------------
        k = min(return_chunk_amount, len(sims))
        top_idx = np.argpartition(-sims, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]  # exact ordering

        results: List[Chunk] = []
        for i in top_idx:
            _, chunk = self.db.store[i]
            results.append(
                Chunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=float(sims[i]),
                    doc_id=chunk.doc_id,
                    metadata=chunk.metadata,
                )
            )
        return results


class NaiveVectorDBRAG(RAGSystem):
    def __init__(
            self,
            *,
            name: str = "vector‑db‑rag",
            log: RunLogger | None = None,
            llm: BaseLLMRunner,
            generator: Generator | None = None,
            chunker: ChunkingStrategy | None = None,
            encoder: EncodingStrategy | None = None,
            generation_max_chunks: int = 5,
    ):
        log = log or RunLogger(run_id=name)

        self.vector_db = InMemoryVectorDB()
        encoder = encoder or QwenEncoder(log=log)
        chunker = chunker or ContextualizedSentenceChunker(tokenizer=encoder.tokenizer)
        indexer = VectorIndexer(db=self.vector_db, encoder=encoder, chunker=chunker, log=log)
        retriever = VectorRetriever(encoder=encoder, vector_db=self.vector_db, log=log)
        generator = generator or StandardMCAnswerGenerator(llm=llm, log=log, max_chunks=generation_max_chunks)

        super().__init__(
            indexer=indexer,
            retriever=retriever,
            generator=generator,
            name=name,
            log=log
        )

    @property
    def db(self) -> InMemoryVectorDB:
        return self.vector_db

    def clear_memory(self) -> None:
        self.vector_db.clear()

