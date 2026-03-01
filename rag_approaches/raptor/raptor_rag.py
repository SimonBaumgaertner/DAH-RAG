from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional

import numpy as np

from common.data_classes.evaluation import LLMCallContext
from common.data_classes.rag_system import Chunk, Generator, RAGSystem
from common.llm.base_llm_runner import BaseLLMRunner
from common.logging.run_logger import RunLogger
from common.strategies.chunking import ChunkingStrategy, ContextualizedSentenceChunker
from common.strategies.encoding import EncodingStrategy

from .raptor_indexer import PreChunkedClusterTreeBuilder, RaptorIndexer
from .raptor_retriever import RaptorRetriever
from .raptor_src.RetrievalAugmentation import RetrievalAugmentationConfig
from .raptor_src.SummarizationModels import BaseSummarizationModel
from .raptor_src.QAModels import BaseQAModel
from .raptor_src.EmbeddingModels import BaseEmbeddingModel
from .raptor_src.cluster_tree_builder import ClusterTreeConfig
from .raptor_src.tree_retriever import TreeRetrieverConfig


class _RunnerBackedSummarizationModel(BaseSummarizationModel):
    """Adapter that lets RAPTOR call our :class:`BaseLLMRunner`."""

    def __init__(self, runner: BaseLLMRunner, log: Optional[RunLogger] = None) -> None:
        self._runner = runner
        self._log = log or logging.getLogger(__name__)

    def summarize(self, context: str, max_tokens: int = 150) -> str:
        prompt = (
            "You are an expert research assistant. Summarize the following text "
            "concisely while preserving key facts. Aim for at most "
            f"{max_tokens} tokens.\n\n{context.strip()}"
        )

        messages = [
            {"role": "system", "content": "You create concise, factual summaries."},
            {"role": "user", "content": prompt},
        ]

        # Use document ID if available (during per-document indexing), otherwise use generic identifier
        # (e.g., during finalize_indexing tree building)
        identifier = "raptor_summarization"
        if self._log and hasattr(self._log, 'indexing_document_id') and self._log.indexing_document_id:
            identifier = f"{self._log.indexing_document_id}_raptor_summarization"
        
        if self._log:
            self._log.debug("📝 Summarising text of %d chars", len(context))

        completion = self._runner.generate_text(
            messages,
            context=LLMCallContext.INDEXING,
            identifier=identifier,
        )
        return completion.strip()


class _RunnerBackedQAModel(BaseQAModel):
    """Basic RAPTOR QA model backed by :class:`BaseLLMRunner`."""

    def __init__(self, runner: BaseLLMRunner, log: Optional[RunLogger] = None) -> None:
        self._runner = runner
        self._log = log or logging.getLogger(__name__)

    def answer_question(self, context: str, question: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise QA assistant. Use only the provided context "
                    "to answer the user's question. If the answer is unavailable, "
                    "say so explicitly."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context.strip()}\n\nQuestion: {question.strip()}\n"
                    "Answer in one or two sentences."
                ),
            },
        ]

        identifier = "raptor_retrieval_qa"
        if self._log:
            self._log.debug("🤖 Answering question with context (%d chars)", len(context))

        completion = self._runner.generate_text(
            messages,
            context=LLMCallContext.RETRIEVAL,
            identifier=identifier,
        )
        return completion.strip()


class _EncodingStrategyEmbedding(BaseEmbeddingModel):
    """Expose our :class:`EncodingStrategy` as a RAPTOR embedding model."""

    def __init__(
        self,
        *,
        encoder: EncodingStrategy,
        query_mode: bool,
    ) -> None:
        self._encoder = encoder
        self._query_mode = query_mode

    def create_embedding(self, text: str) -> Iterable[float]:
        vector = self._encoder.encode(text, query=self._query_mode)
        return np.asarray(vector, dtype=np.float32)


class RaptorRAG(RAGSystem):
    """RAPTOR RAG system integrated with the project's abstractions."""

    def __init__(
        self,
        *,
        llm: BaseLLMRunner,
        log: RunLogger,
        encoder: EncodingStrategy,
        generator: Generator,
        chunker: ChunkingStrategy | None = None,
        name: str = "raptor-rag",
        tree_max_tokens: int = 512,
        tree_num_layers: int = 5,
        tree_top_k: int = 8,
    ) -> None:
        chunker = chunker or ContextualizedSentenceChunker(tokenizer=encoder.tokenizer)

        summarizer = _RunnerBackedSummarizationModel(llm, log)
        qa_model = _RunnerBackedQAModel(llm, log)

        doc_embedding = _EncodingStrategyEmbedding(encoder=encoder, query_mode=False)
        query_embedding = _EncodingStrategyEmbedding(encoder=encoder, query_mode=True)

        tree_builder_config = ClusterTreeConfig(
            tokenizer=encoder.tokenizer,
            max_tokens=tree_max_tokens,
            num_layers=tree_num_layers,
            top_k=tree_top_k,
            summarization_length=tree_max_tokens,
            summarization_model=summarizer,
            embedding_models={"encoder": doc_embedding},
            cluster_embedding_model="encoder",
        )

        tree_retriever_config = TreeRetrieverConfig(
            tokenizer=encoder.tokenizer,
            top_k=tree_top_k,
            embedding_model=query_embedding,
            context_embedding_model="encoder",
            num_layers=1,
            start_layer=0,
        )

        rag_config = RetrievalAugmentationConfig(
            tree_builder_config=tree_builder_config,
            tree_retriever_config=tree_retriever_config,
            qa_model=qa_model,
            tree_builder_type="cluster",
        )

        builder = PreChunkedClusterTreeBuilder(rag_config.tree_builder_config)
        retriever = RaptorRetriever(
            log=log,
            config=rag_config.tree_retriever_config,
            default_top_k=tree_top_k,
        )

        chunk_store: Dict[str, Chunk] = {}
        indexer = RaptorIndexer(
            builder=builder,
            chunker=chunker,
            log=log,
            chunk_store=chunk_store,
            on_tree_update=retriever.update_tree,
        )

        self._encoder = encoder
        self._chunker = chunker
        self._builder = builder
        self._retriever = retriever
        self._chunk_store = chunk_store

        super().__init__(
            indexer=indexer,
            retriever=retriever,
            generator=generator,
            name=name,
            log=log,
        )

    @property
    def chunk_store(self) -> Dict[str, Chunk]:
        return self._chunk_store

