from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from common.data_classes.rag_system import Chunk, Generator, RAGSystem
from common.llm.base_llm_runner import BaseLLMRunner
from common.logging.run_logger import RunLogger
from common.strategies.chunking import (
    ChunkingStrategy,
    ContextualizedSentenceChunker,
)
from common.strategies.encoding import EncodingStrategy, QwenEncoder
from common.strategies.generator import StandardMCAnswerGenerator

import sys
from pathlib import Path

# Add the local HippoRAG src directory to the path
hippo_src_path = Path(__file__).parent / "HippoRAG" / "src"
sys.path.insert(0, str(hippo_src_path))

from hipporag.HippoRAG import HippoRAG as HippoRAGEngine
from hipporag.embedding_model import base as hippo_embedding_base
from hipporag.llm import base as hippo_llm_base
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.llm_utils import TextChatMessage

from rag_approaches.hippo_rag.hippo_indexer import HippoRAGIndexer
from rag_approaches.hippo_rag.hippo_retriever import HippoRAGRetriever


def _chunk_to_dict(chunk: Chunk) -> Dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "doc_id": chunk.doc_id,
        "metadata": chunk.metadata or {},
    }


def _load_chunk_store(path: Path, log: RunLogger) -> Dict[str, Chunk]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        log.warning("⚠️ Could not parse HippoRAG chunk metadata at %s; starting empty.", path)
        return {}

    store: Dict[str, Chunk] = {}
    for entry in data:
        chunk_id = entry.get("chunk_id")
        text = entry.get("text", "")
        if not chunk_id:
            continue
        metadata = entry.get("metadata") or {}
        metadata = {k: v for k, v in metadata.items() if v is not None}
        store[chunk_id] = Chunk(
            chunk_id=chunk_id,
            text=text,
            doc_id=entry.get("doc_id"),
            metadata=metadata or None,
        )
    return store


def _save_chunk_store(path: Path, store: Dict[str, Chunk], log: RunLogger) -> None:
    try:
        payload = [_chunk_to_dict(chunk) for chunk in store.values()]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception as exc:  # pragma: no cover - defensive logging
        log.warning("⚠️ Failed to persist HippoRAG chunk metadata to %s: %s", path, exc)


class _RunnerBackedLLM(hippo_llm_base.BaseLLM):
    """HippoRAG-compatible LLM wrapper around our :class:`BaseLLMRunner`."""

    def __init__(self, *, runner: BaseLLMRunner, global_config: BaseConfig) -> None:
        super().__init__(global_config=global_config)
        self._runner = runner
        self._current_document_id = None  # Store the current document ID for thread safety
        self._current_context = None  # Store the current context (indexing, retrieval, generation)
        # Store the logger reference for thread-safe access
        self._logger = getattr(runner, 'log', None)
        self.llm_config = hippo_llm_base.LLMConfig.from_dict(
            {
                "llm_name": self.llm_name,
                "generate_params": {
                    "model": self.llm_name,
                    "max_completion_tokens": getattr(global_config, "max_new_tokens", None),
                    "n": getattr(global_config, "num_gen_choices", 1),
                    "seed": getattr(global_config, "seed", 0),
                    "temperature": getattr(global_config, "temperature", 0.0),
                },
            }
        )

    def _init_llm_config(self) -> None:  # pragma: no cover - interface requirement
        """HippoRAG expects this hook but our config is initialised in ``__init__``."""

    def set_document_id(self, document_id: str) -> None:
        """Set the current document ID for thread-safe context tracking."""
        self._current_document_id = document_id

    def set_context(self, context: LLMCallContext, identifier: str) -> None:
        """Set the current context and identifier for LLM calls."""
        self._current_context = context
        self._current_document_id = identifier

    def infer(self, messages: List[TextChatMessage], **_: object):
        """Forward HippoRAG prompts to the synchronous runner."""
        formatted_messages = [
            {"role": m["role"], "content": str(m["content"])} for m in messages
        ]
        
        # Use context-aware LLM call
        from common.data_classes.evaluation import LLMCallContext
        
        # Determine context and identifier
        context = self._current_context or LLMCallContext.INDEXING
        identifier = self._current_document_id or "hippo_indexing"
        
        # Log the LLM call start if logger is available
        if self._logger:
            self._logger.info("🤖 HippoRAG LLM call starting (Context: %s, ID: %s)", context.value, identifier)
            
        completion = self._runner.generate_text(formatted_messages, context=context, identifier=identifier)
        
        # Calculate tokens for metadata (HippoRAG expects this)
        prompt_view = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in formatted_messages
        )
        prompt_tokens = len(self._runner.tokenize(prompt_view))
        completion_tokens = len(self._runner.tokenize(completion))
        metadata = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": "stop",
        }
        
        # Log the completion if logger is available
        if self._logger:
            self._logger.info("✅ HippoRAG LLM call completed for document: %s (%d in, %d out tokens)", 
                            identifier, prompt_tokens, completion_tokens)
        
        # HippoRAG's OpenIE path expects a cache hit flag in the tuple response.
        return completion, metadata, False

    def ainfer(self, chat: List[TextChatMessage]):  # pragma: no cover - unused path
        raise NotImplementedError("Asynchronous inference is not supported in this adapter.")

    def batch_infer(self, batch_chat: List[List[TextChatMessage]]):  # pragma: no cover
        # Log batch processing start if logger is available
        if self._logger:
            self._logger.info("🔄 HippoRAG batch processing starting with %d items", len(batch_chat))
            
        results, metadatas, cache_hits = [], [], []
        for i, chat in enumerate(batch_chat):
            if self._logger:
                self._logger.debug("🔄 Processing batch item %d/%d", i + 1, len(batch_chat))
            response, metadata, cache_hit = self.infer(chat)
            results.append(response)
            metadatas.append(metadata)
            cache_hits.append(cache_hit)
            
        # Log batch processing completion if logger is available
        if self._logger:
            self._logger.info("✅ HippoRAG batch processing completed with %d items", len(batch_chat))
            
        return results, metadatas, cache_hits


class _EncodingStrategyEmbedding(hippo_embedding_base.BaseEmbeddingModel):
    """Bridge between HippoRAG embedding expectations and our encoder strategy."""

    def __init__(
        self,
        *,
        encoder: EncodingStrategy,
        global_config: BaseConfig,
        embedding_model_name: str,
    ) -> None:
        self._encoder = encoder
        super().__init__(global_config=global_config)
        self.embedding_model_name = embedding_model_name
        probe = np.asarray(self._encoder.encode("dimension probe", query=False), dtype=np.float32)
        self.embedding_dim = int(probe.shape[0])
        self.embedding_config = hippo_embedding_base.EmbeddingConfig.from_dict(
            {
                "embedding_model_name": embedding_model_name,
                "embedding_dim": self.embedding_dim,
            }
        )

    def batch_encode(self, texts: Iterable[str] | str, **kwargs: object):
        is_single = isinstance(texts, str)
        payload = [texts] if is_single else list(texts)

        instruction = kwargs.get("instruction")
        query_mode = False
        if instruction is not None:
            if isinstance(instruction, str):
                query_mode = instruction.strip() != ""
            else:
                query_mode = True

        embeddings = [
            np.asarray(self._encoder.encode(text, query=query_mode), dtype=np.float32)
            for text in payload
        ]

        matrix = np.vstack(embeddings) if embeddings else np.empty((0, self.embedding_dim), dtype=np.float32)

        if kwargs.get("norm") or kwargs.get("normalize"):
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = matrix / norms

        if is_single:
            return matrix[0]
        return matrix


@contextmanager
def _patched_factories(
    *,
    runner: BaseLLMRunner,
    encoder: EncodingStrategy,
):
    import hipporag.llm as hippo_llm_module
    import hipporag.embedding_model as hippo_embedding_module
    import hipporag.HippoRAG as hippo_rag_module
    import hipporag.StandardRAG as hippo_standard_rag_module

    original_llm_getter = hippo_llm_module._get_llm_class
    original_embedding_getter = hippo_embedding_module._get_embedding_model_class

    def _llm_factory(config: BaseConfig):
        return _RunnerBackedLLM(runner=runner, global_config=config)

    def _embedding_class(embedding_model_name: str):
        class _BoundEmbedding(_EncodingStrategyEmbedding):
            def __init__(self, *, global_config: BaseConfig, embedding_model_name: str):
                super().__init__(
                    encoder=encoder,
                    global_config=global_config,
                    embedding_model_name=embedding_model_name,
                )

        return _BoundEmbedding

    def _patched_llm_getter(config: BaseConfig):
        return _llm_factory(config)

    def _patched_embedding_getter(embedding_model_name: str = ""):
        return _embedding_class(embedding_model_name)

    # Patch at multiple levels to ensure it works
    hippo_llm_module._get_llm_class = _patched_llm_getter
    hippo_embedding_module._get_embedding_model_class = _patched_embedding_getter
    
    # Also patch the imported references in the main modules
    hippo_rag_module._get_llm_class = _patched_llm_getter
    hippo_standard_rag_module._get_llm_class = _patched_llm_getter

    try:
        yield
    finally:
        hippo_llm_module._get_llm_class = original_llm_getter
        hippo_embedding_module._get_embedding_model_class = original_embedding_getter
        hippo_rag_module._get_llm_class = original_llm_getter
        hippo_standard_rag_module._get_llm_class = original_llm_getter


class HippoRAG(RAGSystem):
    """RAG system backed by the official HippoRAG package."""

    def __init__(
        self,
        *,
        llm: BaseLLMRunner,
        name: str = "hippo-rag",
        log: Optional[RunLogger] = None,
        encoder: Optional[EncodingStrategy] = None,
        chunker: Optional[ChunkingStrategy] = None,
        generator: Optional[Generator] = None,
        generation_max_chunks: int = 5,
        working_dir: Optional[str | Path] = None,
        reset: bool = False,
        hippo_config: Optional[Dict[str, object]] = None,
    ) -> None:
        log = log or RunLogger(run_id=name)
        encoder = encoder or QwenEncoder(log=log)
        chunker = chunker or ContextualizedSentenceChunker(tokenizer=encoder.tokenizer)

        work = Path(working_dir or f"./{name}_storage")
        if reset and work.exists():
            shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)

        config = BaseConfig()
        for key, value in (hippo_config or {}).items():
            setattr(config, key, value)

        config.save_dir = str(work)
        config.llm_name = getattr(config, "llm_name", None) or llm.__class__.__name__
        config.run_on_cluster = getattr(llm, "run_on_cluster", False)
        # Set the embedding model name to our encoder class name
        config.embedding_model_name = encoder.__class__.__name__
        print(f"🔧 Setting embedding model name to: {config.embedding_model_name}")
        config.openie_mode = getattr(config, "openie_mode", "online")
        config.force_openie_from_scratch = getattr(
            config, "force_openie_from_scratch", True
        )
        config.force_index_from_scratch = getattr(
            config, "force_index_from_scratch", False
        )
        
        # Pass the main experiment logger to the config so HippoRAG can use it
        config.experiment_logger = log
        
        # Ensure we don't use OpenAI by setting a dummy API key if needed
        # This prevents the OpenAI client from being initialized
        import os
        if not os.getenv('OPENAI_API_KEY'):
            os.environ['OPENAI_API_KEY'] = 'sk-dummy-key-for-hipporag'

        with _patched_factories(runner=llm, encoder=encoder):
            print("🔧 Creating HippoRAG engine with patched modules...")
            rag_engine = HippoRAGEngine(global_config=config)
            print("🔧 HippoRAG engine created successfully!")

            metadata_path = work / "chunk_metadata.json"
            self._chunk_store: Dict[str, Chunk] = _load_chunk_store(metadata_path, log)

            def _persist_store() -> None:
                _save_chunk_store(metadata_path, self._chunk_store, log)

            indexer = HippoRAGIndexer(
                rag=rag_engine,
                chunker=chunker,
                log=log,
                chunk_store=self._chunk_store,
                on_update=_persist_store,
            )
            retriever = HippoRAGRetriever(
                rag=rag_engine,
                log=log,
                chunk_store=self._chunk_store,
            )
            generator = generator or StandardMCAnswerGenerator(
                llm=llm, log=log, max_chunks=generation_max_chunks
            )

            super().__init__(
                indexer=indexer,
                retriever=retriever,
                generator=generator,
                name=name,
                log=log,
            )

            self.engine = rag_engine
            self._persist_store = _persist_store

    def close(self) -> None:
        """HippoRAG currently does not expose background resources; method kept for API parity."""
        if hasattr(self, "_persist_store"):
            self._persist_store()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass
