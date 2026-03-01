from datetime import datetime
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from common.data_classes.data_set import DataSet
from common.data_classes.enums import (
    RegisteredDataset,
    RAG,
    GenerationConfiguration,
    Encoder,
    LLMBackend,
    ChunkingStrategy as ChunkingStrategyEnum,
    Reranker,
)
from common.llm.base_llm_runner import BaseLLMRunner
from common.llm.llm_factory import get_llm_runner
from common.logging.run_logger import RunLogger
from common.strategies.chunking import (
    ContextualizedSentenceChunker,
    FixedSizeWordChunker,
    SemanticChunker,
    ChunkingStrategy,
)
from common.strategies.encoding import QwenEncoder, NVEmbedV2Encoder, JinaEncoder, KaLMEncoder, OpenRouterEncoder
from common.strategies.generator import DummyGenerator, StandardMCAnswerGenerator, StandardOpenAnswerGenerator



from common.strategies.reranking import NoRerank, JinaReranker, RerankingStrategy
from rag_approaches.no_rag_generation.no_rag_generation import NoRAGGeneration
os.environ["MAX_ASYNC"] = "6" # TODO do this more elegantly
from rag_approaches.vector_db_rag.naive_vector_db_rag import NaiveVectorDBRAG


def prepare_log(experiment_name: str, logs_dir_name: str = "logs_and_tracks") -> RunLogger:
    """
    Create a timestamped run id and a RunLogger instance.
    """
    root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now().strftime("%d-%m_%H-%M")
    run_id = f"{experiment_name}_{timestamp}"

    log_dir = root / logs_dir_name
    log_dir.mkdir(parents=True, exist_ok=True)

    return RunLogger(run_id=run_id, log_dir=log_dir)


def prepare_llm(
    run_on_cluster: bool,
    log: RunLogger,
    backend: str = "local-instruct",
    model_path: Optional[Union[Path, str]] = None,
    llm_name: Optional[str] = None,
    gen_kwargs: Optional[Dict[str, Any]] = None,
    max_concurrent_llm_executions: int = 10,
):
    backend_lc = backend.lower()

    if model_path is None:
        if backend_lc in {"openrouter", "open-router", "router"}:
            # Use provided llm_name or default to Llama_3_3_70B
            resolved_model: Union[Path, str] = llm_name if llm_name else "meta-llama/llama-3.3-70b-instruct"
        else:
            resolved_model = "default-model" # Fallback or dummy
    else:
        resolved_model = model_path

    if isinstance(resolved_model, Path):
        resolved_model = resolved_model.expanduser()
        model_argument = resolved_model.as_posix()
    else:
        model_argument = str(resolved_model)

    resolved_backend = _normalize_backend_value(backend)
    llm_short_name = get_llm_name(
        run_on_cluster,
        backend=resolved_backend,
        model_path=resolved_model,
    )

    return get_llm_runner(
        backend=backend,
        model=model_argument,
        log=log,
        run_on_cluster=run_on_cluster,
        llm_short_name=llm_short_name,
        gen_kwargs=gen_kwargs,
        max_concurrent_llm_executions=max_concurrent_llm_executions,
    )

def prepare_chunker(
    chunking_strategy: ChunkingStrategyEnum,
    encoder,
    log: RunLogger,
) -> ChunkingStrategy:
    """
    Create a chunking strategy based on the enum selection.
    
    Args:
        chunking_strategy: The chunking strategy enum value
        encoder: The encoder instance (needed for tokenizer)
        log: Logger instance
        
    Returns:
        ChunkingStrategy instance
    """
    if chunking_strategy == ChunkingStrategyEnum.ContextualizedChunker:
        # Token-aware, sentence-based chunking (default)
        return ContextualizedSentenceChunker(tokenizer=encoder.tokenizer)
    elif chunking_strategy == ChunkingStrategyEnum.FixedSizeChunker:
        # Simple fixed-size word chunking
        return FixedSizeWordChunker(chunk_size=1200)
    elif chunking_strategy == ChunkingStrategyEnum.SemanticChunker:
        # Semantic chunking using embeddings
        return SemanticChunker(
            encoding_strategy=encoder,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
            max_chunk_tokens=1000,
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {chunking_strategy}")

def prepare_reranker(
    reranker: Reranker,
    log: RunLogger,
) -> RerankingStrategy:
    if reranker == Reranker.NoRerank:
        return NoRerank()
    elif reranker == Reranker.Jina_Reranker_v3_600M:
        return JinaReranker(log=log)
    else:
        raise ValueError(f"Unknown reranker: {reranker}")

def prepare_rag_system(
    rag_system: RAG,
    generation: GenerationConfiguration,
    log: RunLogger,
    llm: BaseLLMRunner,
    encoding_strategy: Encoder,
    chunking_strategy: ChunkingStrategyEnum = ChunkingStrategyEnum.ContextualizedChunker,
    checkpoint_name: str = None,
    skip_db_reset: bool = False,
):
    # ---------------- Setup Generator -----------------------------
    if generation == GenerationConfiguration.NoGen:
        generator = DummyGenerator()
    elif generation == GenerationConfiguration.MCQA:
        generator = StandardMCAnswerGenerator(llm=llm, log=log)
    elif generation == GenerationConfiguration.OpenAnswer:
        generator = StandardOpenAnswerGenerator(llm=llm, log=log)
    else:
        raise Exception("Not a valid GenerationConfiguration")

    # ---------------- Setup Encoder -----------------------------
    if encoding_strategy == Encoder.Qwen3_600M:
        encoder = QwenEncoder(
            task="query",
            max_length=1200,
            model_name="Qwen/Qwen3-Embedding-0.6B",
            embedding_dims=1024,
            log=log,
        )
    elif encoding_strategy == Encoder.Qwen3_4B:
        encoder = QwenEncoder(
            task="query",
            max_length=1200,
            model_name="Qwen/Qwen3-Embedding-4B",
            embedding_dims=2560,
            log=log,
        )
    elif encoding_strategy == Encoder.NV_Embed_v2:
        encoder = NVEmbedV2Encoder(
            model_name="nvidia/nv-embed-v2",
            embedding_dims=4096,
            max_length=1200,
            log=log,
        )
    elif encoding_strategy == Encoder.Jina_v3_600M:
        encoder = JinaEncoder(
            model_name="jinaai/jina-embeddings-v3",
            embedding_dims=1024,
            max_length=8192,
            log=log,
        )
    elif encoding_strategy == Encoder.KaLM_EMBEDDING_Gemma3_12B:
        encoder = KaLMEncoder(
            model_name="tencent/KaLM-Embedding-Gemma3-12B-2511",
            embedding_dims=3840,
            max_length=8192,
            log=log,
        )
    elif encoding_strategy == Encoder.Qwen3_8B:
        encoder = QwenEncoder(
            task="query",
            max_length=1200,
            model_name="Qwen/Qwen3-Embedding-8B",
            embedding_dims=4096,
            log=log,
        )
    elif encoding_strategy == Encoder.Qwen3_4B_OpenRouter:
        encoder = OpenRouterEncoder(
            model_name="qwen/qwen3-embedding-4b",
            embedding_dims=2560,
            log=log,
            run_on_cluster=False,  # Can be made configurable if needed
        )
    else:
        raise Exception("Not a valid Encoder")


    # ---------------- Setup Chunker -----------------------------
    chunker = prepare_chunker(chunking_strategy, encoder, log)
    
    # ---------------- Setup RAG -----------------------------
    if rag_system == RAG.NaiveVectorDB:
        rag = NaiveVectorDBRAG(
            llm=llm,
            log=log,
            generator=generator,
            chunker=chunker,
            encoder=encoder,
        )
    elif rag_system == RAG.BM25:
        from rag_approaches.bm25.bm25_rag import BM25RAG

        rag = BM25RAG(
            llm=llm,
            log=log,
            generator=generator,
            chunker=chunker,
        )
    elif rag_system == RAG.DocAwareHybridRAG:
        database_data_root = Path(__file__).parent.parent.parent / "neo4j_database"
        from rag_approaches.doc_aware_hybrid_RAG.doc_aware_hybrid_RAG import DocumentAwareHybridRAG
        from common.strategies.graph_search.document_aware_advanced_graph_search import DocumentAwareAdvancedGraphSearch
        from common.strategies.graph_search.document_aware_experimental_graph_search import DocumentAwareExperimentalGraphSearch
        rag = DocumentAwareHybridRAG(
            llm=llm,
            log=log,
            database_data_root=database_data_root,
            generator=generator,
            encoder=encoder,
            checkpoint_name=checkpoint_name,
            skip_db_reset=skip_db_reset,
        )
    elif rag_system == RAG.FastGraphRAG:
        from rag_approaches.fast_graph_rag.fast_graph_rag import FastGraphRAG
        domain = os.getenv("FAST_GRAPHRAG_DOMAIN", "")
        example_queries = os.getenv("FAST_GRAPHRAG_EXAMPLE_QUERIES", "")
        entity_types_env = os.getenv("FAST_GRAPHRAG_ENTITY_TYPES", "")
        entity_types = [et.strip() for et in entity_types_env.split(",") if et.strip()] or None

        rag = FastGraphRAG(
            llm=llm,
            log=log,
            generator=generator,
            encoder=encoder,
            chunker=chunker,
            domain=domain,
            example_queries=example_queries,
            entity_types=entity_types,
            checkpoint_name=checkpoint_name,
        )
    elif rag_system == RAG.HippoRAG:
        from rag_approaches.hippo_rag.hippo_rag import HippoRAG

        # Use cache_and_storage directory for HippoRAG storage
        cache_storage_dir = Path(__file__).parent / "cache_and_storage"
        if checkpoint_name and checkpoint_name.strip():
            working_dir = cache_storage_dir / checkpoint_name.strip()
            log.info(f"🔄 Using checkpoint directory: {checkpoint_name.strip()}")
        else:
            working_dir = cache_storage_dir / f"{log.run_id}"
            log.info(f"🆕 Using new experiment directory: {log.run_id}")
        
        rag = HippoRAG(
            llm=llm,
            log=log,
            encoder=encoder,
            chunker=chunker,
            generator=generator,
            name="hippo-rag",
            working_dir=str(working_dir),
        )
    elif rag_system == RAG.RaptorRAG:
        from rag_approaches.raptor.raptor_rag import RaptorRAG

        rag = RaptorRAG(
            llm=llm,
            log=log,
            encoder=encoder,
            chunker=chunker,
            generator=generator,
            name="raptor-rag",
        )
    elif rag_system == RAG.NoRAGGeneration:
        rag = NoRAGGeneration(
            llm=llm,
            log=log,
            generator=generator,
        )
    else:
        raise Exception("Not a valid RAG Strategy")

    return rag

def prepare_dataset(dataset: RegisteredDataset) -> DataSet:
    if dataset == RegisteredDataset.Scaling:
        # For Scaling dataset, we combine HotpotQA_1k (base) with HotpotQA_Scaling (distractors)
        base_root = Path(__file__).resolve().parent.parent / "data" / RegisteredDataset.HotpotQA_1k.value
        distractor_root = Path(__file__).resolve().parent.parent / "data" / "HotpotQA_Scaling"

        # Load base documents and QA pairs
        dataset_obj = DataSet(base_root)

        # Load distractor documents
        distractor_docs = []
        if distractor_root.exists():
            for sub in sorted(distractor_root.iterdir()):
                if sub.is_dir():
                    from common.data_classes.documents import Document
                    distractor_docs.append(Document.from_folder(sub))
        
        # Merge documents (base + distractors)
        dataset_obj.documents.extend(distractor_docs)
        
        return dataset_obj

    data_root = Path(__file__).resolve().parent.parent / "data" / dataset.value
    return DataSet(data_root)


def prepare_experiment_name(
    run_on_cluster: bool,
    dataset: RegisteredDataset,
    rag_system: RAG,
    generation: GenerationConfiguration,
    model_path: Optional[Union[Path, str]] = None,
    comment: Optional[str] = None,
    backend: Optional[Union[str, LLMBackend]] = None,
) -> str:
    """Create a standardized experiment name string."""

    # Use your existing method to get the LLM name
    backend_value = _normalize_backend_value(backend)
    llm_name = get_llm_name(run_on_cluster, backend=backend_value, model_path=model_path)

    # Decide suffix based on where it runs
    location = _derive_location_label(run_on_cluster, backend_value)

    # Base experiment name
    experiment_name = (
        f"{dataset.value}-"
        f"{rag_system.value}-"
        f"{llm_name}({location})-"
        f"{generation.value}"
    )

    # Add optional comment
    if comment:
        experiment_name += f"-{comment}"

    return experiment_name


def _normalize_backend_value(backend: Optional[Union[str, LLMBackend]]) -> str:
    if backend is None:
        return ""
    if isinstance(backend, LLMBackend):
        return backend.value
    return str(backend)


def _derive_location_label(run_on_cluster: bool, backend_value: str) -> str:
    backend_lc = backend_value.lower()
    if backend_lc in {"openrouter", "open-router", "router"}:
        return "openrouter"
    if backend_lc in {"openai", "gpt-oss"}:
        return backend_lc
    if run_on_cluster:
        return "cluster"
    return "local"


def _infer_model_short_name(model_path: Optional[Union[Path, str]]) -> Optional[str]:
    if not model_path:
        return None
    name = Path(str(model_path)).name.replace("_", "-")
    parts = [part for part in name.split("-") if part]
    if len(parts) >= 2 and parts[1][-1].lower() == "b":
        base = parts[0].capitalize()
        size = parts[1].upper()
        return f"{base}-{size}"
    if parts:
        return parts[0].capitalize()
    return None


def get_llm_name(
    run_on_cluster: bool,
    backend: Optional[str] = None,
    model_path: Optional[Union[Path, str]] = None,
) -> str:
    backend_lc = (backend or "").lower()
    if backend_lc == "dummy":
        return "Dummy"
    inferred_name = _infer_model_short_name(model_path)
    if backend_lc in {"openrouter", "open-router", "router"}:
        return inferred_name or "OpenRouter"

    if inferred_name:
        return inferred_name

    if run_on_cluster:
        return "Qwen3-30B"
    return "Qwen3-4B"

