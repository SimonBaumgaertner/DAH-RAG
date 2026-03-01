from pathlib import Path
import sys
import os
import argparse

# Add the project root to Python path so it work from where ever you are
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.data_classes.enums import RAG, RegisteredDataset, Encoder, LLMBackend, LLMName, Reranker, ChunkingStrategy as ChunkingStrategyEnum
from common.strategies.generator import StandardMCAnswerGenerator, DummyGenerator
from common.strategies.reranking import NoRerank
from experiments.base_experiment import *
from common.data_classes.data_set import DataSet
from common.evaluation.pipeline import (
    indexing_evaluation_pipeline,
    retrieval_evaluation_pipeline,
)

# The Idea of this class is to make a very simply adjustable experiment
DEFAULT_RUN_ON_CLUSTER = False # Use proxy or not?
DEFAULT_RAG_SYSTEM = RAG.NaiveVectorDB
DEFAULT_DATASET = RegisteredDataset.HotpotQA_100 # good for testing
DEFAULT_GENERATION_CONFIGURATION = GenerationConfiguration.NoGen
DEFAULT_ENCODER = Encoder.Jina_v3_600M
DEFAULT_BACKEND = LLMBackend.OpenRouter
DEFAULT_LLM = LLMName.Llama_3_3_70B #Llama_3_3_70B
DEFAULT_MAX_CONCURRENT_LLM_CALLS = 6
DEFAULT_SKIP_INDEXING = False # e.g. for ablation
DEFAULT_CHUNKING_STRATEGY = ChunkingStrategyEnum.ContextualizedChunker
DEFAULT_RERANKER = Reranker.NoRerank # Jina_Reranker_v3_600M, NoRerank
DEFAULT_COMMENT = "Test"
DEFAULT_CHECKPOINT_NAME = ""

def parse_arguments():
    """Parse command line arguments with defaults from the constants above."""
    parser = argparse.ArgumentParser(description='Adjustable experiment runner')
    
    parser.add_argument('--run-on-cluster', 
                       action='store_true', 
                       default=DEFAULT_RUN_ON_CLUSTER,
                       help=f'Run on cluster? (default: {DEFAULT_RUN_ON_CLUSTER})')
    
    parser.add_argument('--rag-system', 
                       type=str, 
                       default=DEFAULT_RAG_SYSTEM.value,
                       choices=[rag.value for rag in RAG],
                       help=f'RAG system to use (default: {DEFAULT_RAG_SYSTEM.value})')
    
    parser.add_argument('--dataset', 
                       type=str, 
                       default=DEFAULT_DATASET.value,
                       choices=[dataset.value for dataset in RegisteredDataset],
                       help=f'Dataset to use (default: {DEFAULT_DATASET.value})')
    
    parser.add_argument('--generation', 
                       type=str, 
                       default=DEFAULT_GENERATION_CONFIGURATION.value,
                       choices=[gen.value for gen in GenerationConfiguration],
                       help=f'Generation configuration (default: {DEFAULT_GENERATION_CONFIGURATION.value})')
    
    parser.add_argument('--encoder', 
                       type=str, 
                       default=DEFAULT_ENCODER.value,
                       choices=[enc.value for enc in Encoder],
                       help=f'Encoder to use (default: {DEFAULT_ENCODER.value})')
    parser.add_argument('--backend',
                       type=str,
                       default=DEFAULT_BACKEND.value,
                       choices=[backend.value for backend in LLMBackend],
                       help=f'LLM backend to use (default: {DEFAULT_BACKEND.value})')
    
    parser.add_argument('--llm',
                       type=str,
                       default=DEFAULT_LLM.name,
                       choices=[llm.name for llm in LLMName],
                       help=f'LLM model to use (default: {DEFAULT_LLM.name})')
    
    parser.add_argument('--comment',
                       type=str,
                       default=DEFAULT_COMMENT,
                       help=f'Optional comment to add to experiment name (default: "{DEFAULT_COMMENT}")')
    
    parser.add_argument('--max-concurrent',
                       type=int,
                       default=DEFAULT_MAX_CONCURRENT_LLM_CALLS,
                       help='Maximum number of concurrent LLM API calls (default: 6)')
    
    parser.add_argument('--checkpoint-name',
                       type=str,
                       default=DEFAULT_CHECKPOINT_NAME,
                       help=f'Custom checkpoint name for persistent storage (default: "{DEFAULT_CHECKPOINT_NAME}")')
    
    parser.add_argument('--skip-indexing',
                       action='store_true',
                       default=DEFAULT_SKIP_INDEXING,
                       help='Skip indexing if database already exists (useful for ablation studies)')
    
    parser.add_argument('--chunking-strategy',
                       type=str,
                       default=DEFAULT_CHUNKING_STRATEGY.value,
                       choices=[cs.value for cs in ChunkingStrategyEnum],
                       help=f'Chunking strategy to use (default: {DEFAULT_CHUNKING_STRATEGY.value})')
    
    parser.add_argument('--reranker',
                       type=str,
                       default=DEFAULT_RERANKER.value,
                       choices=[r.value for r in Reranker],
                       help=f'Reranker to use (default: {DEFAULT_RERANKER.value})')
    
    return parser.parse_args()

def main() -> None:
    args = parse_arguments()
    
    # Convert string arguments back to enum values
    rag_system = RAG(args.rag_system)
    dataset = RegisteredDataset(args.dataset)
    generation_config = GenerationConfiguration(args.generation)
    encoder = Encoder(args.encoder)
    chunking_strategy = ChunkingStrategyEnum(args.chunking_strategy)
    reranker_strategy = Reranker(args.reranker)
    
    backend = LLMBackend(args.backend)
    llm_name_enum = LLMName[args.llm]

    experiment_name = prepare_experiment_name(
        run_on_cluster=args.run_on_cluster, 
        rag_system=rag_system, 
        dataset=dataset, 
        generation=generation_config,
        backend=backend,
        comment=args.comment if args.comment else None,
    )
    log = prepare_log(experiment_name)

    llm = prepare_llm(
        args.run_on_cluster,
        log,
        backend=backend.value,
        llm_name=llm_name_enum.value,
        max_concurrent_llm_executions=args.max_concurrent,
    )

    if args.checkpoint_name and args.checkpoint_name.strip():
        log.info(f"💾 Initializing {rag_system} with checkpoint: {args.checkpoint_name.strip()}")
    else:
        log.info(f"💾 Initializing {rag_system}…")
    
    # If skip-indexing is set, also skip database reset to preserve existing data
    if args.skip_indexing:
        log.info(f"🔒 Preserving existing database (skip-indexing enabled)")
    
    rag = prepare_rag_system(
        rag_system=rag_system, 
        generation=generation_config, 
        encoding_strategy=encoder,
        chunking_strategy=chunking_strategy,
        log=log, 
        llm=llm,
        checkpoint_name=args.checkpoint_name,
        skip_db_reset=args.skip_indexing,
    )

    log.info("🗎 Preparing dataset %s", dataset)
    dataset_obj = prepare_dataset(dataset)

    if args.skip_indexing:
        log.info(f"⏭️ Skip-indexing flag set, checking database status...")
        
        # Check if the database is actually populated
        is_populated = False
        if hasattr(rag, 'env') and hasattr(rag.env, 'is_database_populated'):
            is_populated = rag.env.is_database_populated()
            if is_populated:
                log.info(f"✅ Database is populated, skipping indexing as requested")
            else:
                log.warning(f"⚠️ Database appears empty but --skip-indexing was set. This may cause retrieval to fail!")
                log.warning(f"⚠️ Consider running without --skip-indexing first to populate the database.")
        else:
            log.info(f"⏭️ Skipping indexing (--skip-indexing flag set, unable to verify database status)")
    else:
        log.info(f"🔍 Starting indexing for {len(dataset_obj.documents)} documents")
        try:
            indexing_evaluation_pipeline(rag, dataset_obj.documents, log, max_concurrent=args.max_concurrent)
        except Exception as e:
            log.error("❌ Indexing failed, stopping experiment: %s", str(e))
            raise

    log.info(f"🔍 Starting retrieval for {len(dataset_obj.qa_pairs)} qa pairs")
    # Prepare the reranker strategy object for the pipeline
    from experiments.base_experiment import prepare_reranker
    reranker = prepare_reranker(reranker_strategy, log)
    
    retrieval_evaluation_pipeline(rag, dataset_obj.qa_pairs, log, reranker=reranker)


if __name__ == "__main__":
    main()
