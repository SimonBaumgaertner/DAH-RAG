from pathlib import Path
import sys
import os
import argparse

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.data_classes.enums import RAG, RegisteredDataset, Encoder, LLMBackend, GenerationConfiguration, LLMName, Reranker
from experiments.base_experiment import *
from common.data_classes.data_set import DataSet
from common.evaluation.scaling_pipeline import scaling_evaluation_pipeline

# Constants for defaults
DEFAULT_RUN_ON_CLUSTER = False
DEFAULT_RAG_SYSTEM = RAG.DocAwareHybridRAG
DEFAULT_ENCODER = Encoder.Qwen3_600M
DEFAULT_BACKEND = LLMBackend.OpenRouter
DEFAULT_LLM = LLMName.Llama_3_3_70B
DEFAULT_MAX_CONCURRENT_LLM_CALLS = 6
DEFAULT_COMMENT = ""
DEFAULT_CHECKPOINT_NAME = "DocAwareHybridRAG_Scaling_2"
DEFAULT_RERANKER = Reranker.NoRerank # Jina_Reranker_v3_600M, NoRerank
DEFAULT_SCALING_STEPS = [1000, 2000, 3000, 4000, 5000, 7500, 10000, 15000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Scaling experiment runner')
    
    parser.add_argument('--run-on-cluster', 
                       action='store_true', 
                       default=DEFAULT_RUN_ON_CLUSTER,
                       help=f'Run on cluster? (default: {DEFAULT_RUN_ON_CLUSTER})')
    
    parser.add_argument('--rag-system', 
                       type=str, 
                       default=DEFAULT_RAG_SYSTEM.value,
                       choices=[rag.value for rag in RAG],
                       help=f'RAG system to use (default: {DEFAULT_RAG_SYSTEM.value})')
    
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
                       
    parser.add_argument('--scaling-steps',
                       nargs='+',
                       type=int,
                       default=DEFAULT_SCALING_STEPS,
                       help=f'List of total document counts to test (default: {DEFAULT_SCALING_STEPS})')
    
    parser.add_argument('--reranker',
                       type=str,
                       default=DEFAULT_RERANKER.value,
                       choices=[r.value for r in Reranker],
                       help=f'Reranker to use (default: {DEFAULT_RERANKER.value})')
    
    return parser.parse_args()

def main() -> None:
    args = parse_arguments()
    
    # Fixed configurations for Scaling Experiment
    dataset_enum = RegisteredDataset.Scaling
    generation_config = GenerationConfiguration.NoGen # no generation in scaling experiment
    
    # Convert string arguments back to enum values
    rag_system = RAG(args.rag_system)
    encoder = Encoder(args.encoder)
    backend = LLMBackend(args.backend)
    llm_name_enum = LLMName[args.llm]
    reranker_strategy = Reranker(args.reranker)

    experiment_name = prepare_experiment_name(
        run_on_cluster=args.run_on_cluster, 
        rag_system=rag_system, 
        dataset=dataset_enum, 
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
    
    rag = prepare_rag_system(
        rag_system=rag_system, 
        generation=generation_config, 
        encoding_strategy=encoder, 
        log=log, 
        llm=llm,
        checkpoint_name=args.checkpoint_name
    )

    log.info("🗎 Preparing dataset %s", dataset_enum)
    dataset_obj = prepare_dataset(dataset_enum)
    
    # Split dataset into base and distractors
    base_root = Path(__file__).parent.parent / "data" / RegisteredDataset.HotpotQA_1k.value
    base_count = 0
    if base_root.exists():
        base_count = sum(1 for x in base_root.iterdir() if x.is_dir())
    
    if len(dataset_obj.documents) < base_count:
        log.warning(f"Dataset has fewer documents ({len(dataset_obj.documents)}) than expected base count ({base_count}). Using all as base.")
        base_docs = dataset_obj.documents
        distractor_docs = []
    else:
        base_docs = dataset_obj.documents[:base_count]
        distractor_docs = dataset_obj.documents[base_count:]

    log.info(f"🔍 Starting SCALING experiment with {len(base_docs)} base docs and {len(distractor_docs)} distractors available")
    
    try:
        # Prepare the reranker strategy object for the pipeline
        from experiments.base_experiment import prepare_reranker
        reranker = prepare_reranker(reranker_strategy, log)

        scaling_evaluation_pipeline(
            rag=rag, 
            base_documents=base_docs, 
            distractor_documents=distractor_docs, 
            qa_pairs=dataset_obj.qa_pairs, 
            log=log,
            scaling_steps=args.scaling_steps,
            reranker=reranker
        )
    except Exception as e:
        log.error("❌ Scaling experiment failed: %s", str(e))
        raise

    log.info("🏁 Scaling experiment finished.")

if __name__ == "__main__":
    main()
