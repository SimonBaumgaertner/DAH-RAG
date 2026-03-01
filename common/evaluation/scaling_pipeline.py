from typing import List, Optional
from common.data_classes.rag_system import RAGSystem
from common.data_classes.documents import Document
from common.logging.run_logger import RunLogger
from common.data_classes.evaluation import EntryType
from common.evaluation.pipeline import indexing_evaluation_pipeline, retrieval_evaluation_pipeline
from common.strategies.reranking import RerankingStrategy

def scaling_evaluation_pipeline(
    rag: RAGSystem, 
    base_documents: List[Document], 
    distractor_documents: List[Document], 
    qa_pairs: list, 
    log: RunLogger,
    scaling_steps: List[int],
    reranker: Optional[RerankingStrategy] = None
):
    """
    Scaling evaluation pipeline.
    
    Iterates through specific total document counts (scaling_steps).
    For each step:
    1. Indexes the *new* documents required to reach the target total.
    2. Runs the full retrieval evaluation.
    3. Tracks indexing and retrieval times using SCALING_INDEXING_TRACK and SCALING_RETRIEVAL_TRACK.
    """
    log.info("🚀 Starting SCALING evaluation pipeline.")
    
    # Combine all potential documents into a single ordered list
    # The experiment script ensures base_documents + distractor_documents structure
    all_documents = base_documents + distractor_documents
    total_available = len(all_documents)
    
    log.info(f"📊 Total available documents: {total_available} ({len(base_documents)} Base + {len(distractor_documents)} Distractors)")
    log.info(f"📏 Scaling steps: {scaling_steps}")
    
    current_indexed_count = 0
    
    for target_total in scaling_steps:
        # Validate target
        if target_total > total_available:
            log.warning(f"⚠️ Target {target_total} exceeds available documents {total_available}. clamping to max.")
            target_total = total_available
            
        if target_total <= current_indexed_count:
            log.info(f"⏭️  Target {target_total} already covered (current: {current_indexed_count}). Skipping.")
            continue
            
        # Determine new documents to index
        # We index from current_indexed_count up to target_total
        new_docs_to_index = all_documents[current_indexed_count:target_total]
        
        log.info(f"🧱 Step target: {target_total} docs. Indexing {len(new_docs_to_index)} new documents...")
        
        # --- Indexing Phase ---
        # Track Indexing Time for this specific step (identifier = target_total)
        log.start(stopwatch_id=EntryType.SCALING_INDEXING_TRACK.value)
        
        try:
            # We assume indexing_evaluation_pipeline adds to the existing index
            indexing_evaluation_pipeline(rag, new_docs_to_index, log)
        except Exception as e:
            log.error(f"❌ Indexing failed at step {target_total}: {e}")
            raise # Stop experiment if indexing fails
            
        indexing_duration = log.stop(stopwatch_id=EntryType.SCALING_INDEXING_TRACK.value)
        log.track(EntryType.SCALING_INDEXING_TRACK.value, str(target_total), f"{indexing_duration:.3f}")
        
        # Update state
        current_indexed_count = target_total
        
        # --- Retrieval Phase ---
        log.info(f"🔍 Running retrieval evaluation for {target_total} documents...")
        
        # Track Retrieval Time for this specific step (identifier = target_total)
        log.start(stopwatch_id=EntryType.SCALING_RETRIEVAL_TRACK.value)
        
        try:
            retrieval_evaluation_pipeline(rag, qa_pairs, log, reranker=reranker)
        except Exception as e:
             log.error(f"❌ Retrieval failed at step {target_total}: {e}")
             # We might optionally continue to next step? But retrieval failure is bad.
             raise

        retrieval_duration = log.stop(stopwatch_id=EntryType.SCALING_RETRIEVAL_TRACK.value)
        log.track(EntryType.SCALING_RETRIEVAL_TRACK.value, str(target_total), f"{retrieval_duration:.3f}")

    log.info("🏁 Finished scaling evaluation pipeline.")
