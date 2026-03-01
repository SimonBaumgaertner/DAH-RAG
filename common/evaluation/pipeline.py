from typing import List, Dict, Optional
import asyncio
import concurrent.futures
import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path

from common.data_classes.data_set import DataSet
from common.data_classes.documents import Document
from common.data_classes.evaluation import EntryType
from common.data_classes.qa import QuestionAnswerPair, Proof
from common.data_classes.rag_system import RAGSystem, Chunk
from common.evaluation.evalutation_util import get_correctness, get_correctness_and_rouge_l, get_proof_map
from common.logging.run_logger import RunLogger
from common.strategies.reranking import RerankingStrategy


# Global thread pool executor for long-running operations
_long_running_executor: concurrent.futures.ThreadPoolExecutor | None = None


def _get_long_running_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Get or create a thread pool executor configured for long-running operations."""
    global _long_running_executor
    if _long_running_executor is None:
        # Create a custom thread pool with appropriate settings for long-running tasks
        _long_running_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2,  # Limit concurrent operations to avoid resource exhaustion
            thread_name_prefix="long-running-indexing",
            # No timeout on individual tasks - let them run as long as needed
        )
    return _long_running_executor


@asynccontextmanager
async def _long_running_executor_context():
    """Context manager for the long-running executor."""
    executor = _get_long_running_executor()
    try:
        yield executor
    finally:
        # Don't shutdown the executor here as it's reused across operations
        pass


async def index_document_async(rag: RAGSystem, document: Document, log: RunLogger, idx: int, total_docs: int) -> None:
    """Async wrapper for indexing a single document with proper long-running operation support."""
    log.info("🔄 [%d/%d] Indexing document: %s", idx, total_docs, document.id)
    log.set_indexing_context(document.id)
    log.start(stopwatch_id=document.id)
    indexing_success = False

    try:
        # Use our custom long-running executor instead of the default one
        async with _long_running_executor_context() as executor:
            loop = asyncio.get_event_loop()
            # Run with no timeout - let it take as long as needed for large documents
            await loop.run_in_executor(executor, rag.index_document, document)
        indexing_success = True
    except Exception as e:
        log.error("❌ Failed to index document %s: %s", document.id, e)
    finally:
        indexing_time_ms = log.stop(stopwatch_id=document.id)
        if indexing_success and indexing_time_ms is not None:
            # Convert to seconds for better readability for long operations
            indexing_time_seconds = indexing_time_ms / 1000.0
            log.track(
                entry_type=EntryType.DOCUMENT_INDEXING_TRACK.value,
                identifier=document.id,
                value=f"{indexing_time_ms:.3f}"
            )
            if indexing_time_seconds > 60:
                log.info("✅ [%d/%d] Indexed document %s in %.1f minutes", idx, total_docs, document.id, indexing_time_seconds / 60.0)
            else:
                log.info("✅ [%d/%d] Indexed document %s in %.1f seconds", idx, total_docs, document.id, indexing_time_seconds)
        log.set_indexing_context(None)


async def _indexing_pipeline_async(rag: RAGSystem, documents: List[Document], log: RunLogger, max_concurrent: int, continue_on_errors: bool = True) -> None:
    """Async implementation of the indexing pipeline with sequential processing for embeddings."""
    log.info("📄 Starting indexing of %d documents (sequential processing for embedding safety).", len(documents))
    log.info("⏱️ Long-running operations are supported - documents may take several minutes to hours to process.")
    log.start(stopwatch_id=EntryType.FULL_INDEXING_TRACK.value)

    total_docs = len(documents)
    failed_documents = []
    
    # Process documents sequentially to avoid embedding concurrency issues
    for idx, document in enumerate(documents, start=1):
        try:
            # Log estimated time for large documents
            if hasattr(document, 'text') and document.text and len(document.text) > 100000:  # > 100k chars
                log.info("📚 Large document detected (%d chars) - this may take 10+ minutes", len(document.text))
            
            await index_document_async(rag, document, log, idx, total_docs)
        except Exception as e:
            failed_documents.append((idx, document.id, e))
            log.error("❌ Failed to index document [%d] %s: %s", idx, document.id, e)
            # Continue processing other documents but track the failure
    
    # Check for any failures and handle based on configuration
    if failed_documents:
        success_count = total_docs - len(failed_documents)
        log.warning("⚠️ Indexing completed with %d successful and %d failed documents", success_count, len(failed_documents))
        
        # Log details of failed documents
        for idx, doc_id, error in failed_documents:
            log.error("❌ Failed document [%d] %s: %s", idx, doc_id, error)
        
        if not continue_on_errors:
            error_msg = f"Failed to index {len(failed_documents)} document(s): "
            for idx, doc_id, error in failed_documents:
                error_msg += f"[{idx}] {doc_id}: {error}; "
            raise RuntimeError(error_msg.rstrip("; "))
        else:
            log.info("✅ Continuing despite failures - %d documents successfully indexed", success_count)

    # Finalize indexing if the indexer supports it (e.g., build RAPTOR tree after all chunks are ready)
    if hasattr(rag.indexer, 'finalize_indexing'):
        log.info("🔧 Finalizing indexing (e.g., building RAPTOR tree from all chunks)...")
        rag.indexer.finalize_indexing()

    total_indexing_time_ms = log.stop(stopwatch_id=EntryType.FULL_INDEXING_TRACK.value)
    log.info("📦 Finished indexing all documents in %.3f ms", total_indexing_time_ms)
    log.track(entry_type=EntryType.FULL_INDEXING_TRACK.value, identifier="total",
              value=f"{total_indexing_time_ms:.3f}")


def _cleanup_long_running_executor():
    """Clean up the long-running executor."""
    global _long_running_executor
    if _long_running_executor is not None:
        _long_running_executor.shutdown(wait=True)
        _long_running_executor = None


def _update_ranking_json_with_correct_chunks(
    question: str,
    chunks: List[Chunk],
    proof_map: Dict[Proof, int],
    log: RunLogger
) -> None:
    """Update the ranking JSON file with correct chunk_ids based on proof_map.
    
    This function finds the ranking JSON file (using question hash), extracts
    chunk_ids from chunks that match proofs (where proof_map value != -1),
    and adds a 'correct_chunk_ids' field to the JSON file.
    
    Args:
        question: The question string used to identify the ranking JSON file
        chunks: List of Chunk objects returned by retrieval
        proof_map: Dictionary mapping Proof objects to chunk ranks (indices)
        log: Logger for tracking operations
    """
    # Hash the question to create the filename (same as in _save_rankings_json)
    question_hash = hashlib.sha256(question.encode('utf-8')).hexdigest()
    
    # Find the rankings directory
    rankings_dir = Path(__file__).parent.parent / "strategies" / "graph_search" / "rankings"
    
    # Use run_id from logger if available
    run_id = log.run_id if hasattr(log, "run_id") else "unknown_run"
    json_path = rankings_dir / run_id / f"{question_hash}.json"
    
    # Check if the JSON file exists
    if not json_path.exists():
        log.info("📄 Ranking JSON file not found: %s (question may not have been processed yet)", json_path)
        return
    
    try:
        # Load the existing JSON data
        with json_path.open('r', encoding='utf-8') as f:
            rankings_data = json.load(f)
        
        # Extract correct chunk_ids from proof_map
        # proof_map maps Proof objects to chunk ranks (indices in chunks list)
        correct_chunk_ids: List[str] = []
        for proof, chunk_rank in proof_map.items():
            if chunk_rank != -1 and 0 <= chunk_rank < len(chunks):
                chunk_id = chunks[chunk_rank].chunk_id
                if chunk_id not in correct_chunk_ids:
                    correct_chunk_ids.append(chunk_id)
        
        # Update the JSON data with correct_chunk_ids
        rankings_data["correct_chunk_ids"] = correct_chunk_ids
        
        # Save the updated JSON data
        with json_path.open('w', encoding='utf-8') as f:
            json.dump(rankings_data, f, indent=2, ensure_ascii=False)
        
        log.info("💾 Updated ranking JSON with %d correct chunk_ids: %s", len(correct_chunk_ids), json_path)
        
    except Exception as e:
        log.error("❌ Failed to update ranking JSON file %s: %s", json_path, e)
        raise


def indexing_evaluation_pipeline(rag: RAGSystem, documents: List[Document], log: RunLogger, max_concurrent: int = 3):
    """Indexing pipeline with configurable concurrent document processing."""
    log.info("🚀 Starting indexing evaluation pipeline.")
    try:
        # Run the async pipeline
        asyncio.run(_indexing_pipeline_async(rag, documents, log, max_concurrent, continue_on_errors=True))
    except Exception as e:
        log.exception("❌ Exception during indexing evaluation pipeline: %s", str(e))
        raise  # Re-raise the exception to stop the entire process
    finally:
        # Clean up the executor when done
        _cleanup_long_running_executor()


def retrieval_evaluation_pipeline(
    rag: RAGSystem,
    qa_pairs: List[QuestionAnswerPair],
    log: RunLogger,
    reranker: Optional[RerankingStrategy] = None
):
    """Retrieval pipeline with proper async context for FastGraphRAG."""
    log.info("🧠 Starting retrieval evaluation pipeline.")
    try:
        log.info("🔍 Starting retrieval for %d questions", len(qa_pairs))
        total_questions = len(qa_pairs)
        
        for idx, qa_pair in enumerate(qa_pairs, start=1):
            # ---- Chunk retrieval ---- #
            log.info("📥 [%d/%d] Retrieving chunks for question: %s", idx, total_questions, qa_pair.question)
            log.start(stopwatch_id=EntryType.CHUNK_RETRIEVAL_TRACK.value)
            
            # Set retrieval context for LLM calls during retrieval
            log.set_retrieval_context(qa_pair.question_id)

            # ----- Reranker ----- #
            chunks = rag.retriever.retrieve(qa_pair.question, 50, qa_pair=qa_pair)

            if reranker and reranker.__class__.__name__ != "NoRerank":
                top_k = 25
                log.info("🔄 Reranking top %d chunks using %s...", top_k, reranker.__class__.__name__)
                chunks = reranker.rerank(qa_pair.question, chunks, top_k)
                log.info("✅ Reranked top %d chunks using %s", top_k, reranker.__class__.__name__)

            # Clear retrieval context after retrieval is complete
            log.set_retrieval_context(None)
            chunk_retrieval_time_ms = log.stop(stopwatch_id=EntryType.CHUNK_RETRIEVAL_TRACK.value)
            log.info("📚 [%d/%d] Retrieved top %d chunks in %d ms for question: %s", idx, total_questions, len(chunks), chunk_retrieval_time_ms, qa_pair.question)
            log.track(EntryType.CHUNK_RETRIEVAL_TRACK.value, qa_pair.question_id, str(chunk_retrieval_time_ms))

            # ------ Generation ------ #
            log.info("📝 [%d/%d] Generating answer for question: %s", idx, total_questions, qa_pair.question)
            log.start(EntryType.ANSWER_TRACK.value)

            # ------------------------- #
            generated_answer = rag.generator.generate(qa_pair, chunks)
            # ------------------------- #

            generation_ms = log.stop(stopwatch_id=EntryType.ANSWER_TRACK.value)
            log.info("⚡ [%d/%d] Generated answer in %.3fms", idx, total_questions, generation_ms)
            log.track(EntryType.ANSWER_TRACK.value, qa_pair.question_id, f"{generation_ms:.3f}")

            # ------ Evaluation ------- #
            log.info("🧪 [%d/%d] Evaluating response. Answer given: %s | Correct answer: %s", idx, total_questions, generated_answer, qa_pair.correct_answer)
            correctness, rouge_l_score = get_correctness_and_rouge_l(qa_pair, generated_answer)
            proof_map = get_proof_map(qa_pair, chunks)
            amount_supported_proofs = sum(1 for v in proof_map.values() if v != -1)
            amount_total_proofs = len(proof_map)
            # ----- Track results ----- #
            log.track(EntryType.CORRECTNESS_TRACK.value, qa_pair.question_id, correctness.value)
            log.track(EntryType.ROUGE_L_TRACK.value, qa_pair.question_id, f"{rouge_l_score:.4f}")
            log.track(EntryType.TOTAL_PROOFS_TRACK.value, qa_pair.question_id, str(amount_total_proofs))
            for proof in proof_map:
                log.track(EntryType.PROOF_TRACK.value, qa_pair.question_id, str(proof_map.get(proof)))

            log.info("📊 [%d/%d] %s (ROUGE-L: %.3f) — Supported proofs: %d/%d", idx, total_questions, correctness.name.capitalize(), rouge_l_score, amount_supported_proofs, amount_total_proofs)

            # ----- Update ranking JSON with correct chunk_ids (only for DocumentAwareHybridRAG) ----- #
            try:
                from rag_approaches.doc_aware_hybrid_RAG.doc_aware_hybrid_RAG import DocumentAwareHybridRAG
                if isinstance(rag, DocumentAwareHybridRAG):
                    _update_ranking_json_with_correct_chunks(qa_pair.question, chunks, proof_map, log)
            except ImportError:
                # DocumentAwareHybridRAG might not be available, skip silently
                pass
            except Exception as e:
                log.warning("⚠️ Failed to update ranking JSON with correct chunk_ids: %s", e)

    except Exception as e:
        log.exception("❌ Exception during evaluation pipeline: %s", str(e))
        raise  # Re-raise the exception to stop the entire process
    finally:
        log.info("🏁 Finished retrieval pipeline evaluation.")
