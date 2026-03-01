from __future__ import annotations

from typing import Dict, List, Optional

from common.data_classes.evaluation import LLMCallContext
from common.data_classes.qa import QuestionAnswerPair
from common.data_classes.rag_system import Chunk, Retriever
from common.logging.run_logger import RunLogger

import sys
from pathlib import Path

# Add the local HippoRAG src directory to the path
hippo_src_path = Path(__file__).parent / "HippoRAG" / "src"
sys.path.insert(0, str(hippo_src_path))

from hipporag.HippoRAG import HippoRAG as HippoRAGEngine
from hipporag.utils.misc_utils import compute_mdhash_id


class HippoRAGRetriever(Retriever):
    """Translate HippoRAG query results into our :class:`Chunk` abstraction."""

    def __init__(
        self,
        *,
        rag: HippoRAGEngine,
        log: RunLogger,
        chunk_store: Dict[str, Chunk],
    ) -> None:
        self.rag = rag
        self.log = log
        self._chunk_store = chunk_store

    def retrieve(self, question: str, k: int = 5, qa_pair: Optional[QuestionAnswerPair] = None) -> List[Chunk]:
        self.log.info("🔍 HippoRAG retrieving top %d chunk(s) for query: %s", k, question)
        
        # Set LLM context for retrieval (e.g. for fact reranking)
        if hasattr(self.rag.llm_model, 'set_context'):
            # Use the question_id if available to match with answer_track in logs/analysis
            identifier = qa_pair.question_id if qa_pair else question
            self.rag.llm_model.set_context(LLMCallContext.RETRIEVAL, identifier=identifier)
            
        solutions = self.rag.retrieve([question], num_to_retrieve=k)
        if not solutions:
            self.log.warning("⚠️ HippoRAG returned no results for query '%s'", question)
            return []

        solution = solutions[0]
        docs = solution.docs or []
        scores_array = solution.doc_scores if solution.doc_scores is not None else []

        results: List[Chunk] = []
        for idx, text in enumerate(docs[:k]):
            hash_id = compute_mdhash_id(text, prefix="chunk-")
            stored = self._chunk_store.get(hash_id)
            score = None
            if scores_array is not None:
                try:
                    score = float(scores_array[idx])
                except Exception:
                    score = None

            metadata: Dict[str, str] = {}
            if stored and stored.metadata:
                metadata.update(stored.metadata)
            metadata["hipporag_rank"] = str(idx)
            if score is not None:
                metadata["hipporag_score"] = str(score)

            results.append(
                Chunk(
                    chunk_id=hash_id,
                    text=text,
                    score=score,
                    doc_id=stored.doc_id if stored else None,
                    metadata=metadata or None,
                )
            )

        self.log.info("🔎 HippoRAG returned %d chunk(s)", len(results))
        return results
