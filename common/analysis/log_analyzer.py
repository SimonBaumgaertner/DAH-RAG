from __future__ import annotations


import csv
from collections import defaultdict
from statistics import mean
from pathlib import Path
from typing import Dict, List, Iterable
from datetime import datetime

from common.data_classes.evaluation import EntryType, CorrectnessType


class LogAnalyzer:
    """Utility class to compute evaluation metrics from a ``RunLogger`` CSV, including document and question counts."""

    def __init__(self, *, csv_file: str | Path) -> None:
        self.csv_file = Path(csv_file)
        with self.csv_file.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.rows = [row for row in reader]

        self._by_type: Dict[str, List[dict]] = defaultdict(list)
        for row in self.rows:
            self._by_type[row["entry_type"]].append(row)

    # ------------------------------------------------------------------ helpers
    def _values(self, t: EntryType | str, ids: Iterable[str] | None = None) -> List[float]:
        etype = t.value if isinstance(t, EntryType) else str(t)
        entries = self._by_type.get(etype, [])
        if ids is not None:
            ids = set(ids)
            entries = [r for r in entries if r["identifier"] in ids]
        return [float(r["value"]) for r in entries]

    def total_execution_time_formatted(self) -> str:
        """Return total execution time as a formatted string: HH h MM m SS s."""
        timestamps = [row["timestamp"] for row in self.rows if row.get("timestamp")]
        if not timestamps:
            return "0 s"

        try:
            parsed_times = [datetime.fromisoformat(ts) for ts in timestamps]
        except ValueError:
            return "Invalid timestamps"

        total_seconds = int((max(parsed_times) - min(parsed_times)).total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours} h {minutes} m {seconds} s"

    def num_documents(self) -> int:
        """Return the number of unique documents that were indexed."""
        doc_ids = {r["identifier"] for r in self._by_type.get(EntryType.DOCUMENT_INDEXING_TRACK.value, [])}
        return len(doc_ids)

    def num_questions(self) -> int:
        """Return the number of unique questions that were answered."""
        q_ids = {r["identifier"] for r in self._by_type.get(EntryType.ANSWER_TRACK.value, [])}
        return len(q_ids)

    def num_chunks(self) -> int:
        """Return the total number of chunks that were indexed."""
        chunk_counts = self._values(EntryType.CHUNK_COUNT_TRACK)
        return int(sum(chunk_counts)) if chunk_counts else 0

    # ---------------------------------------------------------------- metrics
    def indexing_time_avg(self) -> float:
        """Return average time per document during indexing (excluding finalization like tree building)."""
        vals = self._values(EntryType.DOCUMENT_INDEXING_TRACK)
        return mean(vals) if vals else 0.0

    def total_indexing_time(self) -> float:
        """Return total indexing time (including finalization like tree building) in milliseconds."""
        # Look for FULL_INDEXING_TRACK with identifier "total"
        entries = self._by_type.get(EntryType.FULL_INDEXING_TRACK.value, [])
        total_entry = next((e for e in entries if e.get("identifier") == "total"), None)
        if total_entry:
            return float(total_entry["value"])
        # Fallback: sum all document indexing times if total not available
        vals = self._values(EntryType.DOCUMENT_INDEXING_TRACK)
        return sum(vals) if vals else 0.0

    def llm_calls_indexing(self) -> int:
        """Return total number of LLM calls during indexing (including tree building, etc.)."""
        calls = self._values(EntryType.LLM_INDEXING_GENERATION_TIME_TRACK)
        return len(calls)

    def avg_input_tokens_indexing(self) -> float:
        """Return average input tokens per LLM call during indexing (including tree building, etc.)."""
        vals = self._values(EntryType.LLM_INDEXING_INPUT_TOKENS_TRACK)
        return mean(vals) if vals else 0.0

    def avg_output_tokens_indexing(self) -> float:
        """Return average output tokens per LLM call during indexing (including tree building, etc.)."""
        vals = self._values(EntryType.LLM_INDEXING_OUTPUT_TOKENS_TRACK)
        return mean(vals) if vals else 0.0

    def _times_per_q(self, entry_type) -> Dict[str, float]:
        times: Dict[str, float] = defaultdict(float)
        for row in self._by_type.get(entry_type.value, []):
            times[row["identifier"]] += float(row["value"])
        return times

    def chunk_retrieval_times_per_q(self) -> Dict[str, float]:
        return self._times_per_q(EntryType.CHUNK_RETRIEVAL_TRACK)

    def generation_times_per_q(self) -> Dict[str, float]:
        return self._times_per_q(EntryType.ANSWER_TRACK)

    def chunk_retrieval_time_avg(self) -> float:
        times = list(self.chunk_retrieval_times_per_q().values())
        return mean(times) if times else 0.0

    def generation_time_avg(self) -> float:
        times = list(self.generation_times_per_q().values())
        return mean(times) if times else 0.0

    def retrieval_time_avg(self) -> float:
        """Return average retrieval time per question."""
        chunk_times = list(self.chunk_retrieval_times_per_q().values())
        return mean(chunk_times) if chunk_times else 0.0

    def avg_llm_time_indexing(self) -> float:
        """Return average time per LLM call during indexing (including tree building, etc.)."""
        llm_times = self._values(EntryType.LLM_INDEXING_GENERATION_TIME_TRACK)
        return mean(llm_times) if llm_times else 0.0

    def llm_calls_retrieval(self) -> float:
        """Return average number of LLM calls per question during chunk retrieval."""
        q_ids = [r["identifier"] for r in self._by_type.get(EntryType.ANSWER_TRACK.value, [])]
        calls_per_q: Dict[str, int] = defaultdict(int)
        for row in self._by_type.get(EntryType.LLM_RETRIEVAL_GENERATION_TIME_TRACK.value, []):
            if row["identifier"] in q_ids:
                calls_per_q[row["identifier"]] += 1
        return mean(calls_per_q.values()) if calls_per_q else 0.0

    def avg_input_tokens_retrieval(self) -> float:
        """Return average input tokens per LLM call during chunk retrieval."""
        q_ids = [r["identifier"] for r in self._by_type.get(EntryType.ANSWER_TRACK.value, [])]
        vals_1 = self._values(EntryType.LLM_RETRIEVAL_INPUT_TOKENS_TRACK, ids=q_ids)
        vals_2 = self._values("prompt_length", ids=q_ids)
        all_vals = vals_1 + vals_2
        return mean(all_vals) if all_vals else 0.0

    def avg_output_tokens_retrieval(self) -> float:
        """Return average output tokens per LLM call during chunk retrieval."""
        q_ids = [r["identifier"] for r in self._by_type.get(EntryType.ANSWER_TRACK.value, [])]
        vals = self._values(EntryType.LLM_RETRIEVAL_OUTPUT_TOKENS_TRACK, ids=q_ids)
        return mean(vals) if vals else 0.0

    def avg_llm_time_retrieval(self) -> float:
        """Return average time per LLM call during chunk retrieval."""
        q_ids = [r["identifier"] for r in self._by_type.get(EntryType.ANSWER_TRACK.value, [])]
        llm_times = self._values(EntryType.LLM_RETRIEVAL_GENERATION_TIME_TRACK, ids=q_ids)
        return mean(llm_times) if llm_times else 0.0

    def llm_calls_generation(self) -> float:
        """Return average number of LLM calls per question during answer generation."""
        q_ids = [r["identifier"] for r in self._by_type.get(EntryType.ANSWER_TRACK.value, [])]
        calls_per_q: Dict[str, int] = defaultdict(int)
        for row in self._by_type.get(EntryType.LLM_QA_GENERATION_TIME_TRACK.value, []):
            if row["identifier"] in q_ids:
                calls_per_q[row["identifier"]] += 1
        return mean(calls_per_q.values()) if calls_per_q else 0.0

    def avg_input_tokens_generation(self) -> float:
        """Return average input tokens per LLM call during answer generation."""
        q_ids = [r["identifier"] for r in self._by_type.get(EntryType.ANSWER_TRACK.value, [])]
        vals = self._values(EntryType.LLM_QA_INPUT_TOKENS_TRACK, ids=q_ids)
        return mean(vals) if vals else 0.0

    def avg_output_tokens_generation(self) -> float:
        """Return average output tokens per LLM call during answer generation."""
        q_ids = [r["identifier"] for r in self._by_type.get(EntryType.ANSWER_TRACK.value, [])]
        vals = self._values(EntryType.LLM_QA_OUTPUT_TOKENS_TRACK, ids=q_ids)
        return mean(vals) if vals else 0.0

    def avg_llm_time_generation(self) -> float:
        """Return average time per LLM call during answer generation."""
        q_ids = [r["identifier"] for r in self._by_type.get(EntryType.ANSWER_TRACK.value, [])]
        llm_times = self._values(EntryType.LLM_QA_GENERATION_TIME_TRACK, ids=q_ids)
        return mean(llm_times) if llm_times else 0.0

    def _proof_ranks_per_q(self) -> Dict[str, List[int]]:
        ranks: Dict[str, List[int]] = defaultdict(list)
        for row in self._by_type.get(EntryType.PROOF_TRACK.value, []):
            ranks[row["identifier"]].append(int(row["value"]))
        return ranks

    def recall_at_k(self, k: int) -> float:
        """Return recall@k: average percentage of proofs found in the top k retrieved chunks per question.
        
        For each question, computes recall@k as: (proofs found in top k) / (total proofs for that question).
        Returns the average recall across all questions (macro-averaging), so each question has equal weight
        regardless of how many proofs it has.
        
        Example: 
        - Question 1: 5 proofs, 3 found in top 4 → recall@4 = 3/5 = 60%
        - Question 2: 2 proofs, 2 found in top 4 → recall@4 = 2/2 = 100%
        - Overall recall@4 = (60% + 100%) / 2 = 80%
        """
        ranks = self._proof_ranks_per_q()
        if not ranks:
            return 0.0
        
        question_recalls: List[float] = []
        
        for question_ranks in ranks.values():
            total_proofs = len(question_ranks)
            if total_proofs == 0:
                continue
            
            # Count proofs found in top k (rank < k, excluding -1 which means not found)
            found_in_top_k = sum(1 for r in question_ranks if r != -1 and 0 <= r < k)
            question_recall = found_in_top_k / total_proofs
            question_recalls.append(question_recall)
        
        if not question_recalls:
            return 0.0
        
        # Macro-averaging: average of per-question recalls
        return 100.0 * mean(question_recalls)

    def qa_accuracy(self) -> float:
        corrs = [row["value"] for row in self._by_type.get(EntryType.CORRECTNESS_TRACK.value, [])]
        if not corrs:
            return 0.0
        correct = sum(1 for c in corrs if c == CorrectnessType.CORRECT.value)
        return 100.0 * correct / len(corrs)

    def rouge_l_avg(self) -> float:
        """Return average ROUGE-L F1 score across all questions."""
        rouge_scores = self._values(EntryType.ROUGE_L_TRACK)
        return mean(rouge_scores) if rouge_scores else 0.0

    def rouge_l_median(self) -> float:
        """Return median ROUGE-L F1 score across all questions."""
        rouge_scores = self._values(EntryType.ROUGE_L_TRACK)
        if not rouge_scores:
            return 0.0
        sorted_scores = sorted(rouge_scores)
        n = len(sorted_scores)
        if n % 2 == 0:
            return (sorted_scores[n//2 - 1] + sorted_scores[n//2]) / 2
        else:
            return sorted_scores[n//2]

    def faithfulness(self, k: int) -> float:
        ranks = self._proof_ranks_per_q()
        corrs = {row["identifier"]: row["value"] for row in self._by_type.get(EntryType.CORRECTNESS_TRACK.value, [])}
        supported = [qid for qid, rs in ranks.items() if any(0 <= r < k for r in rs)] # partial support is enough
        if not supported:
            return 0.0
        correct_supported = sum(1 for qid in supported if corrs.get(qid) == CorrectnessType.CORRECT.value)
        return 100.0 * correct_supported / len(supported)

    def unsupported_accuracy(self, k: int) -> float:
        ranks = self._proof_ranks_per_q()
        corrs = {row["identifier"]: row["value"] for row in self._by_type.get(EntryType.CORRECTNESS_TRACK.value, [])}
        unsupported = [qid for qid, rs in ranks.items() if not any(0 <= r < k for r in rs)]
        if not unsupported:
            return 0.0
        correct = sum(1 for qid in unsupported if corrs.get(qid) == CorrectnessType.CORRECT.value)
        return 100.0 * correct / len(unsupported)

    # --------------------------------------------------------------- summary
    def summary(self, k: int = 5) -> str:
        """Return a human readable multi-line summary for the log CSV."""

        lines = [
            f"Analysis for {self.csv_file.name}:",
            f"Total Run Execution Time: {self.total_execution_time_formatted()}",
            f"Documents Indexed: {self.num_documents()}",
            f"Chunks Indexed: {self.num_chunks()}",
            f"Questions Answered: {self.num_questions()}",
            f"Avr Indexing Time (Per document): {self.indexing_time_avg():.3f} ms",
            f"Avr LLM Calls (Indexing): {self.llm_calls_indexing()}",
            f"Avg Input Tokens (Indexing): {self.avg_input_tokens_indexing():.3f}",
            f"Avg Output Tokens (Indexing): {self.avg_output_tokens_indexing():.3f}",
            f"Avr Retrieval Time (Per question): {self.retrieval_time_avg():.3f} ms",
            f"Avg LLM Calls (Retrieval): {self.llm_calls_retrieval():.3f}",
            f"Avg Input Tokens (Retrieval): {self.avg_input_tokens_retrieval():.3f}",
            f"Avg Output Tokens (Retrieval): {self.avg_output_tokens_retrieval():.3f}",
            f"Avr Generation Time (Per question): {self.generation_time_avg():.3f} ms",
            f"Avg LLM Calls (Generation): {self.llm_calls_generation():.3f}",
            f"Avg Input Tokens (Generation): {self.avg_input_tokens_generation():.3f}",
            f"Avg Output Tokens (Generation): {self.avg_output_tokens_generation():.3f}",
            f"Recall@{k}: {self.recall_at_k(k):.2f}%",
            f"QA Accuracy: {self.qa_accuracy():.2f}%",
            f"ROUGE-L Avg: {self.rouge_l_avg():.3f}",
            f"ROUGE-L Median: {self.rouge_l_median():.3f}",
            f"Faithfulness@{k}: {self.faithfulness(k):.2f}%",
            f"Unsupported Accuracy@{k}: {self.unsupported_accuracy(k):.2f}%",
        ]

        return "\n".join(lines)
