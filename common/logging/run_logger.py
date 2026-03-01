# run_logger.py
"""Lightweight run logger with nanosecond‑precision stopwatches and CSV export."""

from __future__ import annotations

import csv
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional
from common.data_classes.evaluation import EntryType, LLMCallContext

__all__ = ["RunLogger"]


class RunLogger:
    _LOG_FMT = "%(asctime)s | %(levelname)s | %(message)s"

    def __init__(self, *, run_id: str, log_dir: os.PathLike | str = "logs_and_tracks") -> None:
        self.run_id = str(run_id)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

        self.log = logging.getLogger(self.run_id)
        self.log.handlers.clear()
        self.log.setLevel(logging.INFO)

        fmt = logging.Formatter(self._LOG_FMT)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        self._log_file_path = self.log_dir / f"{self.run_id}.log"
        fh = logging.FileHandler(self._log_file_path, mode="a", encoding="utf‑8")
        fh.setFormatter(fmt)
        self.log.addHandler(sh)
        self.log.addHandler(fh)
        self.log.propagate = False
        
        # Make logging thread-safe by adding a lock
        self._log_lock = RLock()

        self._csv_path = self.log_dir / f"{self.run_id}.csv"
        first_open = not self._csv_path.exists()
        self._csv_fp = self._csv_path.open("a", newline="", encoding="utf‑8")
        self._csv_writer = csv.writer(self._csv_fp)
        if first_open:
            self._csv_writer.writerow(["run_id", "timestamp", "entry_type", "identifier", "value"])

        self._starts: Dict[str, List[int]] = {}
        self._indexing_doc_id: str | None = None
        self._retrieval_question_id: str | None = None

        self.info("🚀 Run %s started. Log file at %s", self.run_id, self._log_file_path)

    def debug(self, *a, **kw):
        with self._log_lock:
            self.log.debug(*a, **kw)

    def info(self, *a, **kw):
        with self._log_lock:
            self.log.info(*a, **kw)

    def warning(self, *a, **kw):
        with self._log_lock:
            self.log.warning(*a, **kw)

    def error(self, *a, **kw):
        with self._log_lock:
            self.log.error(*a, **kw)

    def critical(self, *a, **kw):
        with self._log_lock:
            self.log.critical(*a, **kw)

    def exception(self, *a, **kw):
        with self._log_lock:
            self.log.exception(*a, **kw)

    def start(self, stopwatch_id: str) -> None:
        self._starts.setdefault(stopwatch_id, []).append(time.perf_counter_ns())

    def stop(self, stopwatch_id: str) -> Optional[float]:
        stack = self._starts.get(stopwatch_id)
        if not stack:
            self.warning("⚠️  Tried to stop stopwatch '%s' which was never started.", stopwatch_id)
            return None

        duration_ms = (time.perf_counter_ns() - stack.pop()) / 1e6  # milliseconds
        self.info("⏱️  %-20s %.3f ms", stopwatch_id, duration_ms)
        return duration_ms


    def elapsed(self, stopwatch_id: str) -> Optional[float]:
        accum = getattr(self, "_accum", None)
        if not accum:
            return None
        return accum.get(stopwatch_id)

    def log(self, msg: str, *args, **kwargs) -> None:
        with self._log_lock:
            self.info(msg, *args, **kwargs)

    def track(self, entry_type: str, identifier: str, value: str) -> None:
        with self._lock:
            self._csv_writer.writerow([
                self.run_id,
                time.strftime("%Y-%m-%d %H:%M:%S"),
                entry_type,
                identifier,
                value,
            ])
            self._csv_fp.flush()

    @contextmanager
    def timing(self, stopwatch_id: str):
        self.start(stopwatch_id)
        try:
            yield
        finally:
            self.stop(stopwatch_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        for stopwatch_id, stack in list(self._starts.items()):
            while stack:
                self.stop(stopwatch_id)
        self.info("🏁 Run %s finished.", self.run_id)
        self._csv_fp.close()
        return False

    def set_indexing_context(self, document_id: str | None) -> None:
        """Set the current document id being indexed (None to clear)."""
        self._indexing_doc_id = document_id

    @property
    def indexing_document_id(self) -> str | None:
        """Return the document id currently used for indexing context."""
        return self._indexing_doc_id

    def set_retrieval_context(self, question_id: str | None) -> None:
        """Set the current question id being retrieved (None to clear)."""
        self._retrieval_question_id = question_id

    @property
    def retrieval_question_id(self) -> str | None:
        """Return the question id currently used for retrieval context."""
        return self._retrieval_question_id

    def track_llm_tokens(self, *, context: "LLMCallContext", identifier: str, input_tokens: int | float | None = None, output_tokens: int | float | None = None) -> None:
        """Track LLM input/output tokens for a specific context and identifier."""
        if context.value == "indexing":
            if input_tokens is not None:
                self.track(EntryType.LLM_INDEXING_INPUT_TOKENS_TRACK.value, identifier, str(input_tokens))
            if output_tokens is not None:
                self.track(EntryType.LLM_INDEXING_OUTPUT_TOKENS_TRACK.value, identifier, str(output_tokens))
        elif context.value == "retrieval":
            if input_tokens is not None:
                self.track(EntryType.LLM_RETRIEVAL_INPUT_TOKENS_TRACK.value, identifier, str(input_tokens))
            if output_tokens is not None:
                self.track(EntryType.LLM_RETRIEVAL_OUTPUT_TOKENS_TRACK.value, identifier, str(output_tokens))
        elif context.value == "generation":
            if input_tokens is not None:
                self.track(EntryType.LLM_QA_INPUT_TOKENS_TRACK.value, identifier, str(input_tokens))
            if output_tokens is not None:
                self.track(EntryType.LLM_QA_OUTPUT_TOKENS_TRACK.value, identifier, str(output_tokens))

    def track_llm_cost(self, *, context: "LLMCallContext", identifier: str, cost: float | str) -> None:
        """Track LLM call cost for a specific context and identifier."""
        if context == LLMCallContext.INDEXING:
            self.track(EntryType.LLM_INDEXING_COST_TRACK.value, identifier, str(cost))
        elif context == LLMCallContext.GENERATION:
            self.track(EntryType.LLM_QA_COST_TRACK.value, identifier, str(cost))

    def track_llm_call(self, *, context: "LLMCallContext", identifier: str, duration_ms: float | None = None) -> None:
        """Track an LLM call for a specific context and identifier."""
        if context.value == "indexing":
            if duration_ms is not None:
                self.track(EntryType.LLM_INDEXING_GENERATION_TIME_TRACK.value, identifier, f"{duration_ms:.3f}")
            else:
                self.track(EntryType.LLM_INDEXING_GENERATION_TIME_TRACK.value, identifier, "1")
        elif context.value == "retrieval":
            if duration_ms is not None:
                self.track(EntryType.LLM_RETRIEVAL_GENERATION_TIME_TRACK.value, identifier, f"{duration_ms:.3f}")
            else:
                self.track(EntryType.LLM_RETRIEVAL_GENERATION_TIME_TRACK.value, identifier, "1")
        elif context.value == "generation":
            if duration_ms is not None:
                self.track(EntryType.LLM_QA_GENERATION_TIME_TRACK.value, identifier, f"{duration_ms:.3f}")
            else:
                self.track(EntryType.LLM_QA_GENERATION_TIME_TRACK.value, identifier, "1")

    # Legacy methods for backward compatibility
    def track_indexing_tokens(self, *, input_tokens: int | float | None = None, output_tokens: int | float | None = None) -> None:
        """Track indexing input/output tokens for the current document context."""
        if not self._indexing_doc_id:
            self.warning("⚠️ Attempted to track indexing tokens without a document context.")
            return
        doc_id = self._indexing_doc_id
        if input_tokens is not None:
            self.track(EntryType.LLM_INDEXING_INPUT_TOKENS_TRACK.value, doc_id, str(input_tokens))
        if output_tokens is not None:
            self.track(EntryType.LLM_INDEXING_OUTPUT_TOKENS_TRACK.value, doc_id, str(output_tokens))

    def track_indexing_llm_call(self, *, duration_ms: float | None = None) -> None:
        """Track an indexing LLM call for the current document context."""
        if not self._indexing_doc_id:
            self.warning("⚠️ Attempted to track indexing LLM call without a document context.")
            return
        doc_id = self._indexing_doc_id
        if duration_ms is not None:
            self.track(EntryType.LLM_INDEXING_GENERATION_TIME_TRACK.value, doc_id, f"{duration_ms:.3f}")
        else:
            self.track(EntryType.LLM_INDEXING_GENERATION_TIME_TRACK.value, doc_id, "1")

    def track_retrieval_llm_call(self, *, question_id: str, duration_ms: float | None = None) -> None:
        """Track a retrieval LLM call for a specific question."""
        if duration_ms is not None:
            self.track(EntryType.LLM_RETRIEVAL_GENERATION_TIME_TRACK.value, question_id, f"{duration_ms:.3f}")
        else:
            self.track(EntryType.LLM_RETRIEVAL_GENERATION_TIME_TRACK.value, question_id, "1")

    def track_retrieval_tokens(self, *, question_id: str, input_tokens: int | float | None = None, output_tokens: int | float | None = None) -> None:
        """Track retrieval input/output tokens for a specific question."""
        if input_tokens is not None:
            self.track(EntryType.LLM_RETRIEVAL_INPUT_TOKENS_TRACK.value, question_id, str(input_tokens))
        if output_tokens is not None:
            self.track(EntryType.LLM_RETRIEVAL_OUTPUT_TOKENS_TRACK.value, question_id, str(output_tokens))

    @staticmethod
    def estimate_tokens(text: Any, *, words_per_token: float = 0.75) -> int:
        """Roughly estimate the number of tokens in *text* without invoking a tokenizer."""
        if text is None:
            return 0
        as_str = str(text).strip()
        if not as_str:
            return 0
        words = as_str.split()
        if not words:
            return 0
        if words_per_token <= 0:
            words_per_token = 0.75
        estimated = int(round(len(words) / words_per_token))
        return max(1, estimated)
