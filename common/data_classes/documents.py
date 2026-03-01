from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

from common.data_classes.qa import QuestionAnswerPair

@dataclass
class Document:
    id: str
    title: str
    author: str
    publication_date: Optional[date]  # <-- can be None
    references: List[str]
    text: str = None # Optional
    qa_pairs: List[QuestionAnswerPair] = None # Optional
    centroids: List[List[float]] = field(default_factory=list)

    @classmethod
    def from_folder(cls, folder: str | Path) -> Document:
        folder = Path(folder)
        doc_id = folder.name

        # --- Load metadata ---
        meta_path = folder / f"{doc_id}_metadata.json"
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)

        # --- Raw text + references ---
        raw_path = folder / f"{doc_id}_raw.txt"
        raw_text = raw_path.read_text(encoding="utf-8")
        text, references = process_raw_and_extract_references(raw_text)

        # --- QA pairs ---
        qa_path = folder / f"{doc_id}_qa.json"
        qa_pairs: List[QuestionAnswerPair] = []
        if qa_path.exists():
            try:
                with qa_path.open(encoding="utf-8") as f:
                    qa_list = json.load(f) or []
                qa_pairs = [QuestionAnswerPair.from_dict(q) for q in qa_list]
            except json.JSONDecodeError:
                qa_pairs = []

        # --- Publication date (nullable) ---
        pub_date_raw = meta.get("pub_date")
        pub_date: Optional[date] = None
        if pub_date_raw:
            try:
                pub_date = date.fromisoformat(pub_date_raw)
            except ValueError:
                # not in ISO format, ignore and leave as None
                pub_date = None

        return cls(
            id=doc_id,
            title=meta.get("title"),
            author=meta.get("author"),
            publication_date=pub_date,
            text=text,
            references=references,
            qa_pairs=qa_pairs,
        )

def process_raw_and_extract_references(raw_text) -> Tuple[str, List[str]]:
    """
    Remove all occurrences of `ref{...}` from raw_text and collect the contents.
    Returns a tuple: (processed_text_without_refs, list_of_contents).
    """
    references: List[str] = []

    def _collect_and_strip(match: re.Match) -> str:
        references.append(match.group(1))
        return ""  # remove the entire ref{...} from the text

    # Non-greedy to stop at the first closing brace; DOTALL allows newlines inside {...}
    processed_text = re.sub(r"ref\{(.*?)\}", _collect_and_strip, raw_text, flags=re.DOTALL)

    return processed_text, references