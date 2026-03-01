"""
NovelQA → Custom Ontology Converter
==================================

Converts a NovelQA‑formatted corpus into the custom folder layout, and
now **populates each `_qa.json` file** with transformed Q&A data.

<output_root>/
  └── <sanitised_title>/
        ├── <sanitised_title>_metadata.json
        ├── <sanitised_title>_raw.txt
        └── <sanitised_title>_qa.json  (list of question objects)

Only the Python standard library is required.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

################################################################################
# Utility helpers
################################################################################
TITLE_SAFE_RE = re.compile(r"[^a-z0-9_\-]+")  # characters we will strip


def sanitise_title(title: str) -> str:
    """Return a filesystem‑safe, lower‑case version of *title*."""
    title = title.lower().replace(" ", "_")
    return TITLE_SAFE_RE.sub("", title)


def ensure_dir(path: Path) -> None:
    """Create *path* and all parents; ignore if it already exists."""
    path.mkdir(parents=True, exist_ok=True)


def write_json(dest: Path, data: Any, *, overwrite: bool = False) -> None:
    if dest.exists() and not overwrite:
        print(f"    • {dest.name} exists → skip (use overwrite=True to replace)")
        return
    with dest.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


################################################################################
# File‑copy helpers
################################################################################

def copy_raw_text(book_id: str, meta: Dict[str, Any], src_root: Path, dest_path: Path) -> None:
    """Copy the raw ``*.txt`` for *book_id* into *dest_path*."""
    copyright_folder = meta.get("copyright", "PublicDomain")
    src_txt = src_root / "Books" / copyright_folder / f"{book_id}.txt"
    if not src_txt.exists():
        print(f"[WARN] Raw text not found for {book_id}")
        return
    shutil.copy2(src_txt, dest_path)


################################################################################
# QA conversion
################################################################################

def _locate_qa_file(book_id: str, copyright_folder: str, src_root: Path) -> Optional[Path]:
    """Return the path to the NovelQA question file (``*.json`` or misspelled
    ``*.sjon``). Returns ``None`` if not found."""
    data_dir = src_root / "Data" / copyright_folder
    for ext in (".json", ".sjon"):
        p = data_dir / f"{book_id}{ext}"
        if p.exists():
            return p
    return None


def _transform_q(qid: str, qdata: Dict[str, Any], *, document_id: str) -> Dict[str, Any]:
    """Transform one NovelQA question to the target schema."""
    choices = [
        {"label": label, "text": text}
        for label, text in qdata.get("Options", {}).items()
    ]

    proofs = [
        {
            "document_id": document_id,
            "context": edata.get("Evidence") or edata.get("Sent") or ""
        }
        for edata in qdata.get("Evidences", {}).values()
    ]

    return {
        "question_id": qid,
        "question": qdata.get("Question", ""),
        "choices": choices,
        "correct_answer": qdata.get("Gold"),
        "proofs": proofs,
    }


def fill_qa(
    book_id: str,
    title_safe: str,
    meta: Dict[str, Any],
    src_root: Path,
    dest_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Populate *dest_path* with transformed Q&A for *book_id*.

    If no QA source exists, an **empty list** is written so downstream
    pipelines have a valid JSON file to read.
    """
    copyright_folder = meta.get("copyright", "PublicDomain")
    qa_src = _locate_qa_file(book_id, copyright_folder, src_root)

    if qa_src is None:
        print(f"    • QA file missing → {book_id} (created empty)")
        write_json(dest_path, [], overwrite=True)
        return

    with qa_src.open("r", encoding="utf-8") as fp:
        try:
            qa_raw: Dict[str, Any] = json.load(fp)
        except json.JSONDecodeError as exc:
            print(f"[WARN] Malformed QA JSON for {book_id}: {exc} → skipping")
            write_json(dest_path, [], overwrite=True)
            return

    transformed: List[Dict[str, Any]] = [
        _transform_q(qid, qdata, document_id=title_safe)
        for qid, qdata in qa_raw.items()
    ]

    write_json(dest_path, transformed, overwrite=True)


################################################################################
# Top‑level pipeline
################################################################################

def load_bookmeta(meta_path: Path) -> Dict[str, Dict[str, Any]]:
    try:
        with meta_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except FileNotFoundError:
        sys.exit(f"[ERR] bookmeta.json not found at: {meta_path!s}")
    except json.JSONDecodeError as exc:
        sys.exit(f"[ERR] Malformed JSON in bookmeta.json: {exc}")


def convert_corpus(src_root: Path, out_root: Path, *, overwrite: bool = False) -> None:
    meta_all = load_bookmeta(src_root / "bookmeta.json")

    for book_id, info in meta_all.items():
        title_safe = sanitise_title(info.get("title", book_id)) or book_id.lower()
        book_dir = out_root / title_safe
        ensure_dir(book_dir)

        metadata_path = book_dir / f"{title_safe}_metadata.json"
        raw_path = book_dir / f"{title_safe}_raw.txt"
        qa_path = book_dir / f"{title_safe}_qa.json"

        print(f"→ {book_dir.relative_to(out_root)}")

        # 1. metadata
        write_json(metadata_path, info, overwrite=overwrite)

        # 2. raw text
        if raw_path.exists() and not overwrite:
            print("    • raw exists → skip (use overwrite=True)")
        else:
            copy_raw_text(book_id, info, src_root, raw_path)

        # 3. QA data (always overwrite to ensure fresh transform)
        fill_qa(book_id, title_safe, info, src_root, qa_path, overwrite=True)

    print("\n✓ Conversion complete.")


################################################################################
# Script entry point
################################################################################

def main() -> None:
    # Paths relative to project root for development/testing
    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "util/data 🗃️/NovelQA"
    output_path = project_root / "data 🗃️/NovelQA"
    overwrite = False  # Set to True if you want to overwrite existing files

    if not input_path.is_dir():
        sys.exit(f"Input directory does not exist: {input_path}")

    ensure_dir(output_path)
    convert_corpus(input_path, output_path, overwrite=overwrite)


if __name__ == "__main__":  # pragma: no cover
    main()
