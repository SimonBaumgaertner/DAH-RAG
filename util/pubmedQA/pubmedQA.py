#!/usr/bin/env python3
"""
Transform PubMedQA -> your ontology.

Creates:
  data 🗃️/{pmid}/{pmid}_metadata.json
  data 🗃️/{pmid}/{pmid}_qa.json
  data 🗃️/{pmid}/{pmid}_raw.txt

Assumes input JSON shaped like ori_pqal.json (keys are PMIDs).
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import time
import xml.etree.ElementTree as ET
import requests

# ---- CONFIG ----
INPUT_JSON = "pubmedQA_questions.json"   # path to your PubMedQA json (e.g., ori_pqal.json)
OUTPUT_DIR = "data"                      # output folder created at the same directory level
BATCH_SIZE = 150                         # EFetch can handle large batches; keep it modest
REQUEST_TIMEOUT = 30
PAUSE_BETWEEN_CALLS = 0.4                # ~3 req/s guideline without API key

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

CHOICE_ORDER = [("A", "Yes"), ("B", "No"), ("C", "Maybe")]
ANSWER_TO_LABEL = {"yes": "A", "no": "B", "maybe": "C"}

# ----------------

def load_pubmedqa(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # If file is a list of items with 'pubid', normalize into {pmid: item}
    if isinstance(data, list):
        out = {}
        for it in data:
            pmid = str(it.get("pubid") or it.get("PMID") or it.get("pmid"))
            if not pmid:
                continue
            out[pmid] = it
        return out
    return {str(k): v for k, v in data.items()}

def efetch_pubmed_batch(pmids: List[str]) -> Dict[str, dict]:
    """
    Fetch title, authors, year, abstract for many PMIDs at once.
    Returns dict[pmid] = {"title":..., "authors":..., "year":..., "abstract":...}
    """
    if not pmids:
        return {}
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    r = requests.get(EUTILS_BASE, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()

    root = ET.fromstring(r.text)
    out: Dict[str, dict] = {}
    for art in root.findall(".//PubmedArticle"):
        pmid = (art.findtext(".//PMID") or "").strip()
        if not pmid:
            continue
        title = (art.findtext(".//ArticleTitle") or "").strip()

        # year: try ArticleDate/Journal Issue PubDate
        year = None
        y1 = art.findtext(".//Article/Journal/JournalIssue/PubDate/Year")
        y2 = art.findtext(".//DateCreated/Year")
        for y in (y1, y2):
            if y and y.isdigit():
                year = int(y)
                break

        # authors
        authors_nodes = art.findall(".//AuthorList/Author")
        authors = []
        for a in authors_nodes:
            last = (a.findtext("LastName") or "").strip()
            fore = (a.findtext("ForeName") or "").strip()
            collab = (a.findtext("CollectiveName") or "").strip()
            if collab:
                authors.append(collab)
            elif last or fore:
                full = f"{last}, {fore}" if fore else last
                authors.append(full)
        author_str = None
        if authors:
            author_str = authors[0] if len(authors) == 1 else f"{authors[0]} et al."

        # abstract (concatenate structured parts without labels to look like website)
        parts = []
        for node in art.findall(".//Abstract/AbstractText"):
            text = "".join(node.itertext()).strip()
            if text:
                parts.append(text)
        abstract = " ".join(parts).strip() if parts else ""

        out[pmid] = {
            "title": title,
            "authors": author_str,
            "year": year,
            "abstract": abstract,
        }
    return out

def reconstruct_abstract_from_item(item: dict) -> str:
    contexts = item.get("CONTEXTS") or []
    long_answer = item.get("LONG_ANSWER") or ""
    segments = [c.strip() for c in contexts if c and c.strip()]
    if long_answer and long_answer.strip():
        segments.append(long_answer.strip())
    return " ".join(segments)

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def token_len(text: str) -> int:
    return len(text.split())

def make_metadata(pmid: str, fetched: dict, fallback_year: str, abstract_text: str) -> dict:
    # Map into your ontology’s fields
    yearpub = fetched.get("year")
    if yearpub is None:
        # try the dataset YEAR if numeric
        try:
            yearpub = int(fallback_year)
        except Exception:
            yearpub = None

    author_str = fetched.get("authors")
    if not author_str:
        author_str = None

    return {
        "title": fetched.get("title") or f"PMID {pmid}",
        "source": "PubMed",
        "link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "whichgtb": None,
        "copyright": "AbstractOnly",
        "yearpub": yearpub,
        "author": author_str,
        "yearperish": None,
        "period": "Biomedical",
        "tokenlen": token_len(abstract_text),
    }

def make_qa_item(pmid: str, item: dict) -> dict:
    qtext = item.get("QUESTION") or item.get("question") or ""
    final = (item.get("final_decision") or "").lower().strip()
    correct_letter = ANSWER_TO_LABEL.get(final)  # could be None for unlabeled

    choices = [{"label": lab, "text": txt} for lab, txt in CHOICE_ORDER]

    # proofs: use CONTEXTS and LONG_ANSWER if present
    proofs = []
    for ctx in (item.get("CONTEXTS") or []):
        proofs.append({"document_id": pmid, "context": ctx})
    if item.get("LONG_ANSWER"):
        proofs.append({"document_id": pmid, "context": item["LONG_ANSWER"]})

    return {
        "question_id": f"PMID{pmid}",
        "question": qtext,
        "choices": choices,
        "correct_answer": correct_letter,
        "proofs": proofs,
    }

def write_doc(pmid: str, meta: dict, qa: List[dict], raw_abs: str, out_root: Path) -> None:
    doc_dir = out_root / pmid
    ensure_dir(doc_dir)

    with open(doc_dir / f"{pmid}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(doc_dir / f"{pmid}_qa.json", "w", encoding="utf-8") as f:
        json.dump(qa, f, ensure_ascii=False, indent=2)

    with open(doc_dir / f"{pmid}_raw.txt", "w", encoding="utf-8") as f:
        f.write(raw_abs)

def batched(iterable, n):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf

def main():
    src = load_pubmedqa(INPUT_JSON)
    out_root = Path(OUTPUT_DIR)
    ensure_dir(out_root)

    pmids = list(src.keys())

    # 1) Fetch metadata in batches
    fetched_all: Dict[str, dict] = {}
    for batch in batched(pmids, BATCH_SIZE):
        try:
            fetched = efetch_pubmed_batch(batch)
            fetched_all.update(fetched)
        except Exception as e:
            print(f"[WARN] EFetch failed for batch starting {batch[0]}: {e}")
        time.sleep(PAUSE_BETWEEN_CALLS)

    # 2) Build and write files per PMID
    for pmid in pmids:
        item = src[pmid]

        fetched = fetched_all.get(pmid, {})
        # choose abstract: fetched if available, else reconstruct from dataset
        abs_text = fetched.get("abstract") or reconstruct_abstract_from_item(item)

        meta = make_metadata(
            pmid=pmid,
            fetched=fetched,
            fallback_year=item.get("YEAR"),
            abstract_text=abs_text,
        )
        qa_item = make_qa_item(pmid, item)

        write_doc(pmid, meta, [qa_item], abs_text, out_root)

    print(f"Done. Wrote {len(pmids)} documents under {out_root.resolve()}")

if __name__ == "__main__":
    main()
