#!/usr/bin/env python3
"""
HotpotQA → custom corpus folderizer with budgeted full-coverage selection.

Pipeline:
1) Parse HotpotQA and compute, for each question, the set of required article slugs.
2) Check which slugs exist in ARTICLES_DIR; drop questions requiring any missing slug.
3) Greedily pick up to ARTICLE_BUDGET articles to maximize the number of fully covered questions.
4) Copy ONLY the selected articles into output/<Slug>/, write metadata, clean <a> links.
5) Emit OUTPUT_DIR/QA.json with ONLY fully covered questions.
"""

import json
import os
import re
import shutil
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import unquote

# ---------- CONFIG CONSTANTS ----------
HOTPOT_JSON     = Path("hotpot_dev_fullwiki_v1.json")       # path to HotpotQA JSON
ARTICLES_DIR    = Path("~/Wiki2017/articles").expanduser()  # where .txt articles live
OUTPUT_DIR      = Path("../../data/HotpotQA_100")                            # output root
ARTICLE_BUDGET  = 100                                        # how many articles to export
# --------------------------------------

# ---------- filename sanitation (IDENTICAL to your splitter) ----------
BAD_CHARS = re.compile(r'[\\/:#<>\[\]\{\}\|\x00-\x1F\x7F]')
SPACE_RE  = re.compile(r'\s+')

def sanitize_title(title: str) -> str:
    t = unicodedata.normalize("NFC", (title or "").strip().replace("\u00A0", " "))
    t = BAD_CHARS.sub("_", t)
    t = SPACE_RE.sub(" ", t).replace(" ", "_")
    t = re.sub(r'_+', '_', t).strip('_')
    return (t.encode("utf-8")[:200].decode("utf-8", "ignore")) or "UNTITLED"

# ---------- I/O ----------
def load_hotpot(hotpot_path: Path) -> List[dict]:
    with open(hotpot_path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_article_file(articles_dir: Path, base_slug: str) -> Optional[Path]:
    exact = articles_dir / f"{base_slug}.txt"
    if exact.exists():
        return exact
    hashed = sorted(articles_dir.glob(f"{base_slug}__*.txt"))
    if hashed:
        return hashed[0]
    return None

def write_out(slug: str, article_file: Path, out_root: Path) -> Path:
    """Copy source article to <out>/<slug>/<slug>_raw.txt and write metadata.
    Returns path to the _raw file."""
    out_dir = out_root / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_dst = out_dir / f"{slug}_raw.txt"
    meta_dst = out_dir / f"{slug}_metadata.json"

    shutil.copyfile(article_file, raw_dst)

    meta = {
        "title": slug,
        "source": "wiki",
        "yearpub": "2017",
        "author": "wikipedia",
    }
    with open(meta_dst, "w", encoding="utf-8") as w:
        json.dump(meta, w, ensure_ascii=False, indent=2)

    return raw_dst

def write_qa_json(entries: List[dict], out_root: Path) -> Path:
    path = out_root / "QA.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    return path

# ---------- link cleaning ----------
A_TAG_RE = re.compile(
    r'<a\b[^>]*?href=(["\'])(.*?)\1[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)

def href_to_slug(href: str) -> Optional[str]:
    if not href:
        return None
    part = href.split("/")[-1]
    part = part.split("#", 1)[0].split("?", 1)[0]
    part = unquote(part)
    if not part.strip():
        return None
    return sanitize_title(part)

def replace_a_with_ref(match: re.Match) -> str:
    href = match.group(2)
    text = match.group(3)
    slug = href_to_slug(href)
    if not slug:
        return text
    return f"{text}\\ref{{{slug}}}"

def clean_links_inplace(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if "<a" not in content.lower():
        return 0
    n_before = len(list(A_TAG_RE.finditer(content)))
    content = A_TAG_RE.sub(replace_a_with_ref, content)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return n_before

# ---------- modeling the questions ----------
class QExample:
    __slots__ = ("qid", "question", "answer", "supp_pairs", "req_slugs", "ctx_map", "raw_ex")
    def __init__(self, ex: dict):
        self.qid = str(ex.get("_id") or ex.get("id") or "")
        self.question = ex.get("question", "")
        self.answer = ex.get("answer", "")
        self.supp_pairs: List[Tuple[str, int]] = []
        for t, idx in ex.get("supporting_facts", []) or []:
            if isinstance(t, str) and isinstance(idx, int):
                self.supp_pairs.append((t, idx))
        self.req_slugs: Set[str] = {sanitize_title(t) for t, _ in self.supp_pairs}
        self.ctx_map: Dict[str, List[str]] = {}
        for pair in ex.get("context", []) or []:
            if isinstance(pair, list) and len(pair) == 2 and isinstance(pair[0], str):
                self.ctx_map[pair[0]] = pair[1] if isinstance(pair[1], list) else []
        self.raw_ex = ex

def build_examples(data: List[dict]) -> List[QExample]:
    return [QExample(ex) for ex in data if ex.get("supporting_facts")]

def map_slug_to_path(articles_dir: Path, slugs: Set[str]) -> Dict[str, Path]:
    """Return mapping for available slugs -> file path (existing only)."""
    out: Dict[str, Path] = {}
    for s in slugs:
        p = find_article_file(articles_dir, s)
        if p is not None:
            out[s] = p
    return out

# ---------- greedy selector (full-coverage, simple sequential) ----------
def greedy_select_articles(questions: List[QExample], available_slugs: Set[str], budget: int) -> Set[str]:
    """
    Process questions in dataset order. For each question, if adding all of its
    required slugs fits within the remaining budget, take them; otherwise skip.
    Returns the set of selected slugs.
    """
    selected: Set[str] = set()
    if budget <= 0:
        return selected

    for q in questions:
        # Skip if any required slug is missing on disk
        if not q.req_slugs or not q.req_slugs.issubset(available_slugs):
            continue

        needed = q.req_slugs - selected
        if len(selected) + len(needed) <= budget:
            selected |= needed
            if len(selected) == budget:
                break  # budget reached

    return selected


# ---------- QA builder (full coverage only) ----------
def build_qa_from_covered(covered: List[QExample]) -> List[dict]:
    qa: List[dict] = []
    for q in covered:
        # Build proofs in the same order as supporting_facts; dedup (title, idx)
        seen: Set[Tuple[str, int]] = set()
        proofs: List[dict] = []
        for t, idx in q.supp_pairs:
            if (t, idx) in seen:  # dedup
                continue
            seen.add((t, idx))
            sentences = q.ctx_map.get(t, [])
            ctx = sentences[idx] if 0 <= idx < len(sentences) else ""
            parsed_ctx = re.sub(r'\\"', '"', (ctx or t))
            proofs.append({
                "document_id": sanitize_title(t),
                # normalize escaped quotes like \"Red\" → "Red"
                "context": parsed_ctx,
            })

        qa.append({
            "question_id": q.qid,
            "question": q.question,
            "choices": [],            # Hotpot has no MC choices
            "correct_answer": q.answer,
            "proofs": proofs,
        })
    return qa

# ---------- main ----------
def main() -> None:
    if not ARTICLES_DIR.exists():
        print(f"❌ Articles directory not found: {ARTICLES_DIR}", file=sys.stderr)
        sys.exit(1)
    if not HOTPOT_JSON.exists():
        print(f"❌ HotpotQA JSON not found: {HOTPOT_JSON}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"📥 Hotpot: {HOTPOT_JSON}")
    print(f"📚 Articles dir: {ARTICLES_DIR}")
    print(f"📤 Output dir: {OUTPUT_DIR}")
    print(f"🎯 Article budget: {ARTICLE_BUDGET}")

    data = load_hotpot(HOTPOT_JSON)
    examples = build_examples(data)

    # Universe of slugs required by all questions
    all_slugs: Set[str] = set()
    for q in examples:
        all_slugs |= q.req_slugs

    # Which slugs do we actually have on disk?
    slug_to_path = map_slug_to_path(ARTICLES_DIR, all_slugs)
    available_slugs = set(slug_to_path.keys())

    # Greedy select up to ARTICLE_BUDGET slugs to maximize fully coverable questions
    selected_slugs = greedy_select_articles(examples, available_slugs, ARTICLE_BUDGET)
    print(f"🧩 Selected {len(selected_slugs)} / {ARTICLE_BUDGET} articles.")

    # Determine covered questions (FULL coverage)
    covered_questions: List[QExample] = [
        q for q in examples if q.req_slugs and q.req_slugs.issubset(selected_slugs)
    ]
    print(f"✅ Fully covered questions: {len(covered_questions)}")

    # Copy ONLY selected articles, clean links
    written_raw_paths: List[Path] = []
    for slug in sorted(selected_slugs):
        p = slug_to_path.get(slug)
        if not p:
            continue  # shouldn't happen (filtered already)
        raw_path = write_out(slug, p, OUTPUT_DIR)
        written_raw_paths.append(raw_path)

    print("🧼 Cleaning links in the _raw files…")
    total_repls = 0
    for p in written_raw_paths:
        n = clean_links_inplace(p)
        total_repls += n
        print(f"  - {p.name}: {n} links cleaned")
    print(f"✨ Link conversion complete: {total_repls} <a> tags across {len(written_raw_paths)} files.")

    # Write QA.json for fully covered questions only
    qa_entries = build_qa_from_covered(covered_questions)
    qa_path = write_qa_json(qa_entries, OUTPUT_DIR)
    print(f"📄 Wrote {len(qa_entries)} questions to {qa_path}")

if __name__ == "__main__":
    main()
