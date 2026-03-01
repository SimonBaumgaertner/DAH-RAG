#!/usr/bin/env python3
"""
HotpotQA utilities (no argparse), with anchor → ref{...} conversion.

- 'stats'          Count how often Wikipedia titles are referenced in a HotpotQA JSON.
- 'extract'        Build a k-article subset + fetch matching wiki pages from a dump.
- 'clean-articles' Clean HTML/noise in extracted article folders.
- 'clean-json'     Clean HTML/noise in proof contexts inside a subset JSON.
"""

from __future__ import annotations

import bz2
import html
import json
import re
import tarfile
import urllib.parse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple


# ------------------------------ Paths (yours) --------------------------------

# For cleaning articles / subset JSON (your later pipeline)
ROOT_ARTICLES = Path(__file__).resolve().parent / "data/HotpotQA"
JSON_IN = ROOT_ARTICLES / "hotpot_subset_k1000.json"

# For stats/extract (your original locations)
FULL_DATA_JSON = ROOT_ARTICLES / "hotpot_dev_fullwiki_v1.json"
WIKI_DUMP = ROOT_ARTICLES / "enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2"
SUBSET_OUT = ROOT_ARTICLES / "hotpot_subset_k1000.json"
DOCS_EXTRACT_DIR = ROOT_ARTICLES  # where extract writes article folders

# ------------------------------ Stats ---------------------------------------

def count_supporting_titles(path: Path) -> Tuple[int, Dict[str, int]]:
    """Count how many questions reference each Wikipedia title in a HotpotQA JSON."""
    title_counts: Dict[str, int] = defaultdict(int)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        total_questions = len(data)

        for ex in data:
            titles = {title for title, _ in ex.get("supporting_facts", [])}
            for title in titles:
                title_counts[title] += 1

    print(f"📊 Total questions: {total_questions}")
    print(f"📚 Total unique articles referenced: {len(title_counts)}")
    print("🏆 Top 20 most-used articles:")
    for title, count in sorted(title_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"{title:50} → {count} questions")

    return total_questions, dict(title_counts)


# ------------------------- Wiki dump iteration ------------------------------

def iter_wiki_json_lines(dump_path: Path, wanted_titles: Set[str]) -> Iterator[Tuple[str, dict]]:
    """
    Iterate wiki JSON lines either from a .bz2 stream OR a .tar.bz2 containing .bz2 shards.
    Yields (title, obj) only for titles in `wanted_titles`.
    """
    dump_path = Path(dump_path)

    # Case 1: tarball of bz2 shards
    if dump_path.suffixes[-2:] in [['.tar', '.bz2'], ['.tar', '.bz']]:
        with tarfile.open(dump_path, mode="r:bz2") as tf:
            for m in tf:
                if not (m.isfile() and m.name.endswith(".bz2")):
                    continue
                with bz2.open(tf.extractfile(m), "rb") as stream:  # type: ignore[arg-type]
                    for raw in stream:
                        try:
                            obj = json.loads(raw.decode("utf-8", errors="replace"))
                        except json.JSONDecodeError:
                            continue
                        if isinstance(obj, dict):
                            title = obj.get("title", "").strip()
                            if title in wanted_titles:
                                yield title, obj
        return

    # Case 2: single file (bz2 or plain)
    opener = bz2.open if dump_path.suffix in (".bz2", ".bz") else open
    with opener(dump_path, "rb") as f:  # type: ignore[call-arg]
        for raw in f:
            try:
                obj = json.loads(raw.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                title = obj.get("title", "").strip()
                if title in wanted_titles:
                    yield title, obj


# ---------------------------- Text cleanup ----------------------------------

# Anchor patterns
ANCHOR_FULL_RE = re.compile(
    r'<a\s[^>]*?href=(["\'])(.*?)\1[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
ANCHOR_OPEN_RE = re.compile(
    r'<a\s[^>]*?href=(["\'])(.*?)\1[^>]*>',
    re.IGNORECASE | re.DOTALL,
)

# Generic tag/whitespace
TAG_RE = re.compile(r"<[^>]+>")
WS_RE  = re.compile(r"\s+")

# ref{...} tokens
REF_TOKEN_RE = re.compile(r'\s*ref\{[^}]*\}')


def _href_to_title(href: str) -> str:
    """Best-effort convert an href to a human title."""
    href = html.unescape(href)
    href = urllib.parse.unquote(href)
    try:
        parts = urllib.parse.urlsplit(href)
        path = parts.path or href
    except Exception:
        path = href
    seg = path.rsplit("/", 1)[-1] if "/" in path else path
    seg = seg.split("#", 1)[0]  # strip fragment
    title = seg.replace("_", " ")
    return WS_RE.sub(" ", title).strip()


def clean_html(text: str, *, keep_refs: bool = True) -> str:
    """
    Clean HTML & anchors.

    - If keep_refs=True: convert anchors to "inner_text ref{Title}" (or "ref{Title}" if no inner text).
    - If keep_refs=False: convert anchors to plain visible text; drop any ref{...} tokens entirely.
    - Always: strip tags, decode entities and %-escapes, normalize whitespace.

    This lets us KEEP refs in article files but REMOVE them in JSON proof contexts.
    """
    def _replace_full_anchor(m: re.Match) -> str:
        href  = m.group(2)
        inner = m.group(3)
        title = _href_to_title(href)
        inner_clean = TAG_RE.sub(" ", html.unescape(inner))
        inner_clean = WS_RE.sub(" ", inner_clean).strip()
        if keep_refs:
            return f"{inner_clean} ref{{{title}}}" if inner_clean else f"ref{{{title}}}"
        else:
            # Prefer visible text; fall back to title if anchor had no text
            return inner_clean if inner_clean else title

    def _replace_open_anchor(m: re.Match) -> str:
        title = _href_to_title(m.group(2))
        return f"ref{{{title}}}" if keep_refs else title

    # 1) Convert anchors
    text = ANCHOR_FULL_RE.sub(_replace_full_anchor, text)
    text = ANCHOR_OPEN_RE.sub(_replace_open_anchor, text)

    # 2) Drop remaining tags
    text = TAG_RE.sub(" ", text)

    # 3) Decode entities and %-escapes
    text = html.unescape(text)
    text = urllib.parse.unquote(text)

    # 4) Optionally remove any lingering ref{...} tokens (e.g., prior passes)
    if not keep_refs:
        text = REF_TOKEN_RE.sub("", text)

    # 5) Normalize whitespace and spacing before punctuation
    text = WS_RE.sub(" ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text


def drop_first_duplicated_title(text: str, title: str) -> str:
    """If the text starts with 'Title Title ...', remove the first 'Title '."""
    if not title:
        return text
    title_esc = re.escape(title.strip())
    text = text.lstrip("\ufeff\u200e\u200f \t")  # strip BOM/LRM/space
    return re.sub(rf"^(?:{title_esc})\s+(?={title_esc}\b)", "", text, count=1)


# -------------------- Extract subset and articles ---------------------------

def extract_questions_with_k_articles_and_docs(
    json_path: Path,
    wiki_dump_path: Path,
    k: int = 1000,
    out_path: Path | None = None,
    doc_dir: Path | None = None,
) -> None:
    """
    - Load HotpotQA JSON.
    - Select questions referencing up to K unique articles.
    - Load those articles from the wiki dump.
    - Write article raw text + minimal metadata.
    - Emit a simplified subset JSON with proofs replaced by sentence strings.
    """
    if out_path is None:
        out_path = Path("subset_k_articles.json")
    if doc_dir is None:
        doc_dir = Path("articles")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    selected_titles: Set[str] = set()
    used_questions: List[dict] = []

    def question_titles(q: dict) -> Set[str]:
        return {t for t, _ in q.get("supporting_facts", [])}

    for ex in data:
        q_titles = question_titles(ex)
        new_titles = q_titles - selected_titles
        if len(selected_titles) < k or not new_titles:
            selected_titles.update(q_titles)
            used_questions.append(ex)
        if len(selected_titles) >= k:
            break

    print(f"✅ Selected {len(used_questions)} questions covering {len(selected_titles)} unique articles.")

    wiki_articles: Dict[str, dict] = {}
    for title, obj in iter_wiki_json_lines(Path(wiki_dump_path), selected_titles):
        wiki_articles[title] = obj
        if len(wiki_articles) % 100 == 0:
            print(f"   … collected {len(wiki_articles)}/{len(selected_titles)}")
        if len(wiki_articles) >= len(selected_titles):
            break

    print(f"📚 Loaded {len(wiki_articles)} / {len(selected_titles)} requested articles.")

    doc_dir_path = Path(doc_dir)
    doc_dir_path.mkdir(parents=True, exist_ok=True)

    sentence_lookup: Dict[str, List[str]] = {}
    for title, obj in wiki_articles.items():
        sentences = sum(obj.get("text", []), [])
        sentence_lookup[title] = sentences

        safe_title = title.replace("/", "_").strip()
        article_path = doc_dir_path / safe_title
        article_path.mkdir(exist_ok=True)

        raw_file = article_path / f"{safe_title}_raw.txt"
        meta_file = article_path / f"{safe_title}_metadata.json"

        if not raw_file.exists():  # resume-safe
            raw_file.write_text(" ".join(sentences), encoding="utf-8")
            meta_file.write_text(json.dumps({"title": title}, indent=2), encoding="utf-8")

    mc_questions: List[dict] = []
    for ex in used_questions:
        qid = ex["_id"]
        qtext = ex["question"]
        answer = ex["answer"].strip()

        proofs: List[dict] = []
        for title, sent_id in ex.get("supporting_facts", []):
            title = title.strip()
            sentences = sentence_lookup.get(title, [])
            sentence = sentences[sent_id] if 0 <= sent_id < len(sentences) else None
            if sentence:
                proofs.append({"document_id": title, "context": sentence.strip()})

        if answer.lower() in ("yes", "no"):
            qa_obj = {
                "question_id": qid,
                "question": qtext,
                "choices": [{"label": "A", "text": "Yes"}, {"label": "B", "text": "No"}],
                "correct_answer": "A" if answer.lower() == "yes" else "B",
                "proofs": proofs,
            }
        else:
            qa_obj = {
                "question_id": qid,
                "question": qtext,
                "correct_span": answer,
                "proofs": proofs,
            }

        mc_questions.append(qa_obj)

    with open(out_path, "w", encoding="utf-8") as out:
        json.dump(mc_questions, out, indent=2, ensure_ascii=False)

    print(f"📁 Wrote output to: {out_path}")
    print(f"📂 Article folders written to: {doc_dir_path}")


# ------------------ Article & JSON cleaning pipelines -----------------------

def process_article_folder(folder: Path, root_articles: Path | None = None) -> None:
    """
    For a single article folder:
    - read *_raw.txt & *_metadata.json (if present) for the original title,
    - clean HTML/noise (KEEP ref{...}),
    - drop duplicated leading page title,
    - write cleaned text back to *_raw.txt (keep original as *_unprocessed.txt).
    """
    raw_files = list(folder.glob("*_raw.txt"))
    if not raw_files:
        return
    raw_file = raw_files[0]
    safe_stem = raw_file.stem.replace("_raw", "")

    meta_file = raw_file.with_name(f"{safe_stem}_metadata.json")
    page_title = None
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            page_title = (meta.get("title") or "").strip()
        except Exception:
            page_title = None
            print(f"Did not find title for {safe_stem}")
    if not page_title:
        page_title = safe_stem.replace("_", " ").strip()

    unprocessed = raw_file.with_name(f"{safe_stem}_unprocessed.txt")
    if not unprocessed.exists():
        raw_file.rename(unprocessed)

    text = unprocessed.read_text(encoding="utf-8", errors="replace")
    cleaned = clean_html(text, keep_refs=True)  # keep refs in article files
    cleaned = drop_first_duplicated_title(cleaned, page_title)

    raw_file.write_text(cleaned, encoding="utf-8")
    try:
        rel = raw_file.relative_to(root_articles) if root_articles else raw_file
    except Exception:
        rel = raw_file
    print(f"✔ cleaned {rel}")


def clean_articles(articles_root: Path) -> None:
    """Clean every article folder in `articles_root` (keeps ref{...})."""
    for article_dir in Path(articles_root).iterdir():
        if article_dir.is_dir():
            try:
                process_article_folder(article_dir, root_articles=Path(articles_root))
            except Exception as e:
                print(f"⚠️  {article_dir.name}: {e}")


def clean_subset_json(json_path: Path) -> None:
    """
    Clean proof contexts in a subset JSON:
      - strip HTML,
      - convert anchors to visible text,
      - REMOVE any ref{...} tokens,
      - normalize spacing.
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    changed = 0
    for qa in data:
        for prf in qa.get("proofs", []):
            orig = prf["context"]
            cleaned = clean_html(orig, keep_refs=False)  # drop ref{...}
            if cleaned != orig:
                prf["context"] = cleaned
                changed += 1

    out_path = Path(json_path).with_name(Path(json_path).stem + "_clean.json")
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✔ cleaned {changed} proof contexts → {out_path.name}")


# -------------------------------- main --------------------------------------

def main() -> None:
    """
    Chained run using your original paths:
      1) Stats on full HotpotQA JSON
      2) Extract K=1000 subset + articles (to your util/data/HotpotQA/)
      3) Clean articles in ROOT_ARTICLES (keeps ref{...})
      4) Clean proof contexts in JSON_IN (removes ref{...})
    """
    # 1) Stats
    print("▶️  Stats")
    count_supporting_titles(FULL_DATA_JSON)

    # 2) Extract
    K = 1000
    print(f"▶️  Extract  K={K}  docs={DOCS_EXTRACT_DIR}  out={SUBSET_OUT}")
    if True:
        extract_questions_with_k_articles_and_docs(
            json_path=FULL_DATA_JSON,
            wiki_dump_path=WIKI_DUMP,
            k=K,
            out_path=SUBSET_OUT,
            doc_dir=DOCS_EXTRACT_DIR,
        )

    # 3) Clean Articles (your separate articles root; keeps ref{...})
    print(f"▶️  Clean Articles  {ROOT_ARTICLES}")
    clean_articles(ROOT_ARTICLES)

    # 4) Clean Subset JSON (remove ref{...} from contexts)
    print(f"▶️  Clean Subset JSON  {JSON_IN}")
    clean_subset_json(JSON_IN)


if __name__ == "__main__":
    main()