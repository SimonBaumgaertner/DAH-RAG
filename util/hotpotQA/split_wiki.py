#!/usr/bin/env python3
"""
Split enwiki-20171001-pages-meta-current-withlinks-processed shards into
flat .txt files. Preserves <a ...> links and restores paragraph breaks.

Configure INPUT_DIR and OUTPUT_DIR below, then run:
    python3 split_wiki.py
"""

import os, sys, re, json, html, unicodedata, hashlib, bz2

# ===== CONFIGURATION =====
INPUT_DIR  = os.path.expanduser("~/Wiki2017/shards")   # where wiki_*.bz2 live
OUTPUT_DIR = os.path.expanduser("~/Wiki2017/articles") # where .txt files go
MAX_FILES  = None   # set to e.g. 10 to only process first 10 bz2 files for testing
# =========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----- filename sanitation (safe on all filesystems) -----
BAD_CHARS = re.compile(r'[\\/:#<>\[\]\{\}\|\x00-\x1F\x7F]')
SPACE_RE  = re.compile(r'\s+')

def sanitize_title(title: str) -> str:
    t = unicodedata.normalize("NFC", (title or "").strip().replace("\u00A0", " "))
    t = BAD_CHARS.sub("_", t)
    t = SPACE_RE.sub(" ", t).replace(" ", "_")
    t = re.sub(r'_+', '_', t).strip('_')
    return (t.encode("utf-8")[:200].decode("utf-8", "ignore")) or "UNTITLED"

# ----- newline normalization -----
def normalize_newlines(s: str) -> str:
    return re.sub(r'\r\n?', '\n', s)

# ----- text cleaning -----
TAG_RE = re.compile(r'<(?!/?a\b)[^>]+>', re.IGNORECASE)

def to_plain_text_preserve_links(text_field):
    """Reconstruct plain text with <a> links preserved, paragraphs separated by blank lines."""
    if isinstance(text_field, str):
        s = text_field
    elif isinstance(text_field, list):
        paragraphs = []
        for para in text_field:
            if isinstance(para, list):
                paragraphs.append(''.join(para))
            elif isinstance(para, str):
                paragraphs.append(para)
        s = '\n\n'.join(paragraphs)
    else:
        s = ""
    s = TAG_RE.sub('', s)
    s = html.unescape(s)
    s = normalize_newlines(s)
    return s

# ----- write one article -----
def write_txt(title, page_id, text):
    slug = sanitize_title(title)
    path = os.path.join(OUTPUT_DIR, f"{slug}.txt")
    if os.path.exists(path):
        suf = hashlib.md5(f"{title}\t{page_id}".encode("utf-8","ignore")).hexdigest()[:8]
        path = os.path.join(OUTPUT_DIR, f"{slug}__{suf}.txt")
    if text and not text.endswith('\n'):
        text = text + '\n'
    with open(path, "w", encoding="utf-8", newline="\n") as w:
        w.write(text)

# ----- process one bz2 file -----
def process_bz2(path):
    print(f"[INFO] Processing {path}")
    in_doc = False
    doc_title = None
    doc_id = None
    buf = []
    article_count = 0

    with bz2.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")

            # JSON-lines fast path
            if line.startswith("{") and line.endswith("}"):
                try:
                    obj = json.loads(line)
                    title = (obj.get("title") or "").strip()
                    txt = to_plain_text_preserve_links(obj.get("text"))
                    if title and txt is not None:
                        write_txt(title, obj.get("id"), txt)
                        article_count += 1
                    continue
                except json.JSONDecodeError:
                    pass

            # <doc ...> ... </doc> fallback
            if not in_doc:
                m = re.match(r'<doc\b[^>]*\btitle="([^"]+)"[^>]*\bid="([^"]+)"[^>]*>', line)
                if m:
                    in_doc, doc_title, doc_id, buf = True, m.group(1), m.group(2), []
            else:
                if line.strip() == "</doc>":
                    text = normalize_newlines(''.join(buf))
                    write_txt(doc_title or "UNTITLED", doc_id, text)
                    article_count += 1
                    in_doc, doc_title, doc_id, buf = False, None, None, []
                else:
                    buf.append(raw)

    if in_doc and buf:
        text = normalize_newlines(''.join(buf))
        write_txt(doc_title or "UNTITLED", doc_id, text)
        article_count += 1

    print(f"[INFO] Finished {path} ({article_count} articles)")

# ===== MAIN =====
def main():
    files = []
    for root, dirs, filenames in os.walk(INPUT_DIR):
        for f in filenames:
            if f.startswith("wiki_") and f.endswith(".bz2"):
                files.append(os.path.join(root, f))
    files.sort()

    if MAX_FILES:
        files = files[:MAX_FILES]

    print(f"[INFO] Found {len(files)} bz2 files to process in {INPUT_DIR}")
    total_articles = 0
    for i, path in enumerate(files, 1):
        process_bz2(path)
        print(f"[PROGRESS] Completed {i}/{len(files)} files")
    print("[DONE] All files processed.")

if __name__ == "__main__":
    main()
