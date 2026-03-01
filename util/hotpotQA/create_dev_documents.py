#!/usr/bin/env python3
import json
import os
import re
import shutil
import sys
import unicodedata
from pathlib import Path
from urllib.parse import unquote

# Paths
# Adjust paths relative to project root or use absolute paths as needed.
# Since this script is likely run from project root:
PROJECT_ROOT = Path(os.getcwd())
HOTPOT_QA_DEV = PROJECT_ROOT / "data/HotpotQA_Dev/QA.json"
OUTPUT_ROOT = PROJECT_ROOT / "data/HotpotQA_Dev"
ARTICLES_DIR = Path("~/Wiki2017/articles").expanduser()

# ---------- Sanitation Logic (from hotpotQA_parser.py) ----------
BAD_CHARS = re.compile(r'[\\/:#<>\[\]\{\}\|\x00-\x1F\x7F]')
SPACE_RE  = re.compile(r'\s+')

def sanitize_title(title: str) -> str:
    # Normalize title to match file system conventions used in the repo
    t = unicodedata.normalize("NFC", (title or "").strip().replace("\u00A0", " "))
    t = BAD_CHARS.sub("_", t)
    t = SPACE_RE.sub(" ", t).replace(" ", "_")
    t = re.sub(r'_+', '_', t).strip('_')
    # Truncate to reasonable length to avoid filesystem errors
    return (t.encode("utf-8")[:200].decode("utf-8", "ignore")) or "UNTITLED"

# ---------- Link Cleaning Logic (from hotpotQA_parser.py) ----------
A_TAG_RE = re.compile(
    r'<a\b[^>]*?href=(["\'])(.*?)\1[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)

def href_to_slug(href: str):
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
    # Convert HTML <a> to LaTeX-style reference format used in the dataset
    return f"{text}\\ref{{{slug}}}"

def clean_links_in_content(content: str) -> str:
    if "<a" not in content.lower():
        return content
    return A_TAG_RE.sub(replace_a_with_ref, content)

# ---------- File Finder ----------
def find_article_file(base_slug: str) -> Path:
    # Try exact match first
    exact = ARTICLES_DIR / f"{base_slug}.txt"
    if exact.exists():
        return exact
    # Try with hash suffix (some wikiextractor versions do this)
    hashed = sorted(ARTICLES_DIR.glob(f"{base_slug}__*.txt"))
    if hashed:
        return hashed[0]
    return None

def process_documents():
    if not HOTPOT_QA_DEV.exists():
        print(f"Error: {HOTPOT_QA_DEV} not found.")
        sys.exit(1)
    if not ARTICLES_DIR.exists():
        print(f"Error: {ARTICLES_DIR} not found.")
        sys.exit(1)

    print(f"Loading QA from {HOTPOT_QA_DEV}...")
    with open(HOTPOT_QA_DEV, 'r') as f:
        qa_data = json.load(f)

    # 1. Collect all unique required titles
    required_titles = set()
    for q in qa_data:
        # Supporting facts: [title, sentence_index]
        for fact in q.get('supporting_facts', []):
            required_titles.add(fact[0])
        # Context: [title, [sentences]]
        # User requested ONLY documents needed to answer (gold facts), so we skip the full context (distractors)
        # for ctx in q.get('context', []):
        #     required_titles.add(ctx[0])
    
    # Sanitize them to get 'slugs'
    slugs = {title: sanitize_title(title) for title in required_titles}
    unique_slugs = sorted(set(slugs.values()))

    print(f"Found {len(required_titles)} unique titles referenced.")
    print(f"Mapped to {len(unique_slugs)} unique filesystem slugs.")

    found_count = 0
    missing_count = 0

    for i, slug in enumerate(unique_slugs):
        if not slug:
            continue
            
        src_path = find_article_file(slug)
        if not src_path:
            # print(f"Warning: Article file for '{slug}' not found.")
            missing_count += 1
            continue
        
        found_count += 1
        
        # Prepare destination
        doc_dir = OUTPUT_ROOT / slug
        doc_dir.mkdir(parents=True, exist_ok=True)
        
        raw_output_path = doc_dir / f"{slug}_raw.txt"
        meta_output_path = doc_dir / f"{slug}_metadata.json"

        # Read, Clean, Write Raw
        try:
            with open(src_path, 'r', encoding='utf-8') as f_src:
                content = f_src.read()
            
            cleaned_content = clean_links_in_content(content)
            
            with open(raw_output_path, 'w', encoding='utf-8') as f_dest:
                f_dest.write(cleaned_content)
                
            # Write Metadata
            meta = {
                "title": slug,
                "source": "wiki",
                "yearpub": "2017",
                "author": "wikipedia",
            }
            with open(meta_output_path, 'w', encoding='utf-8') as f_meta:
                json.dump(meta, f_meta, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Error processing {slug}: {e}")
            missing_count += 1 # Count as missing/failed if error

        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(unique_slugs)} slugs...")

    print("-" * 40)
    print(f"Total Slugs: {len(unique_slugs)}")
    print(f"Successfully Extracted: {found_count}")
    print(f"Missing/Failed: {missing_count}")
    
    if missing_count > 0:
        print("WARNING: Some articles were missing or failed to process.")
    else:
        print("SUCCESS: All articles found and processed.")

if __name__ == "__main__":
    process_documents()
