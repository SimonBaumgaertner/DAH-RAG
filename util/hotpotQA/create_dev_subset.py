import json
import os
import re
import sys
import unicodedata
from urllib.parse import unquote

# ---------- Helpers from qa_parser.py / hotpotQA_parser.py ----------

BAD_CHARS = re.compile(r'[\\/:#<>\[\]\{\}\|\x00-\x1F\x7F]')
SPACE_RE  = re.compile(r'\s+')

def sanitize_title(title: str) -> str:
    """Sanitize title to match document folder naming convention."""
    t = unicodedata.normalize("NFC", (title or "").strip().replace("\u00A0", " "))
    t = BAD_CHARS.sub("_", t)
    t = SPACE_RE.sub(" ", t).replace(" ", "_")
    t = re.sub(r'_+', '_', t).strip('_')
    return (t.encode("utf-8")[:200].decode("utf-8", "ignore")) or "UNTITLED"

A_TAG_RE = re.compile(
    r'<a\b[^>]*?href=(["\'])(.*?)\1[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)

def clean_links_in_text(text: str) -> str:
    """Clean <a> tags from text, converting them to \\ref{slug} format."""
    if "<a" not in text.lower():
        return text
    
    def replace_a_with_ref(match: re.Match) -> str:
        href = match.group(2)
        text_content = match.group(3)
        # Extract slug from href
        # Typically /wiki/Some_Title
        if href:
            part = href.split("/")[-1].split("#")[0].split("?")[0]
            part = unquote(part)
            slug = sanitize_title(part)
            if slug:
                return f"{text_content}\\ref{{{slug}}}"
        return text_content
    
    return A_TAG_RE.sub(replace_a_with_ref, text)

# ---------- Transformation ----------

def transform_entry(raw_entry: dict) -> dict:
    """Transform raw Hotpot entry to the desired schema."""
    # 1. Basic fields
    new_entry = {
        "question_id": raw_entry.get("_id", ""),
        "question": raw_entry.get("question", ""),
        "choices": [],  # HotpotQA has no MC choices
        "correct_answer": raw_entry.get("answer", ""),
        "proofs": []
    }

    # 2. Build proofs
    # supporting_facts is [[title, sent_idx], ...]
    # context is [[title, [sent0, sent1, ...]], ...]
    
    # Build context map for O(1) lookup
    ctx_map = {}
    for item in raw_entry.get("context", []):
        if len(item) == 2:
            title, sentences = item
            ctx_map[title] = sentences

    seen_facts = set()
    proofs = []
    
    for fact in raw_entry.get("supporting_facts", []):
        if len(fact) != 2:
            continue
        title, idx = fact
        
        if (title, idx) in seen_facts:
            continue
        seen_facts.add((title, idx))
        
        sentences = ctx_map.get(title, [])
        if idx < len(sentences):
            raw_sent = sentences[idx]
            clean_sent = clean_links_in_text(raw_sent)
            
            proofs.append({
                "document_id": sanitize_title(title),
                "context": clean_sent
            })
    
    new_entry["proofs"] = proofs
    return new_entry

def create_disjoint_subset():
    # Paths
    path_full_dev = 'util/hotpotQA/hotpot_dev_distractor_v1.json'
    path_in_use = 'data/HotpotQA_10k/QA.json'
    output_dir = 'data/HotpotQA_Dev'
    path_output = os.path.join(output_dir, 'QA.json')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading full dev set from {path_full_dev}...")
    with open(path_full_dev, 'r') as f:
        full_dev = json.load(f)
    print(f"Total items in full dev: {len(full_dev)}")

    print(f"Loading existing 10k set from {path_in_use}...")
    with open(path_in_use, 'r') as f:
        in_use = json.load(f)
    print(f"Total items in 10k set: {len(in_use)}")

    # Create exclusion set of IDs
    def get_id(q):
        return q.get('_id') or q.get('question_id')

    exclude_ids = set(get_id(q) for q in in_use)
    print(f"Unique IDs to exclude: {len(exclude_ids)}")

    # Filter AND Transform
    new_subset = []
    for q in full_dev:
        if get_id(q) not in exclude_ids:
            new_subset.append(transform_entry(q))
            
    print(f"Created new filtered & transformed subset with {len(new_subset)} items.")

    # Verification
    # Check for zero proofs (optional warning)
    zero_proofs = sum(1 for q in new_subset if not q['proofs'])
    if zero_proofs > 0:
        print(f"Warning: {zero_proofs} questions have 0 proofs.")

    # Save
    print(f"Saving to {path_output}...")
    with open(path_output, 'w') as f:
        json.dump(new_subset, f, indent=2)
    
    print("Done.")

if __name__ == "__main__":
    create_disjoint_subset()
