#!/usr/bin/env python3
"""
QA Parser for HotpotQA datasets.

This script fixes the QA.json files by extracting the correct supporting facts
from the hotpot_dev_distractor_v1.json file, which contains the actual proofs
in the context field.

Usage:
    python3 qa_parser.py <dataset_dir>
    
Example:
    python3 qa_parser.py ../../data/HotpotQA_100
"""

import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------- Title sanitization (from hotpotQA_parser.py) ----------
BAD_CHARS = re.compile(r'[\\/:#<>\[\]\{\}\|\x00-\x1F\x7F]')
SPACE_RE  = re.compile(r'\s+')

def sanitize_title(title: str) -> str:
    """Sanitize title to match document folder naming convention."""
    t = unicodedata.normalize("NFC", (title or "").strip().replace("\u00A0", " "))
    t = BAD_CHARS.sub("_", t)
    t = SPACE_RE.sub(" ", t).replace(" ", "_")
    t = re.sub(r'_+', '_', t).strip('_')
    return (t.encode("utf-8")[:200].decode("utf-8", "ignore")) or "UNTITLED"

# ---------- Link cleaning (from hotpotQA_parser.py) ----------
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
        # Extract slug from href (simplified version)
        if href.startswith("/wiki/"):
            slug = href.split("/")[-1].split("#")[0].split("?")[0]
            return f"{text_content}\\ref{{{slug}}}"
        return text_content
    
    return A_TAG_RE.sub(replace_a_with_ref, text)

# ---------- Data structures ----------
class DistractorEntry:
    """Represents an entry from the distractor JSON."""
    __slots__ = ("qid", "question", "answer", "supporting_facts", "context_map", "raw_data")
    
    def __init__(self, data: dict):
        self.qid = str(data.get("_id", ""))
        self.question = data.get("question", "")
        self.answer = data.get("answer", "")
        self.supporting_facts: List[Tuple[str, int]] = []
        for fact in data.get("supporting_facts", []):
            if isinstance(fact, list) and len(fact) == 2:
                title, idx = fact
                if isinstance(title, str) and isinstance(idx, int):
                    self.supporting_facts.append((title, idx))
        
        # Build context map: title -> list of sentences
        self.context_map: Dict[str, List[str]] = {}
        for context_item in data.get("context", []):
            if isinstance(context_item, list) and len(context_item) == 2:
                title, sentences = context_item
                if isinstance(title, str) and isinstance(sentences, list):
                    self.context_map[title] = sentences
        
        self.raw_data = data

class QAEntry:
    """Represents an entry from the existing QA.json."""
    __slots__ = ("question_id", "question", "choices", "correct_answer", "proofs", "raw_data")
    
    def __init__(self, data: dict):
        self.question_id = data.get("question_id", "")
        self.question = data.get("question", "")
        self.choices = data.get("choices", [])
        self.correct_answer = data.get("correct_answer", "")
        self.proofs = data.get("proofs", [])
        self.raw_data = data

# ---------- I/O functions ----------
def load_distractor_data(distractor_path: Path) -> Dict[str, DistractorEntry]:
    """Load distractor JSON and return mapping from question_id to DistractorEntry."""
    print(f"📥 Loading distractor data from {distractor_path}")
    with open(distractor_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    entries = {}
    for item in data:
        entry = DistractorEntry(item)
        entries[entry.qid] = entry
    
    print(f"✅ Loaded {len(entries)} distractor entries")
    return entries

def load_qa_data(qa_path: Path) -> List[QAEntry]:
    """Load existing QA.json and return list of QAEntry objects."""
    print(f"📥 Loading QA data from {qa_path}")
    with open(qa_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    entries = [QAEntry(item) for item in data]
    print(f"✅ Loaded {len(entries)} QA entries")
    return entries

def save_qa_data(qa_entries: List[QAEntry], output_path: Path) -> None:
    """Save QA entries to JSON file."""
    print(f"💾 Saving updated QA data to {output_path}")
    data = [entry.raw_data for entry in qa_entries]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(data)} QA entries")

# ---------- Proof extraction ----------
def extract_proofs_from_distractor(qa_entry: QAEntry, distractor_entry: DistractorEntry) -> List[dict]:
    """Extract correct proofs from distractor entry for a QA entry."""
    proofs = []
    seen_facts: Set[Tuple[str, int]] = set()
    
    for title, sentence_idx in distractor_entry.supporting_facts:
        # Skip duplicates
        if (title, sentence_idx) in seen_facts:
            continue
        seen_facts.add((title, sentence_idx))
        
        # Get sentences for this title from context
        sentences = distractor_entry.context_map.get(title, [])
        
        if sentence_idx < len(sentences):
            context_text = sentences[sentence_idx]
            # Clean any links in the context
            context_text = clean_links_in_text(context_text)
            
            # Convert title to document ID using same sanitization as hotpotQA_parser
            doc_id = sanitize_title(title)
            
            proofs.append({
                "document_id": doc_id,
                "context": context_text
            })
        else:
            print(f"⚠️  Warning: Sentence index {sentence_idx} out of range for '{title}' (only {len(sentences)} sentences)")
    
    return proofs

# ---------- Main processing ----------
def process_dataset(dataset_dir: Path) -> None:
    """Process a single dataset directory."""
    print(f"🔍 Processing dataset: {dataset_dir}")
    
    # Paths
    qa_path = dataset_dir / "QA.json"
    distractor_path = Path(__file__).parent / "hotpot_dev_distractor_v1.json"
    
    # Check if files exist
    if not qa_path.exists():
        print(f"❌ QA.json not found: {qa_path}")
        return
    
    if not distractor_path.exists():
        print(f"❌ Distractor JSON not found: {distractor_path}")
        return
    
    # Load data
    distractor_entries = load_distractor_data(distractor_path)
    qa_entries = load_qa_data(qa_path)
    
    # Process each QA entry
    updated_count = 0
    missing_count = 0
    
    for qa_entry in qa_entries:
        # Find matching distractor entry
        distractor_entry = distractor_entries.get(qa_entry.question_id)
        
        if distractor_entry is None:
            print(f"⚠️  No distractor entry found for question {qa_entry.question_id}")
            missing_count += 1
            continue
        
        # Extract correct proofs
        new_proofs = extract_proofs_from_distractor(qa_entry, distractor_entry)
        
        if new_proofs:
            # Update the QA entry
            qa_entry.proofs = new_proofs
            qa_entry.raw_data["proofs"] = new_proofs
            updated_count += 1
        else:
            print(f"⚠️  No proofs extracted for question {qa_entry.question_id}")
    
    # Save updated QA data
    save_qa_data(qa_entries, qa_path)
    
    print(f"📊 Summary:")
    print(f"  - Updated: {updated_count}")
    print(f"  - Missing distractor data: {missing_count}")
    print(f"  - Total processed: {len(qa_entries)}")

# ---------- Main ----------
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 qa_parser.py <dataset_dir>")
        print("Example: python3 qa_parser.py ../../data/HotpotQA_100")
        sys.exit(1)
    
    dataset_dir = Path(sys.argv[1])
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    
    process_dataset(dataset_dir)

if __name__ == "__main__":
    main()
