#!/usr/bin/env python3
"""
MultiHopRAG Dataset Parser

Converts MultiHopRAG dataset format to the standard dataset format used by the project.
The MultiHopRAG dataset consists of:
- MultiHopRAG.json: Questions with evidence lists
- corpus.json: Document corpus with articles

Output format matches HotpotQA structure:
- Root directory with QA.json
- Document folders with metadata.json, raw.txt, and optional qa.json
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple
from urllib.parse import urlparse

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from common.data_classes.documents import Document
from common.data_classes.qa import QuestionAnswerPair, Proof, Choice


def sanitize_filename(filename: str) -> str:
    """Convert a string to a valid filename by replacing invalid characters."""
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    # Limit length to avoid filesystem issues
    if len(sanitized) > 200:
        sanitized = sanitized[:200]
    return sanitized


def extract_domain_from_url(url: str) -> str:
    """Extract domain name from URL for use as source."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except Exception:
        return "unknown_source"


def parse_publication_date(date_str: str) -> str:
    """Parse publication date string to YYYY format."""
    try:
        # Handle ISO format: 2023-09-28T12:00:00+00:00
        if 'T' in date_str:
            date_part = date_str.split('T')[0]
            return date_part.split('-')[0]  # Extract year
        # Handle other formats if needed
        return "2023"  # Default fallback
    except Exception:
        return "2023"


def create_document_from_corpus_item(item: Dict, doc_id: str) -> Document:
    """Create a Document object from a corpus item."""
    # Extract text content
    text = item.get("body", "")
    
    # Create references list (empty for MultiHopRAG as it doesn't use ref{} format)
    references: List[str] = []
    
    # Parse publication date
    pub_date_str = item.get("published_at", "")
    year = parse_publication_date(pub_date_str)
    
    # Create metadata
    metadata = {
        "title": item.get("title", doc_id),
        "author": item.get("author", "unknown"),
        "source": item.get("source", extract_domain_from_url(item.get("url", ""))),
        "yearpub": year,
        "category": item.get("category", "general"),
        "url": item.get("url", "")
    }
    
    return Document(
        id=doc_id,
        title=metadata["title"],
        author=metadata["author"],
        publication_date=None,  # We only have year, not full date
        references=references,
        text=text
    )


def create_qa_pair_from_multihop_item(item: Dict, question_id: str, title_to_doc_id: Dict[str, str]) -> QuestionAnswerPair:
    """Create a QuestionAnswerPair from a MultiHopRAG item."""
    # Create proofs from evidence_list
    proofs: List[Proof] = []
    for evidence in item.get("evidence_list", []):
        # Map title to document ID using the mapping from corpus
        title = evidence.get("title", "unknown")
        doc_id = title_to_doc_id.get(title, sanitize_filename(title))
        context = evidence.get("fact", "")
        proofs.append(Proof(document_id=doc_id, context=context))
    
    # MultiHopRAG doesn't have multiple choice, so create empty choices
    choices: List[Choice] = []
    
    return QuestionAnswerPair(
        question_id=question_id,
        question=item.get("query", ""),
        choices=choices,
        correct_answer=item.get("answer", ""),
        proofs=proofs
    )


def parse_multihop_dataset(input_dir: Path, output_dir: Path) -> None:
    """
    Parse MultiHopRAG dataset and convert to standard format.
    
    Args:
        input_dir: Directory containing MultiHopRAG.json and corpus.json
        output_dir: Directory to write the converted dataset
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load corpus data
    corpus_path = input_dir / "corpus.json"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    
    print(f"Loading corpus from {corpus_path}")
    with corpus_path.open(encoding="utf-8") as f:
        corpus_data = json.load(f)
    
    # Load questions data
    questions_path = input_dir / "MultiHopRAG.json"
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    
    print(f"Loading questions from {questions_path}")
    with questions_path.open(encoding="utf-8") as f:
        questions_data = json.load(f)
    
    # Create title to document ID mapping for corpus (needed for proof creation)
    title_to_doc_id: Dict[str, str] = {}
    for item in corpus_data:
        title = item.get("title", "")
        if title:
            doc_id = sanitize_filename(title)
            title_to_doc_id[title] = doc_id
    
    # Create documents from corpus
    print(f"Creating {len(corpus_data)} documents...")
    all_qa_pairs: List[QuestionAnswerPair] = []
    
    for i, item in enumerate(corpus_data):
        title = item.get("title", f"document_{i}")
        doc_id = sanitize_filename(title)
        
        # Create document folder
        doc_folder = output_dir / doc_id
        doc_folder.mkdir(exist_ok=True)
        
        # Create document object
        document = create_document_from_corpus_item(item, doc_id)
        
        # Write metadata
        metadata = {
            "title": document.title,
            "author": document.author,
            "source": item.get("source", "unknown"),
            "yearpub": parse_publication_date(item.get("published_at", "")),
            "category": item.get("category", "general"),
            "url": item.get("url", "")
        }
        
        metadata_path = doc_folder / f"{doc_id}_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Write raw text
        raw_path = doc_folder / f"{doc_id}_raw.txt"
        with raw_path.open("w", encoding="utf-8") as f:
            f.write(document.text)
    
    # Create QA pairs from questions
    print(f"Creating {len(questions_data)} QA pairs...")
    for i, item in enumerate(questions_data):
        question_id = f"multihop_{i:06d}"
        qa_pair = create_qa_pair_from_multihop_item(item, question_id, title_to_doc_id)
        all_qa_pairs.append(qa_pair)
    
    # Write main QA.json file
    qa_data = []
    for qa_pair in all_qa_pairs:
        qa_dict = {
            "question_id": qa_pair.question_id,
            "question": qa_pair.question,
            "choices": [{"label": c.label, "text": c.text} for c in qa_pair.choices],
            "correct_answer": qa_pair.correct_answer,
            "proofs": [{"document_id": p.document_id, "context": p.context} for p in qa_pair.proofs]
        }
        qa_data.append(qa_dict)
    
    qa_path = output_dir / "QA.json"
    with qa_path.open("w", encoding="utf-8") as f:
        json.dump(qa_data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully parsed MultiHopRAG dataset!")
    print(f"Created {len(corpus_data)} documents in {output_dir}")
    print(f"Created {len(all_qa_pairs)} QA pairs")
    print(f"Output written to: {output_dir}")


def main():
    
    input_dir = Path(__file__).parent
    output_dir =Path(__file__).parent / "dataset"
    
    try:
        parse_multihop_dataset(input_dir, output_dir)
    except Exception as e:
        print(f"Error parsing dataset: {e}")
        raise


if __name__ == "__main__":
    main()
