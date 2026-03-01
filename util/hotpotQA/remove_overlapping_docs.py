#!/usr/bin/env python3
"""
Remove documents from HotpotQA_Scaling that are also in HotpotQA_1k.
This ensures the datasets are completely separate.
"""

import os
import shutil
import json
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
HOTPOT_1K = PROJECT_ROOT / "data/HotpotQA_1k"
HOTPOT_SCALING = PROJECT_ROOT / "data/HotpotQA_Scaling"
PROGRESS_FILE = HOTPOT_SCALING / ".progress.json"

def get_document_names(dataset_path):
    """Get all document folder names from a dataset."""
    if not dataset_path.exists():
        print(f"Error: {dataset_path} does not exist")
        return set()
    
    # Get all directories (excluding hidden files and QA.json)
    doc_names = set()
    for item in dataset_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            doc_names.add(item.name)
    
    return doc_names

def remove_overlapping_documents():
    """Remove documents from HotpotQA_Scaling that exist in HotpotQA_1k."""
    print("Analyzing datasets...")
    
    # Get document names from both datasets
    docs_1k = get_document_names(HOTPOT_1K)
    docs_scaling = get_document_names(HOTPOT_SCALING)
    
    print(f"Documents in HotpotQA_1k: {len(docs_1k)}")
    print(f"Documents in HotpotQA_Scaling: {len(docs_scaling)}")
    
    # Find overlapping documents
    overlapping = docs_1k & docs_scaling
    
    print(f"Overlapping documents: {len(overlapping)}")
    
    if len(overlapping) == 0:
        print("No overlapping documents found. Datasets are already separate.")
        return
    
    print(f"\nRemoving {len(overlapping)} overlapping documents from HotpotQA_Scaling...")
    
    removed_count = 0
    failed_count = 0
    
    for doc_name in overlapping:
        doc_path = HOTPOT_SCALING / doc_name
        try:
            if doc_path.exists():
                shutil.rmtree(doc_path)
                removed_count += 1
                if removed_count % 100 == 0:
                    print(f"Removed {removed_count}/{len(overlapping)} documents...")
        except Exception as e:
            print(f"Error removing {doc_name}: {e}")
            failed_count += 1
    
    # Update progress file
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'r') as f:
                progress = json.load(f)
            
            # Remove overlapping documents from processed list
            original_count = len(progress.get("processed", []))
            progress["processed"] = [doc for doc in progress.get("processed", []) if doc not in overlapping]
            progress["total_processed"] = len(progress["processed"])
            
            with open(PROGRESS_FILE, 'w') as f:
                json.dump(progress, f, indent=2)
            
            print(f"\nUpdated progress file: removed {original_count - len(progress['processed'])} entries")
        except Exception as e:
            print(f"Warning: Could not update progress file: {e}")
    
    print("\n" + "="*80)
    print("Summary:")
    print(f"  Successfully removed: {removed_count}")
    print(f"  Failed to remove: {failed_count}")
    print(f"  Remaining documents in HotpotQA_Scaling: {len(docs_scaling) - removed_count}")
    print("="*80)

if __name__ == "__main__":
    remove_overlapping_documents()
