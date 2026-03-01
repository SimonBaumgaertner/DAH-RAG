#!/usr/bin/env python3
"""
Create HotpotQA_Scaling dataset with 100k randomly selected Wikipedia articles.
This is a distractor set for scalability studies without QA files.

Usage:
    python3 create_scaling_dataset.py [--sample-size N] [--test-mode]
"""

import json
import os
import re
import sys
import random
import logging
import unicodedata
from pathlib import Path
from urllib.parse import unquote
from datetime import datetime
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scaling_dataset_creation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "data/HotpotQA_Scaling"
ARTICLES_DIR = Path("~/Wiki2017/articles").expanduser()
PROGRESS_FILE = OUTPUT_ROOT / ".progress.json"

# Default configuration
DEFAULT_SAMPLE_SIZE = 100000
PROGRESS_INTERVAL = 1000  # Log progress every N documents

# ---------- Sanitation Logic (from create_dev_documents.py) ----------
BAD_CHARS = re.compile(r'[\\/:#<>\[\]\{\}\|\x00-\x1F\x7F]')
SPACE_RE = re.compile(r'\s+')

def sanitize_title(title: str) -> str:
    """Normalize title to match file system conventions."""
    t = unicodedata.normalize("NFC", (title or "").strip().replace("\u00A0", " "))
    t = BAD_CHARS.sub("_", t)
    t = SPACE_RE.sub(" ", t).replace(" ", "_")
    t = re.sub(r'_+', '_', t).strip('_')
    # Truncate to reasonable length to avoid filesystem errors
    return (t.encode("utf-8")[:200].decode("utf-8", "ignore")) or "UNTITLED"

# ---------- Link Cleaning Logic (from create_dev_documents.py) ----------
A_TAG_RE = re.compile(
    r'<a\b[^>]*?href=(["\'])(.*?)\1[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)

def href_to_slug(href: str):
    """Convert href to slug format."""
    if not href:
        return None
    part = href.split("/")[-1]
    part = part.split("#", 1)[0].split("?", 1)[0]
    part = unquote(part)
    if not part.strip():
        return None
    return sanitize_title(part)

def replace_a_with_ref(match: re.Match) -> str:
    """Replace <a> tags with LaTeX-style references."""
    href = match.group(2)
    text = match.group(3)
    slug = href_to_slug(href)
    if not slug:
        return text
    return f"{text}\\ref{{{slug}}}"

def clean_links_in_content(content: str) -> str:
    """Clean HTML links in content, converting to \ref{} format."""
    if "<a" not in content.lower():
        return content
    return A_TAG_RE.sub(replace_a_with_ref, content)

# ---------- Progress Tracking ----------
def load_progress():
    """Load progress from file if it exists."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load progress file: {e}")
    return {"processed": [], "failed": [], "total_processed": 0}

def save_progress(progress):
    """Save progress to file."""
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f, indent=2)
    except Exception as e:
        logger.error(f"Could not save progress: {e}")

# ---------- Article Processing ----------
def get_all_article_files():
    """Get all article files from the Wiki2017 directory."""
    if not ARTICLES_DIR.exists():
        logger.error(f"Articles directory not found: {ARTICLES_DIR}")
        sys.exit(1)
    
    logger.info(f"Scanning for articles in {ARTICLES_DIR}...")
    article_files = list(ARTICLES_DIR.glob("*.txt"))
    logger.info(f"Found {len(article_files)} total articles")
    return article_files

def process_article(article_path: Path, progress: dict) -> bool:
    """
    Process a single article file and create the dataset structure.
    Returns True if successful, False otherwise.
    """
    try:
        # Extract slug from filename (remove .txt and any hash suffix)
        filename = article_path.stem
        slug = filename.split("__")[0]  # Remove hash suffix if present
        
        # Skip if already processed
        if slug in progress["processed"]:
            return True
        
        # Prepare destination
        doc_dir = OUTPUT_ROOT / slug
        doc_dir.mkdir(parents=True, exist_ok=True)
        
        raw_output_path = doc_dir / f"{slug}_raw.txt"
        meta_output_path = doc_dir / f"{slug}_metadata.json"
        
        # Read and clean content
        with open(article_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        cleaned_content = clean_links_in_content(content)
        
        # Write raw content
        with open(raw_output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        # Write metadata
        meta = {
            "title": slug,
            "source": "wiki",
            "yearpub": "2017",
            "author": "wikipedia",
        }
        with open(meta_output_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        # Update progress
        progress["processed"].append(slug)
        progress["total_processed"] += 1
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {article_path.name}: {e}")
        progress["failed"].append(str(article_path))
        return False

def create_scaling_dataset(sample_size: int = DEFAULT_SAMPLE_SIZE, test_mode: bool = False):
    """
    Create the HotpotQA_Scaling dataset with randomly selected articles.
    
    Args:
        sample_size: Number of articles to include (default: 100k)
        test_mode: If True, only process 100 articles for testing
    """
    if test_mode:
        sample_size = 100
        logger.info("Running in TEST MODE - processing only 100 articles")
    
    logger.info(f"Starting HotpotQA_Scaling dataset creation")
    logger.info(f"Target sample size: {sample_size:,} articles")
    logger.info(f"Output directory: {OUTPUT_ROOT}")
    
    # Create output directory
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Load progress
    progress = load_progress()
    already_processed = len(progress["processed"])
    
    if already_processed > 0:
        logger.info(f"Resuming from previous run - {already_processed:,} articles already processed")
    
    # Get all available articles
    all_articles = get_all_article_files()
    
    if len(all_articles) < sample_size:
        logger.warning(f"Only {len(all_articles):,} articles available, less than requested {sample_size:,}")
        sample_size = len(all_articles)
    
    # Filter out already processed articles
    remaining_articles = [a for a in all_articles if a.stem.split("__")[0] not in progress["processed"]]
    
    # Calculate how many more we need
    needed = sample_size - already_processed
    
    if needed <= 0:
        logger.info(f"Dataset already complete with {already_processed:,} articles!")
        return
    
    logger.info(f"Need to process {needed:,} more articles")
    
    # Randomly sample from remaining articles
    if len(remaining_articles) < needed:
        logger.warning(f"Only {len(remaining_articles):,} remaining articles, less than needed {needed:,}")
        selected_articles = remaining_articles
    else:
        selected_articles = random.sample(remaining_articles, needed)
    
    logger.info(f"Selected {len(selected_articles):,} articles to process")
    
    # Process articles
    start_time = datetime.now()
    success_count = 0
    fail_count = 0
    
    for i, article_path in enumerate(selected_articles, 1):
        if process_article(article_path, progress):
            success_count += 1
        else:
            fail_count += 1
        
        # Log progress and save periodically
        if i % PROGRESS_INTERVAL == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = i / elapsed if elapsed > 0 else 0
            remaining = len(selected_articles) - i
            eta_seconds = remaining / rate if rate > 0 else 0
            eta_minutes = eta_seconds / 60
            
            logger.info(
                f"Progress: {i:,}/{len(selected_articles):,} "
                f"({100*i/len(selected_articles):.1f}%) | "
                f"Success: {success_count:,} | Failed: {fail_count:,} | "
                f"Rate: {rate:.1f} docs/sec | ETA: {eta_minutes:.1f} min"
            )
            save_progress(progress)
    
    # Final save
    save_progress(progress)
    
    # Summary
    total_time = (datetime.now() - start_time).total_seconds()
    logger.info("-" * 80)
    logger.info(f"Dataset creation complete!")
    logger.info(f"Total articles processed: {progress['total_processed']:,}")
    logger.info(f"Successfully processed: {success_count:,}")
    logger.info(f"Failed: {fail_count:,}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Average rate: {progress['total_processed']/total_time:.1f} docs/sec")
    logger.info(f"Output directory: {OUTPUT_ROOT}")
    
    if fail_count > 0:
        logger.warning(f"{fail_count} articles failed to process. Check the log for details.")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create HotpotQA_Scaling dataset with randomly selected Wikipedia articles"
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f'Number of articles to include (default: {DEFAULT_SAMPLE_SIZE:,})'
    )
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Test mode: only process 100 articles'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset progress and start from scratch'
    )
    
    args = parser.parse_args()
    
    # Reset progress if requested
    if args.reset and PROGRESS_FILE.exists():
        logger.info("Resetting progress...")
        PROGRESS_FILE.unlink()
    
    # Create dataset
    create_scaling_dataset(sample_size=args.sample_size, test_mode=args.test_mode)

if __name__ == "__main__":
    main()
