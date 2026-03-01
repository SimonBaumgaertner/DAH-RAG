#!/usr/bin/env python3
"""
Transform PubMedQA -> your ontology + optional augmentation.

Creates per PMID:
  data 🗃️/{pmid}/{pmid}_metadata.json
  data 🗃️/{pmid}/{pmid}_qa.json
  data 🗃️/{pmid}/{pmid}_raw.txt

- Base mode: parses pubmedQA_questions.json and fetches abstracts via EFetch.
- Augmentation: pulls ~N extra PubMed articles via ESearch and adds them
  with EMPTY QA lists (metadata + raw only).

Set constants in CONFIG; no CLI args needed.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import time
import xml.etree.ElementTree as ET
import requests
import random

# ---- CONFIG (edit these) ----
INPUT_JSON = "pubmedQA_questions.json"   # path to your PubMedQA json (e.g., ori_pqal.json)
OUTPUT_DIR = "data"                      # output folder created at the same directory level (will create PubMedQA_10k)

# E-utilities pacing
BATCH_SIZE = 150                         # EFetch tolerates moderate batches
REQUEST_TIMEOUT = 30
PAUSE_BETWEEN_CALLS = 0.4                # ~3 req/s guideline without API key

# Augmentation controls for PubMedQA_10k
AUGMENT_N = 6000                         # 0 to disable; otherwise how many extra docs to add
AUGMENT_QUERY = (
    "AND hasabstract[text] AND english[lang] NOT (letter[pt] OR editorial[pt])"
)
YEAR_FROM: Optional[int] = 2000          # None to disable lower bound
YEAR_TO: Optional[int] = 2025            # None to disable upper bound

# Distractor configuration
DISTRACTOR_RATIO = 3                     # Number of distractor docs per question (3:1 ratio)
DIFFICULTY_LEVELS = {
    "easy": {"distractors": 1, "conflict_ratio": 0.1},      # 1 distractor, 10% conflicting
    "medium": {"distractors": 2, "conflict_ratio": 0.3},    # 2 distractors, 30% conflicting  
    "hard": {"distractors": 4, "conflict_ratio": 0.5}       # 4 distractors, 50% conflicting
}

# Optional: identify yourself to NCBI (recommended)
TOOL_NAME = "pubmedqa_augment"
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "")    # set if you like
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")      # speeds & raises limits

# Endpoints
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ESEARCH_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

# HTTP headers
HEADERS = {
    "Accept": "application/json,text/xml,application/xml;q=0.9,*/*;q=0.8",
    "User-Agent": f"{TOOL_NAME}/1.0 (+{CONTACT_EMAIL})" if CONTACT_EMAIL else f"{TOOL_NAME}/1.0",
}

# Retry/backoff
MAX_RETRIES = 5
BACKOFF_BASE = 0.8  # seconds
JITTER = 0.3

# QA mapping
CHOICE_ORDER = [("A", "Yes"), ("B", "No"), ("C", "Maybe")]
ANSWER_TO_LABEL = {"yes": "A", "no": "B", "maybe": "C"}
# -----------------------------


def common_params() -> dict:
    params = {}
    if TOOL_NAME:
        params["tool"] = TOOL_NAME
    if CONTACT_EMAIL:
        params["email"] = CONTACT_EMAIL
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    return params


def _get_with_retries(url: str, params: dict) -> requests.Response:
    """GET with exponential backoff and jitter on network/HTTP errors."""
    attempt = 0
    while True:
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            # Some E-utilities return 200 with HTML error; we still check status for non-200s.
            r.raise_for_status()
            return r
        except Exception as e:
            attempt += 1
            if attempt > MAX_RETRIES:
                raise
            sleep_s = BACKOFF_BASE * (2 ** (attempt - 1)) + random.uniform(0, JITTER)
            print(f"[RETRY] GET {url} failed (attempt {attempt}/{MAX_RETRIES}): {e}. Sleeping {sleep_s:.2f}s")
            time.sleep(sleep_s)


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
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", **common_params()}
    r = _get_with_retries(EUTILS_BASE, params=params)

    root = ET.fromstring(r.text)
    out: Dict[str, dict] = {}
    for art in root.findall(".//PubmedArticle"):
        pmid = (art.findtext(".//PMID") or "").strip()
        if not pmid:
            continue
        title = (art.findtext(".//ArticleTitle") or "").strip()

        # year: try ArticleDate/Journal Issue PubDate, then DateCreated
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

        # abstract (concatenate structured parts)
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


def _parse_esearch_xml_for_ids(xml_text: str) -> List[str]:
    try:
        root = ET.fromstring(xml_text)
        return [el.text.strip() for el in root.findall(".//IdList/Id") if el.text]
    except Exception:
        return []


def extract_keywords_from_question(question: str) -> List[str]:
    """Extract biomedical keywords from a question for targeted search."""
    # Simple keyword extraction - could be enhanced with NER
    question_lower = question.lower()
    keywords = []
    
    # Common biomedical terms
    biomedical_terms = [
        "treatment", "therapy", "drug", "medication", "disease", "disorder", 
        "syndrome", "cancer", "tumor", "infection", "bacteria", "virus",
        "protein", "gene", "mutation", "cell", "tissue", "organ",
        "clinical", "trial", "study", "patient", "diagnosis", "symptom"
    ]
    
    for term in biomedical_terms:
        if term in question_lower:
            keywords.append(term)
    
    # Extract potential medical terms (words that might be diseases/treatments)
    words = question.split()
    for word in words:
        if len(word) > 4 and word.isalpha():
            # Simple heuristic: capitalized words or words ending in common medical suffixes
            if (word[0].isupper() or 
                word.endswith(('itis', 'osis', 'emia', 'uria', 'pathy', 'plasia'))):
                keywords.append(word.lower())
    
    return keywords[:5]  # Limit to top 5 keywords


def esearch_pubmed_ids(term: str, target_n: int, mindate: Optional[str] = None, maxdate: Optional[str] = None) -> List[str]:
    """
    Use ESearch to collect up to target_n PMIDs matching `term`.
    Robust to non-JSON responses (falls back to XML) and retries on transient failures.
    """
    if target_n <= 0:
        return []

    # Normalize term whitespace to reduce chances of weird server parsing
    term = " ".join((term or "").split())

    collected: List[str] = []
    retstart = 0
    page = 500  # safe chunk size
    while len(collected) < target_n:
        params = {
            "db": "pubmed",
            "term": term,
            "retmode": "json",
            "retmax": page,
            "retstart": retstart,
            "sort": "relevance",
            **common_params(),
        }
        if mindate or maxdate:
            params["datetype"] = "pdat"
            if mindate:
                params["mindate"] = mindate
            if maxdate:
                params["maxdate"] = maxdate

        r = _get_with_retries(ESEARCH_BASE, params=params)

        ids: List[str] = []
        # Prefer JSON; if it fails, try XML fallback.
        try:
            js = r.json()
            ids = js.get("esearchresult", {}).get("idlist", []) or []
        except Exception:
            # Not JSON; maybe XML or HTML
            text_head = r.text.lstrip()
            if text_head.startswith("<"):
                ids = _parse_esearch_xml_for_ids(r.text)
            else:
                # Unexpected plaintext; log a tiny snippet and skip this page
                snippet = r.text[:200].replace("\n", " ")
                print(f"[WARN] ESearch returned non-JSON/non-XML. Skipping page at retstart={retstart}. Snippet: {snippet}")

        if not ids:
            # No more results (or this page failed). Bail to avoid infinite loop.
            print(f"[INFO] No IDs returned at retstart={retstart}. Stopping ESearch pagination.")
            break

        collected.extend(str(x) for x in ids)
        retstart += page
        time.sleep(PAUSE_BETWEEN_CALLS)

    return collected[:target_n]


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


def make_metadata(pmid: str, fetched: dict, fallback_year: Optional[str], abstract_text: str) -> dict:
    # Map into your ontology’s fields
    yearpub = fetched.get("year")
    if yearpub is None and fallback_year:
        try:
            yearpub = int(fallback_year) if str(fallback_year).isdigit() else None
        except Exception:
            yearpub = None

    author_str = fetched.get("authors") or None

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


# -------------- DISTRACTOR SELECTION -------------- #

def find_distractor_documents(question: str, 
                             existing_pmids: Set[str], 
                             num_distractors: int,
                             conflict_ratio: float) -> List[str]:
    """
    Find distractor documents for a question.
    Returns a mix of related but irrelevant and potentially conflicting documents.
    """
    keywords = extract_keywords_from_question(question)
    if not keywords:
        print(f"      ⚠️  No keywords extracted from question")
        return []
    
    print(f"      🔑 Extracted keywords: {keywords}")
    
    # Create search terms for distractors
    distractor_terms = []
    
    # 1. Related but broader terms (should be irrelevant)
    for keyword in keywords[:3]:  # Use top 3 keywords
        distractor_terms.append(f'"{keyword}"[All Fields]')
    
    # 2. Add some conflicting terms (e.g., if question is about treatment, search for side effects)
    conflict_terms = []
    question_lower = question.lower()
    if "treatment" in question_lower or "therapy" in question_lower:
        conflict_terms.extend(["side effects", "adverse", "complications"])
    if "efficacy" in question_lower or "effective" in question_lower:
        conflict_terms.extend(["ineffective", "resistance", "failure"])
    if "benefit" in question_lower:
        conflict_terms.extend(["risk", "harm", "contraindication"])
    
    for term in conflict_terms[:2]:  # Limit conflict terms
        distractor_terms.append(f'"{term}"[All Fields]')
    
    print(f"      🔍 Search terms: {distractor_terms}")
    
    # Search for distractor documents
    all_distractors = []
    for i, term in enumerate(distractor_terms):
        try:
            print(f"      🔎 Searching term {i+1}/{len(distractor_terms)}: {term}")
            # Search for documents with these terms
            candidates = esearch_pubmed_ids(
                term, 
                target_n=num_distractors * 2,  # Get more candidates
                mindate=str(YEAR_FROM) if YEAR_FROM else None,
                maxdate=str(YEAR_TO) if YEAR_TO else None
            )
            all_distractors.extend(candidates)
            print(f"      ✅ Found {len(candidates)} candidates for term: {term}")
        except Exception as e:
            print(f"      ❌ Failed to search for distractor term '{term}': {e}")
    
    # Remove duplicates and existing PMIDs
    unique_distractors = []
    seen = set(existing_pmids)
    for pmid in all_distractors:
        if pmid not in seen:
            unique_distractors.append(pmid)
            seen.add(pmid)
        if len(unique_distractors) >= num_distractors:
            break
    
    print(f"      📊 Total candidates: {len(all_distractors)}, Unique: {len(unique_distractors)}")
    return unique_distractors[:num_distractors]


# -------------- AUGMENTATION -------------- #

def augment_corpus(out_root: Path,
                   existing_pmids: Set[str],
                   target_new: int,
                   query: str,
                   year_from: Optional[int],
                   year_to: Optional[int]) -> int:
    """
    Collect up to target_new additional PubMed PMIDs via ESearch, excluding existing ones.
    Fetch metadata/abstract and write files with EMPTY QA lists.
    Returns the number of docs written.
    """
    if target_new <= 0:
        return 0

    mindate = str(year_from) if year_from else None
    maxdate = str(year_to) if year_to else None

    print(f"[AUGMENT] Searching PubMed for up to {target_new} docs...")
    # Pull more candidates than we need to improve dedup chances
    candidate_ids = esearch_pubmed_ids(query, target_new * 3, mindate=mindate, maxdate=maxdate)
    if not candidate_ids:
        print("[AUGMENT] No candidates found via ESearch.")
        return 0

    unique_new = []
    seen = set(existing_pmids)
    for pid in candidate_ids:
        if pid not in seen:
            unique_new.append(pid)
            seen.add(pid)
        if len(unique_new) >= target_new:
            break

    if not unique_new:
        print("[AUGMENT] All candidate PMIDs were already present.")
        return 0

    print(f"[AUGMENT] Will fetch {len(unique_new)} new docs.")

    written = 0
    for batch in batched(unique_new, BATCH_SIZE):
        try:
            fetched = efetch_pubmed_batch(batch)
        except Exception as e:
            print(f"[WARN] EFetch failed for augment batch starting {batch[0]}: {e}")
            fetched = {}
        time.sleep(PAUSE_BETWEEN_CALLS)

        for pmid in batch:
            meta_src = fetched.get(pmid, {"title": f"PMID {pmid}", "authors": None, "year": None, "abstract": ""})
            abs_text = meta_src.get("abstract", "") or ""
            meta = make_metadata(pmid=pmid, fetched=meta_src, fallback_year=None, abstract_text=abs_text)
            # Empty QA list for augmented docs
            write_doc(pmid, meta, [], abs_text, out_root)
            written += 1

    return written


# ------------------- MAIN ------------------- #

def main():
    print("🚀 Starting PubMedQA_10k dataset creation with distractors...")
    print(f"📋 Configuration:")
    print(f"   - Target dataset size: 10,000 documents")
    print(f"   - Distractor ratio: {DISTRACTOR_RATIO}:1")
    print(f"   - Difficulty levels: {list(DIFFICULTY_LEVELS.keys())}")
    print(f"   - Augmentation: {AUGMENT_N} additional documents")
    print(f"   - Year range: {YEAR_FROM}-{YEAR_TO}")
    print()
    
    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    out_root = script_dir / OUTPUT_DIR
    print(f"📁 Script directory: {script_dir}")
    print(f"📁 Output directory: {out_root}")
    ensure_dir(out_root)

    # 1) Load PubMedQA and write structured outputs
    input_json_path = script_dir / INPUT_JSON
    print(f"📖 Loading input JSON: {input_json_path}")
    src = load_pubmedqa(str(input_json_path))
    pmids = list(src.keys())
    
    print(f"✅ Loaded {len(pmids)} base PubMedQA documents")
    print(f"🎯 Target: PubMedQA_10k with {DISTRACTOR_RATIO}:1 distractor ratio")
    print()
    # Fetch metadata in batches for the base set
    print("🔍 Fetching metadata for base documents...")
    fetched_all: Dict[str, dict] = {}
    total_batches = (len(pmids) + BATCH_SIZE - 1) // BATCH_SIZE
    batch_num = 0
    
    for batch in batched(pmids, BATCH_SIZE):
        batch_num += 1
        print(f"   📦 Batch {batch_num}/{total_batches}: Fetching {len(batch)} documents (PMID {batch[0]} to {batch[-1]})")
        try:
            fetched = efetch_pubmed_batch(batch)
            fetched_all.update(fetched)
            print(f"   ✅ Successfully fetched {len(fetched)} documents")
        except Exception as e:
            print(f"   ❌ EFetch failed for batch starting {batch[0]}: {e}")
        time.sleep(PAUSE_BETWEEN_CALLS)
    
    print(f"✅ Metadata fetching complete: {len(fetched_all)}/{len(pmids)} documents fetched")
    print()

    # Process each question with distractors
    print("🎯 Starting distractor processing...")
    all_pmids = set(pmids)
    questions_processed = 0
    
    for pmid in pmids:
        item = src[pmid]
        question = item.get("QUESTION") or item.get("question") or ""
        
        if not question:
            print(f"⚠️  No question found for PMID {pmid}, skipping")
            continue
            
        # Determine difficulty level (distribute evenly)
        difficulty = list(DIFFICULTY_LEVELS.keys())[questions_processed % len(DIFFICULTY_LEVELS)]
        difficulty_config = DIFFICULTY_LEVELS[difficulty]
        num_distractors = difficulty_config["distractors"]
        
        print(f"🔍 Processing PMID {pmid} (difficulty: {difficulty}, {num_distractors} distractors)")
        print(f"   Question: {question[:100]}{'...' if len(question) > 100 else ''}")
        
        # Find distractor documents for this question
        print(f"   🔎 Searching for {num_distractors} distractor documents...")
        distractor_pmids = find_distractor_documents(
            question=question,
            existing_pmids=all_pmids,
            num_distractors=num_distractors,
            conflict_ratio=difficulty_config["conflict_ratio"]
        )
        print(f"   ✅ Found {len(distractor_pmids)} distractor documents: {distractor_pmids}")
        
        # Add distractors to our collection
        all_pmids.update(distractor_pmids)
        
        # Fetch metadata for distractors
        if distractor_pmids:
            print(f"   📥 Fetching metadata for {len(distractor_pmids)} distractor documents...")
            for batch in batched(distractor_pmids, BATCH_SIZE):
                try:
                    fetched = efetch_pubmed_batch(batch)
                    fetched_all.update(fetched)
                    print(f"   ✅ Fetched metadata for {len(fetched)} distractor documents")
                except Exception as e:
                    print(f"   ❌ EFetch failed for distractor batch starting {batch[0]}: {e}")
                time.sleep(PAUSE_BETWEEN_CALLS)
        
        # Write the main document with QA
        print(f"   💾 Writing main document {pmid} with QA...")
        fetched = fetched_all.get(pmid, {})
        abs_text = fetched.get("abstract") or reconstruct_abstract_from_item(item)
        meta = make_metadata(
            pmid=pmid,
            fetched=fetched,
            fallback_year=item.get("YEAR"),
            abstract_text=abs_text,
        )
        qa_item = make_qa_item(pmid, item)
        write_doc(pmid, meta, [qa_item], abs_text, out_root)
        
        # Write distractor documents (empty QA)
        for distractor_pmid in distractor_pmids:
            distractor_fetched = fetched_all.get(distractor_pmid, {})
            distractor_abs = distractor_fetched.get("abstract", "")
            distractor_meta = make_metadata(
                pmid=distractor_pmid,
                fetched=distractor_fetched,
                fallback_year=None,
                abstract_text=distractor_abs,
            )
            write_doc(distractor_pmid, distractor_meta, [], distractor_abs, out_root)
        
        questions_processed += 1
        print(f"   ✅ Completed PMID {pmid} - Total documents so far: {len(all_pmids)}")
        
        # Progress update
        if questions_processed % 10 == 0:
            print(f"📊 Progress: {questions_processed}/{len(pmids)} questions processed, {len(all_pmids)} total documents")
            print()

    print(f"✅ Base processing complete!")
    print(f"   📊 Wrote {len(pmids)} PubMedQA documents with {len(all_pmids) - len(pmids)} distractors")
    print(f"   📊 Total documents so far: {len(all_pmids)}")
    print()

    # 2) Augment with additional PubMed docs (empty QA) to reach 10k
    remaining_needed = 10000 - len(all_pmids)
    written = 0  # Initialize in case no augmentation is needed
    
    if remaining_needed > 0 and AUGMENT_N > 0:
        print(f"🔧 Augmentation phase: Adding {remaining_needed} additional documents to reach 10k total")
        print(f"   🔍 Using query: {AUGMENT_QUERY}")
        print(f"   📅 Year range: {YEAR_FROM}-{YEAR_TO}")
        print()
        
        written = augment_corpus(
            out_root=out_root,
            existing_pmids=all_pmids,
            target_new=remaining_needed,
            query=AUGMENT_QUERY,
            year_from=YEAR_FROM,
            year_to=YEAR_TO,
        )
        print(f"✅ Augmentation complete!")
        print(f"   📊 Added {written} extra PubMed documents with empty QA lists")
        print(f"   📊 Final dataset size: {len(all_pmids) + written} documents")
    else:
        print(f"ℹ️  No augmentation needed - already have {len(all_pmids)} documents")
    
    print()
    print("🎉 PubMedQA_10k dataset creation complete!")
    print(f"📁 Output directory: {out_root}")
    print(f"📊 Final statistics:")
    print(f"   - Base questions: {len(pmids)}")
    print(f"   - Distractor documents: {len(all_pmids) - len(pmids)}")
    print(f"   - Augmented documents: {written}")
    print(f"   - Total documents: {len(all_pmids) + written}")


if __name__ == "__main__":
    main()
