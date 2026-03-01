# QA Parser for HotpotQA Datasets

This directory contains scripts to fix the QA.json files in HotpotQA datasets by extracting the correct supporting facts from the `hotpot_dev_distractor_v1.json` file.

## Problem Solved

The original `hotpotQA_parser.py` was extracting document titles instead of the actual supporting sentences because it was trying to get the proofs from the `context` field in the wrong JSON file. The `hotpot_dev_distractor_v1.json` file contains the actual supporting facts in its `context` field.

## Scripts

### `qa_parser.py`

Main script that processes a single dataset directory.

**Usage:**
```bash
python3 qa_parser.py <dataset_dir>
```

**Example:**
```bash
python3 qa_parser.py ../../data/HotpotQA_100
```

**What it does:**
1. Loads the `hotpot_dev_distractor_v1.json` file
2. Loads the existing `QA.json` from the dataset directory
3. Matches questions by question_id
4. Extracts the correct supporting facts from the distractor data
5. Cleans any links in the extracted text
6. Updates the `QA.json` file with the corrected proofs

### `process_all_datasets.py`

Batch script to process all HotpotQA datasets at once.

**Usage:**
```bash
python3 process_all_datasets.py
```

**What it does:**
1. Finds all directories starting with "HotpotQA_" in `../../data/`
2. Processes each dataset using `qa_parser.py`
3. Reports the results for each dataset

## Before and After

**Before (incorrect):**
```json
{
  "question_id": "5a8b57f25542995d1e6f1371",
  "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
  "correct_answer": "yes",
  "proofs": [
    {
      "document_id": "Scott_Derrickson",
      "context": "Scott Derrickson"
    },
    {
      "document_id": "Ed_Wood", 
      "context": "Ed Wood"
    }
  ]
}
```

**After (correct):**
```json
{
  "question_id": "5a8b57f25542995d1e6f1371",
  "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
  "correct_answer": "yes",
  "proofs": [
    {
      "document_id": "Scott Derrickson",
      "context": "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer."
    },
    {
      "document_id": "Ed Wood",
      "context": "Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker, actor, writer, producer, and director."
    }
  ]
}
```

## Requirements

- Python 3.6+
- The `hotpot_dev_distractor_v1.json` file must be in the same directory as the scripts
- HotpotQA dataset directories must contain a `QA.json` file

## Files Modified

The scripts will update the `QA.json` file in each dataset directory with the corrected proofs. The original file is overwritten, so make sure to backup if needed.
