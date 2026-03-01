from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import argparse

# Add the project root to Python path so it work from where ever you are
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from log_analyzer import LogAnalyzer
from analysis_utils import format_duration


# Indexing efficiency metrics - all related to document indexing phase
INDEXING_EFFICIENCY_METRICS: list[tuple[str, callable[[LogAnalyzer], float]]] = [
    ("Index Time", lambda a: a.total_indexing_time()),
    #("Avg Doc Time", lambda a: a.indexing_time_avg()),
    ("In Tokens", lambda a: a.avg_input_tokens_indexing()),
    ("Out Tokens", lambda a: a.avg_output_tokens_indexing()),
    ("LLM Calls", lambda a: float(a.llm_calls_indexing())),
    ("LLM Time", lambda a: a.avg_llm_time_indexing()),
    ("Documents", lambda a: float(a.num_documents())),
    ("Chunks", lambda a: float(a.num_chunks())),
]

# Column descriptions for indexing efficiency metrics
INDEXING_COLUMN_DESCRIPTIONS = {
    "Index Time": "Total time for complete indexing including finalization (e.g., tree building)",
    #"Avg Doc Time": "Average time (ms) to process each document during indexing (excluding finalization)",
    "In Tokens": "Average input tokens consumed per LLM call during indexing",
    "Out Tokens": "Average output tokens generated per LLM call during indexing", 
    "LLM Calls": "Total number of LLM calls made during the indexing phase",
    "LLM Time": "Average time (ms) per LLM call during indexing",
    "Documents": "Total number of documents that were indexed",
    "Chunks": "Total number of chunks that were indexed"
}

# ---------------------------------------------------------------------------
# Build indexing efficiency table
# ---------------------------------------------------------------------------
def build_indexing_efficiency_table(csv_paths: list[Path]) -> pd.DataFrame:
    """Build a table with indexing efficiency metrics (time and token usage during document indexing)."""
    rows: list[dict[str, float | str]] = []
    for path in csv_paths:
        analyzer = LogAnalyzer(csv_file=path)
        row: dict[str, float | str] = {"Approach": path.stem}
        for col, func in INDEXING_EFFICIENCY_METRICS:
            val = func(analyzer)
            if col == "Index Time":
                # Convert ms to seconds for format_duration
                row[col] = format_duration(val / 1000.0)
            else:
                row[col] = val
        rows.append(row)
    return pd.DataFrame(rows)

def print_column_descriptions(descriptions: dict[str, str]) -> None:
    """Print column descriptions in a formatted way."""
    print("Column Descriptions:")
    for col, desc in descriptions.items():
        print(f"  • {col}: {desc}")
    print()

if __name__ == "__main__":
    # Define the root path for analysis data when running standalone
    ROOT = Path(__file__).resolve().parents[2] / "logs_and_tracks 🗒️/archived_experiment_data"
    CSV_PATHS = sorted(ROOT.glob("*.csv"))
    
    df = build_indexing_efficiency_table(CSV_PATHS)
    print("Indexing Efficiency Analysis Results:")
    print("=" * 60)
    print_column_descriptions(INDEXING_COLUMN_DESCRIPTIONS)
    print(df.to_markdown(index=False, floatfmt=".2f"))
