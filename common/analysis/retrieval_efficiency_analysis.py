from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import argparse

# Add the project root to Python path so it work from where ever you are
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from log_analyzer import LogAnalyzer


# Retrieval efficiency metrics - chunk retrieval phase only
RETRIEVAL_EFFICIENCY_METRICS: list[tuple[str, callable[[LogAnalyzer], float]]] = [
    ("Retr. Time", lambda a: a.chunk_retrieval_time_avg()),
    ("In Tokens", lambda a: a.avg_input_tokens_retrieval()),
    ("Out Tokens", lambda a: a.avg_output_tokens_retrieval()),
    ("LLM Calls", lambda a: a.llm_calls_retrieval()),
    ("LLM Time", lambda a: a.avg_llm_time_retrieval()),
    ("Questions", lambda a: float(a.num_questions())),
]

# Generation efficiency metrics - answer generation phase only
GENERATION_EFFICIENCY_METRICS: list[tuple[str, callable[[LogAnalyzer], float]]] = [
    ("Gen. Time", lambda a: a.generation_time_avg()),
    ("In Tokens", lambda a: a.avg_input_tokens_generation()),
    ("Out Tokens", lambda a: a.avg_output_tokens_generation()),
    ("LLM Calls", lambda a: a.llm_calls_generation()),
    ("LLM Time", lambda a: a.avg_llm_time_generation()),
    ("Questions", lambda a: float(a.num_questions())),
]

# Column descriptions for retrieval efficiency metrics
RETRIEVAL_COLUMN_DESCRIPTIONS = {
    "Retr. Time": "Average time (ms) to retrieve relevant chunks for each question",
    "In Tokens": "Average input tokens consumed per LLM call during chunk retrieval",
    "Out Tokens": "Average output tokens generated per LLM call during chunk retrieval",
    "LLM Calls": "Average number of LLM calls made per question during chunk retrieval",
    "LLM Time": "Average time (ms) per LLM call during chunk retrieval",
    "Questions": "Total number of questions that were answered"
}

# Column descriptions for generation efficiency metrics
GENERATION_COLUMN_DESCRIPTIONS = {
    "Gen. Time": "Average time (ms) to generate the final answer for each question",
    "In Tokens": "Average input tokens consumed per LLM call during answer generation",
    "Out Tokens": "Average output tokens generated per LLM call during answer generation",
    "LLM Calls": "Average number of LLM calls made per question during answer generation",
    "LLM Time": "Average time (ms) per LLM call during answer generation",
    "Questions": "Total number of questions that were answered"
}

# ---------------------------------------------------------------------------
# Build retrieval efficiency table
# ---------------------------------------------------------------------------
def build_retrieval_efficiency_table(csv_paths: list[Path]) -> pd.DataFrame:
    """Build a table with retrieval efficiency metrics (time and token usage during chunk retrieval)."""
    rows: list[dict[str, float | str]] = []
    for path in csv_paths:
        analyzer = LogAnalyzer(csv_file=path)
        row: dict[str, float | str] = {"Approach": path.stem}
        for col, func in RETRIEVAL_EFFICIENCY_METRICS:
            row[col] = func(analyzer)
        rows.append(row)
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Build generation efficiency table
# ---------------------------------------------------------------------------
def build_generation_efficiency_table(csv_paths: list[Path]) -> pd.DataFrame:
    """Build a table with generation efficiency metrics (time and token usage during answer generation)."""
    rows: list[dict[str, float | str]] = []
    for path in csv_paths:
        analyzer = LogAnalyzer(csv_file=path)
        row: dict[str, float | str] = {"Approach": path.stem}
        for col, func in GENERATION_EFFICIENCY_METRICS:
            row[col] = func(analyzer)
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
    
    df = build_retrieval_efficiency_table(CSV_PATHS)
    print("Retrieval Efficiency Analysis Results:")
    print("=" * 60)
    print_column_descriptions(RETRIEVAL_COLUMN_DESCRIPTIONS)
    print(df.to_markdown(index=False, floatfmt=".2f"))
