from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import argparse

# Add the project root to Python path so it work from where ever you are
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from log_analyzer import LogAnalyzer


# Performance metrics (accuracy, recall, faithfulness, ROUGE-L)
PERFORMANCE_METRICS: list[tuple[str, callable[[LogAnalyzer], float]]] = [
    ("QA Accuracy (%)", lambda a: a.qa_accuracy()),
    ("ROUGE-L Avg", lambda a: a.rouge_l_avg()),
    ("R@1 (%)", lambda a: a.recall_at_k(1)),
    ("R@2 (%)", lambda a: a.recall_at_k(2)),
    ("R@5 (%)", lambda a: a.recall_at_k(5)),
    ("R@10 (%)", lambda a: a.recall_at_k(10)),
    ("R@25 (%)", lambda a: a.recall_at_k(25)),

    # ("Faithfulness@5 (%)", lambda a: a.faithfulness(5)),
    # ("Unsupported Acc@5 (%)", lambda a: a.unsupported_accuracy(5)),
]

# Column descriptions for performance metrics
PERFORMANCE_COLUMN_DESCRIPTIONS = {
    "QA Accuracy (%)": "Percentage of questions answered correctly",
    "ROUGE-L Avg": "Average ROUGE-L F1 score measuring answer quality (0-1 scale)",
    "R@k (%)": "Recall@k: Average percentage of proofs found in the top k retrieved chunks per question (macro-averaged, so each question has equal weight)",
}

# ---------------------------------------------------------------------------
# Build performance table
# ---------------------------------------------------------------------------
def build_performance_table(csv_paths: list[Path]) -> pd.DataFrame:
    """Build a table with performance metrics (accuracy, recall, faithfulness)."""
    rows: list[dict[str, float | str]] = []
    for path in csv_paths:
        analyzer = LogAnalyzer(csv_file=path)
        row: dict[str, float | str] = {"Approach": path.stem}
        for col, func in PERFORMANCE_METRICS:
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
    
    df = build_performance_table(CSV_PATHS)
    print("Performance Analysis Results:")
    print("=" * 50)
    print_column_descriptions(PERFORMANCE_COLUMN_DESCRIPTIONS)
    print(df.to_markdown(index=False, floatfmt=".3f"))
