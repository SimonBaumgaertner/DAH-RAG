from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import argparse

# Add the project root to Python path so it work from where ever you are
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from performance_analysis import build_performance_table, PERFORMANCE_COLUMN_DESCRIPTIONS, print_column_descriptions as print_perf_descriptions
from indexing_efficiency_analysis import build_indexing_efficiency_table, INDEXING_COLUMN_DESCRIPTIONS, print_column_descriptions
from retrieval_efficiency_analysis import (
    build_retrieval_efficiency_table, 
    RETRIEVAL_COLUMN_DESCRIPTIONS,
    build_generation_efficiency_table,
    GENERATION_COLUMN_DESCRIPTIONS
)
from experiment_meta_analysis import analyze_experiment_metadata

# Define the root path for analysis data
ROOT = Path(__file__).resolve().parents[2] / "logs_and_tracks/prime_studies"

# Find all CSV files under ROOT
CSV_PATHS = sorted(ROOT.glob("*.csv"))

# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------
def run_analysis(analysis_type: str = "all") -> None:
    """Run analysis based on the specified type.
    
    Args:
        analysis_type: "performance", "indexing", "retrieval", "generation", "efficiency", or "all"
    """
    if analysis_type in ["performance", "all"]:
        print("Performance Analysis Results:")
        print("=" * 60)
        print_perf_descriptions(PERFORMANCE_COLUMN_DESCRIPTIONS)
        df_perf = build_performance_table(CSV_PATHS)
        print(df_perf.to_markdown(index=False, floatfmt=".3f"))
        print("\n")
    
    if analysis_type in ["indexing", "efficiency", "all"]:
        print("Indexing Efficiency Analysis Results:")
        print("=" * 60)
        print_column_descriptions(INDEXING_COLUMN_DESCRIPTIONS)
        df_idx = build_indexing_efficiency_table(CSV_PATHS)
        print(df_idx.to_markdown(index=False, floatfmt=".3f"))
        print("\n")
    
    if analysis_type in ["retrieval", "efficiency", "all"]:
        print("Retrieval Efficiency Analysis Results:")
        print("=" * 60)
        print_column_descriptions(RETRIEVAL_COLUMN_DESCRIPTIONS)
        df_ret = build_retrieval_efficiency_table(CSV_PATHS)
        print(df_ret.to_markdown(index=False, floatfmt=".3f"))
        print("\n")
    
    if analysis_type in ["generation", "efficiency", "all"]:
        print("Generation Efficiency Analysis Results:")
        print("=" * 60)
        print_column_descriptions(GENERATION_COLUMN_DESCRIPTIONS)
        df_gen = build_generation_efficiency_table(CSV_PATHS)
        print(df_gen.to_markdown(index=False, floatfmt=".3f"))
        print("\n")
    
    if analysis_type == "all":
        print("Experiment Metadata Analysis Results:")
        print("=" * 60)
        analyze_experiment_metadata(CSV_PATHS)
        print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument(
        "--type", 
        choices=["performance", "indexing", "retrieval", "generation", "efficiency", "all"], 
        default="all",
        help="Type of analysis to run: performance (accuracy/recall), indexing (indexing efficiency), retrieval (retrieval efficiency), generation (generation efficiency), efficiency (indexing, retrieval, and generation), or all (default: all)"
    )
    
    args = parser.parse_args()
    run_analysis(args.type)
