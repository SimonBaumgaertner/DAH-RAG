from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import argparse

# Add the project root to Python path so it work from where ever you are
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from log_analyzer import LogAnalyzer


def get_document_indexing_times_with_order(csv_file: Path) -> List[Tuple[str, float]]:
    """Extract indexing times per document in chronological order from a CSV file.
    
    Args:
        csv_file: Path to the CSV file
        
    Returns:
        List of tuples (document_id, indexing_time) in chronological order
    """
    analyzer = LogAnalyzer(csv_file=csv_file)
    
    # Get all document indexing track entries
    doc_indexing_entries = analyzer._by_type.get("document_indexing_track", [])
    
    # Sort by timestamp to get chronological order
    from datetime import datetime
    from typing import List, Tuple
    
    # Parse timestamps and sort
    entries_with_times = []
    for entry in doc_indexing_entries:
        try:
            timestamp = datetime.fromisoformat(entry["timestamp"])
            doc_id = entry["identifier"]
            indexing_time = float(entry["value"])
            entries_with_times.append((timestamp, doc_id, indexing_time))
        except (ValueError, KeyError) as e:
            print(f"Warning: Could not parse entry: {e}")
            continue
    
    # Sort by timestamp (chronological order)
    entries_with_times.sort(key=lambda x: x[0])
    
    # Return list of (doc_id, indexing_time) in chronological order
    return [(doc_id, indexing_time) for _, doc_id, indexing_time in entries_with_times]


def create_indexing_time_plot(csv_paths: List[Path], output_file: str | None = None) -> None:
    """Create a plot showing indexing times per document in chronological order.
    
    Args:
        csv_paths: List of CSV file paths to analyze
        output_file: Optional output file path for the plot
    """
    plt.figure(figsize=(15, 10))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, csv_path in enumerate(csv_paths):
        # Get document times in chronological order
        doc_times_ordered = get_document_indexing_times_with_order(csv_path)
        
        if not doc_times_ordered:
            print(f"Warning: No document indexing data found in {csv_path.name}")
            continue
            
        # Extract just the indexing times in chronological order
        indexing_times = [time for _, time in doc_times_ordered]
        doc_indices = range(len(indexing_times))
        
        # Use full filename as approach name
        approach_name = csv_path.stem
        
        plt.plot(doc_indices, indexing_times, 
                marker='o', markersize=3, linewidth=1.5, 
                color=colors[i % len(colors)], 
                label=approach_name, alpha=0.7)
    
    plt.xlabel('Document Index (chronological order)', fontsize=12)
    plt.ylabel('Indexing Time (ms)', fontsize=12)
    plt.title('Document Indexing Times in Chronological Order', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    
    plt.show()


def create_cumulative_indexing_plot(csv_paths: List[Path], output_file: str | None = None) -> None:
    """Create a plot showing cumulative indexing time as documents are processed chronologically.
    
    Args:
        csv_paths: List of CSV file paths to analyze
        output_file: Optional output file path for the plot
    """
    plt.figure(figsize=(15, 10))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, csv_path in enumerate(csv_paths):
        # Get document times in chronological order
        doc_times_ordered = get_document_indexing_times_with_order(csv_path)
        
        if not doc_times_ordered:
            print(f"Warning: No document indexing data found in {csv_path.name}")
            continue
            
        # Extract just the indexing times in chronological order
        indexing_times = [time for _, time in doc_times_ordered]
        cumulative_times = np.cumsum(indexing_times)
        doc_indices = range(len(cumulative_times))
        
        # Use full filename as approach name
        approach_name = csv_path.stem
        
        plt.plot(doc_indices, cumulative_times, 
                marker='o', markersize=3, linewidth=2, 
                color=colors[i % len(colors)], 
                label=approach_name, alpha=0.8)
    
    plt.xlabel('Number of Documents Processed (chronological)', fontsize=12)
    plt.ylabel('Cumulative Indexing Time (ms)', fontsize=12)
    plt.title('Cumulative Indexing Time as Documents are Processed', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Cumulative plot saved to: {output_file}")
    
    plt.show()


def create_indexing_time_statistics_table(csv_paths: List[Path]) -> pd.DataFrame:
    """Create a statistics table for indexing times across approaches.
    
    Args:
        csv_paths: List of CSV file paths to analyze
        
    Returns:
        DataFrame with statistics for each approach
    """
    rows: List[Dict[str, float | str]] = []
    
    for csv_path in csv_paths:
        doc_times_ordered = get_document_indexing_times_with_order(csv_path)
        
        if not doc_times_ordered:
            print(f"Warning: No document indexing data found in {csv_path.name}")
            continue
            
        times = [time for _, time in doc_times_ordered]
        
        # Use full filename as approach name
        approach_name = csv_path.stem
        
        row = {
            "Approach": approach_name,
            "Documents": len(times),
            "Min Time (ms)": min(times),
            "Max Time (ms)": max(times),
            "Mean Time (ms)": np.mean(times),
            "Median Time (ms)": np.median(times),
            "Std Dev (ms)": np.std(times),
            "Total Time (ms)": sum(times),
            "Total Time (s)": sum(times) / 1000,
            "Total Time (min)": sum(times) / 60000
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def run_advanced_indexing_analysis(csv_paths: List[Path], plot_type: str = "individual") -> None:
    """Run advanced indexing analysis with plots and statistics.
    
    Args:
        csv_paths: List of CSV file paths to analyze
        plot_type: Type of plot to create ("individual" or "cumulative")
    """
    print("Advanced Indexing Efficiency Analysis")
    print("=" * 60)
    
    # Create statistics table
    stats_df = create_indexing_time_statistics_table(csv_paths)
    print("\nIndexing Time Statistics:")
    print(stats_df.to_markdown(index=False, floatfmt=".2f"))
    
    # Filter to get only naive approaches
    naive_csv_paths = [p for p in csv_paths if "NaiveVectorDB" in p.stem]
    
    if plot_type == "individual":
        print("\nCreating individual indexing time plot for all approaches...")
        create_indexing_time_plot(csv_paths, "indexing_times_per_document_all.png")
        
        if naive_csv_paths:
            print("\nCreating individual indexing time plot for naive approaches only...")
            create_indexing_time_plot(naive_csv_paths, "indexing_times_per_document_naive.png")
    
    elif plot_type == "cumulative":
        print("\nCreating cumulative indexing time plot for all approaches...")
        create_cumulative_indexing_plot(csv_paths, "cumulative_indexing_times_all.png")
        
        if naive_csv_paths:
            print("\nCreating cumulative indexing time plot for naive approaches only...")
            create_cumulative_indexing_plot(naive_csv_paths, "cumulative_indexing_times_naive.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced indexing efficiency analysis")
    parser.add_argument(
        "--plot-type", 
        choices=["individual", "cumulative"], 
        default="individual",
        help="Type of plots to create: individual (per document in chronological order) or cumulative (cumulative time)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save output plots (default: current directory)"
    )
    
    args = parser.parse_args()
    
    # Define the root path for analysis data when running standalone
    ROOT = Path(__file__).resolve().parents[2] / "logs_and_tracks/prime_studies"
    
    # Filter for the specific datasets mentioned by the user
    target_files = [
        "HotpotQA_1k-DocAwareHybridRAG.csv",
    ]
    
    csv_paths = []
    for filename in target_files:
        file_path = ROOT / filename
        if file_path.exists():
            csv_paths.append(file_path)
        else:
            print(f"Warning: File not found: {file_path}")
    
    if not csv_paths:
        print("No valid CSV files found!")
        sys.exit(1)
    
    run_advanced_indexing_analysis(csv_paths, args.plot_type)
