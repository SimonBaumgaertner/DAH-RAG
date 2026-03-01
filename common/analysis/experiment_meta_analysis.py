from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional
import pandas as pd
import argparse

# Add the project root to Python path so it work from where ever you are
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from log_analyzer import LogAnalyzer
from common.data_classes.evaluation import EntryType
from analysis_utils import format_duration, format_tokens, format_cost






def calculate_cost(input_tokens: float, output_tokens: float, actual_cost: float = 0.0) -> float:
    """Calculate cost, using actual cost if available, otherwise fallback to approximate.
    
    Args:
        input_tokens: Total number of input tokens
        output_tokens: Total number of output tokens
        actual_cost: Logged actual cost from LLM runner
        
    Returns:
        Total cost in dollars
    """
    if actual_cost > 0:
        return actual_cost
        
    # Fallback to old Llama 3.3 70B pricing if no actual cost was logged
    # (per 1M tokens)
    INPUT_TOKEN_COST_PER_MILLION = 0.25
    OUTPUT_TOKEN_COST_PER_MILLION = 0.50
    
    input_cost = (input_tokens / 1_000_000) * INPUT_TOKEN_COST_PER_MILLION
    output_cost = (output_tokens / 1_000_000) * OUTPUT_TOKEN_COST_PER_MILLION
    return input_cost + output_cost




def get_csv_duration(csv_path: Path) -> Optional[float]:
    """Get duration in seconds for a single CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Duration in seconds, or None if unable to calculate
    """
    try:
        df = pd.read_csv(csv_path)
        
        if "timestamp" not in df.columns:
            return None
        
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        valid_timestamps = df["timestamp"].dropna()
        
        if len(valid_timestamps) == 0:
            print(f"DEBUG: No valid timestamps found in {csv_path.name}")
            return None
        
        earliest = valid_timestamps.min()
        latest = valid_timestamps.max()
        duration_seconds = (latest - earliest).total_seconds()
        
        return duration_seconds
    except Exception as e:
        print(f"DEBUG: Error calculating duration for {csv_path.name}: {e}")
        return None


def get_total_tokens_indexing(analyzer: LogAnalyzer) -> tuple[float, float]:
    """Get total input and output tokens for indexing phase (including tree building, etc.).
    
    Args:
        analyzer: LogAnalyzer instance
        
    Returns:
        Tuple of (total_input_tokens, total_output_tokens)
    """
    # Count all indexing tokens, not just those with document IDs
    # This includes tree-building calls (e.g., RaptorRAG summarization)
    input_tokens = analyzer._values(EntryType.LLM_INDEXING_INPUT_TOKENS_TRACK)
    output_tokens = analyzer._values(EntryType.LLM_INDEXING_OUTPUT_TOKENS_TRACK)
    
    total_input = sum(input_tokens) if input_tokens else 0.0
    total_output = sum(output_tokens) if output_tokens else 0.0
    
    return total_input, total_output


def get_total_tokens_retrieval(analyzer: LogAnalyzer) -> tuple[float, float]:
    """Get total input and output tokens for retrieval phase.
    
    Args:
        analyzer: LogAnalyzer instance
        
    Returns:
        Tuple of (total_input_tokens, total_output_tokens)
    """
    q_ids = [r["identifier"] for r in analyzer._by_type.get(EntryType.ANSWER_TRACK.value, [])]
    input_tokens_1 = analyzer._values(EntryType.LLM_RETRIEVAL_INPUT_TOKENS_TRACK, ids=q_ids)
    input_tokens_2 = analyzer._values("prompt_length", ids=q_ids)
    output_tokens = analyzer._values(EntryType.LLM_RETRIEVAL_OUTPUT_TOKENS_TRACK, ids=q_ids)
    
    total_input = sum(input_tokens_1) + sum(input_tokens_2) if (input_tokens_1 or input_tokens_2) else 0.0
    total_output = sum(output_tokens) if output_tokens else 0.0
    
    return total_input, total_output


def get_total_tokens_generation(analyzer: LogAnalyzer) -> tuple[float, float]:
    """Get total input and output tokens for generation phase.
    
    Args:
        analyzer: LogAnalyzer instance
        
    Returns:
        Tuple of (total_input_tokens, total_output_tokens)
    """
    q_ids = [r["identifier"] for r in analyzer._by_type.get(EntryType.ANSWER_TRACK.value, [])]
    input_tokens = analyzer._values(EntryType.LLM_QA_INPUT_TOKENS_TRACK, ids=q_ids)
    output_tokens = analyzer._values(EntryType.LLM_QA_OUTPUT_TOKENS_TRACK, ids=q_ids)
    
    total_input = sum(input_tokens) if input_tokens else 0.0
    total_output = sum(output_tokens) if output_tokens else 0.0
    
    return total_input, total_output


def get_actual_cost_indexing(analyzer: LogAnalyzer) -> float:
    """Get total actual cost for indexing phase.
    
    Args:
        analyzer: LogAnalyzer instance
        
    Returns:
        Total cost in dollars
    """
    costs = analyzer._values(EntryType.LLM_INDEXING_COST_TRACK)
    return sum(costs) if costs else 0.0


def get_actual_cost_generation(analyzer: LogAnalyzer) -> float:
    """Get total actual cost for generation phase.
    
    Args:
        analyzer: LogAnalyzer instance
        
    Returns:
        Total cost in dollars
    """
    q_ids = [r["identifier"] for r in analyzer._by_type.get(EntryType.ANSWER_TRACK.value, [])]
    costs = analyzer._values(EntryType.LLM_QA_COST_TRACK, ids=q_ids)
    return sum(costs) if costs else 0.0


def build_experiment_metadata_table(csv_paths: list[Path]) -> pd.DataFrame:
    """Build a table with experiment metadata including duration and token usage.
    
    Args:
        csv_paths: List of paths to CSV files
        
    Returns:
        DataFrame with columns: Name, Duration, Index In/Out, Retrieval In/Out, Generation In/Out, Total In/Out, Approximate Cost
    """
    rows: list[dict[str, str]] = []
    
    for csv_path in csv_paths:
        try:
            # print(f"DEBUG: Processing {csv_path.name}...")
            analyzer = LogAnalyzer(csv_file=csv_path)
            duration_seconds = get_csv_duration(csv_path)
            
            if duration_seconds is None:
                # print(f"DEBUG: Skipping {csv_path.name} due to missing duration")
                continue
            
            duration_formatted = format_duration(duration_seconds)
            
            idx_in, idx_out = get_total_tokens_indexing(analyzer)
            ret_in, ret_out = get_total_tokens_retrieval(analyzer)
            gen_in, gen_out = get_total_tokens_generation(analyzer)
            
            idx_cost = get_actual_cost_indexing(analyzer)
            gen_cost = get_actual_cost_generation(analyzer)
            actual_total_cost = idx_cost + gen_cost
            
            total_in = idx_in + ret_in + gen_in
            total_out = idx_out + ret_out + gen_out
            cost = calculate_cost(total_in, total_out, actual_total_cost)
            
            rows.append({
                "Name": csv_path.stem,
                "Duration": duration_formatted,
                "Index In/Out": f"{format_tokens(idx_in)} / {format_tokens(idx_out)}",
                "Retr. In/Out": f"{format_tokens(ret_in)} / {format_tokens(ret_out)}",
                "Gen. In/Out": f"{format_tokens(gen_in)} / {format_tokens(gen_out)}",
                "Total In/Out": f"{format_tokens(total_in)} / {format_tokens(total_out)}",
                "Total Cost": format_cost(cost)
            })
        except Exception as e:
            print(f"DEBUG: Error processing {csv_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return pd.DataFrame(rows)


def analyze_experiment_metadata(csv_paths: list[Path]) -> None:
    """Analyze experiment metadata and print table.
    
    Args:
        csv_paths: List of paths to CSV files.
    """
    if not csv_paths:
        return
    
    df = build_experiment_metadata_table(csv_paths)
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze experiment metadata")
    args = parser.parse_args()
    
    # Define the root path for analysis data when running standalone
    ROOT = Path(__file__).resolve().parents[2] / "logs_and_tracks/prime_studies"
    CSV_PATHS = sorted(ROOT.glob("*.csv"))
    
    analyze_experiment_metadata(CSV_PATHS)

