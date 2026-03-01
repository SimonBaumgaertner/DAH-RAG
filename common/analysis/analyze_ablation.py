from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import glob

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Define the root path for ablation study data
ROOT = Path(__file__).resolve().parents[2] / "logs_and_tracks/ablation_studies"

# Embedding models to analyze
EMBEDDING_MODELS = {
    "Jina_v3": "Jina_v3",
    "Qwen4": "Qwen_4B",
    "Qwen8": "Qwen_8B",
    "Gemma3": "Gemma3"
}

# Retrieval component configurations
ALPHA_CONFIGS = ["alpha_dense", "alpha_lexical", "alpha_entity"]
BETA_CONFIGS = ["beta_dense", "beta_lexical", "beta_ppr"]

# Datasets
DATASETS = ["HotpotQA", "MultihopRAG"]

# RAG versions
RAG_VERSIONS = ["DAHR", "Vector"]


def calculate_recall_at_k(csv_path: Path, k: int = 5) -> float:
    """
    Calculate Recall@k from a CSV file containing proof_track entries.
    
    Args:
        csv_path: Path to the CSV file
        k: The k value for Recall@k (default: 5)
    
    Returns:
        Recall@k as a percentage, or NaN if file doesn't exist or has no data
    """
    if not csv_path.exists():
        return np.nan
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Failed to read {csv_path.name}: {e}")
        return np.nan
    
    # Filter for proof_track entries
    proof_df = df[df['entry_type'] == 'proof_track']
    
    if proof_df.empty:
        return np.nan
    
    # Group by question (identifier) and collect ranks
    ranks = {}  # question_id -> list of ranks
    for _, row in proof_df.iterrows():
        qid = row['identifier']
        try:
            rank = int(float(row['value']))
        except (ValueError, TypeError):
            continue
        
        if qid not in ranks:
            ranks[qid] = []
        ranks[qid].append(rank)
    
    if not ranks:
        return np.nan
    
    # Calculate recall for each question
    question_recalls = []
    for q_ranks in ranks.values():
        total_proofs = len(q_ranks)
        if total_proofs == 0:
            continue
        # Count how many proofs were found in top k (rank != -1 and 0 <= rank < k)
        found = sum(1 for r in q_ranks if r != -1 and 0 <= r < k)
        question_recalls.append(found / total_proofs)
    
    if not question_recalls:
        return np.nan
    
    # Return average recall as percentage
    return 100.0 * (sum(question_recalls) / len(question_recalls))


# ============================================================================
# EMBEDDING MODEL ANALYSIS
# ============================================================================

def build_embedding_comparison_table(dataset: str) -> pd.DataFrame:
    """
    Build a comparison table for embedding models on a specific dataset.
    
    Args:
        dataset: Dataset name (HotpotQA or MultihopRAG)
    
    Returns:
        DataFrame with RAG versions as rows and embedding models as columns
    """
    results = {}
    
    for model_name, model_file_pattern in EMBEDDING_MODELS.items():
        results[model_name] = {}
        
        for rag_version in RAG_VERSIONS:
            # Construct the expected filename
            filename = f"{dataset}-{rag_version}-{model_file_pattern}.csv"
            csv_path = ROOT / filename
            
            # Calculate Recall@5
            recall = calculate_recall_at_k(csv_path, k=5)
            results[model_name][rag_version] = recall
    
    # Create DataFrame and transpose so models are columns
    df = pd.DataFrame(results)
    df.index.name = "RAG Version"
    
    return df


def find_data_path(dataset: str, rag_version: str, config_name: str) -> Path:
    """
    Find the CSV path, handling potential casing inconsistencies for MultihopRAG
    and special file naming patterns (e.g. no_graph).
    """
    # Standard pattern: Dataset-RAG-Config.csv
    filename = f"{dataset}-{rag_version}-{config_name}.csv"
    path = ROOT / filename
    if path.exists():
        return path
        
    # Pattern for no_graph: Dataset-RAG_no_graph.csv
    if config_name == "no_graph":
         alt_filename = f"{dataset}-{rag_version}_{config_name}.csv"
         alt_path = ROOT / alt_filename
         if alt_path.exists():
             return alt_path

    # Try alternative casing for dataset
    if dataset == "MultihopRAG":
        # Check standard pattern with MultiHopRAG
        alt_path = ROOT / f"MultiHopRAG-{rag_version}-{config_name}.csv"
        if alt_path.exists():
            return alt_path
        
        # Check no_graph pattern with MultiHopRAG
        if config_name == "no_graph":
            alt_path = ROOT / f"MultiHopRAG-{rag_version}_{config_name}.csv"
            if alt_path.exists():
                return alt_path
                
    elif dataset == "MultiHopRAG":
        # Check standard pattern with MultihopRAG
        alt_path = ROOT / f"MultihopRAG-{rag_version}-{config_name}.csv"
        if alt_path.exists():
            return alt_path

        # Check no_graph pattern with MultihopRAG
        if config_name == "no_graph":
            alt_path = ROOT / f"MultihopRAG-{rag_version}_{config_name}.csv"
            if alt_path.exists():
                return alt_path
            
    return path


def analyze_embeddings(dataset: str):
    """
    Generate embedding comparison table for specific dataset.
    """
    df = build_embedding_comparison_table(dataset)
    print_table(df, f"{dataset} - Recall@5 (%)")

def build_alpha_table(dataset: str) -> pd.DataFrame:
    """
    Build comparison table for Alpha retrieval configurations.
    Columns: Baseline, alpha_dense, alpha_entity, alpha_lexical, alpha_cohesion
    """
    results = {}
    
    # Add baseline first
    baseline_path = find_data_path(dataset, "DAHR", "Qwen_4B")
    results["Baseline"] = calculate_recall_at_k(baseline_path, k=5)
    
    # Build results for alpha configs
    for config in ALPHA_CONFIGS:
        csv_path = find_data_path(dataset, "DAHR", config)
        results[config] = calculate_recall_at_k(csv_path, k=5)
    
    # Create DataFrame
    df = pd.DataFrame([results])
    df.index = [dataset]
    df.index.name = "Dataset"
    return df


def build_beta_table(dataset: str) -> pd.DataFrame:
    """
    Build comparison table for Beta retrieval configurations.
    Columns: Baseline, beta_PPR, beta_dense, beta_lexical, no_graph
    """
    results = {}
    
    # Add baseline first
    baseline_path = find_data_path(dataset, "DAHR", "Qwen_4B")
    results["Baseline"] = calculate_recall_at_k(baseline_path, k=5)
    
    # Build results for beta configs (includes no_graph)
    for config in BETA_CONFIGS:
        csv_path = find_data_path(dataset, "DAHR", config)
        results[config] = calculate_recall_at_k(csv_path, k=5)
    
    # Create DataFrame
    df = pd.DataFrame([results])
    df.index = [dataset]
    df.index.name = "Dataset"
    return df


def analyze_retrieval_components(dataset: str):
    """
    Analyze retrieval components for specific dataset.
    """
    # Alpha Table
    df_alpha = build_alpha_table(dataset)
    print_table(df_alpha, f"{dataset} - Alpha Retrieval Configs - Recall@5 (%)")
    
    # Beta Table
    df_beta = build_beta_table(dataset)
    print_table(df_beta, f"{dataset} - Beta Retrieval Configs - Recall@5 (%)")


# ============================================================================
# OTHER ABLATION STUDIES (AUTO-DETECTION)
# ============================================================================

def is_known_config(filename: str) -> bool:
    """
    Check if a filename corresponds to a known configuration already analyzed.
    
    Args:
        filename: The CSV filename
    
    Returns:
        True if this is a known configuration, False otherwise
    """
    # Check for embedding models
    for model in EMBEDDING_MODELS.values():
        if model in filename:
            return True
    
    # Check for alpha/beta configs
    for config in ALPHA_CONFIGS + BETA_CONFIGS:
        if config in filename:
            return True
    
    return False


def discover_other_configs() -> list[tuple[str, str, str, str]]:
    """
    Discover all other ablation study configurations not yet analyzed.
    
    Returns:
        List of tuples (dataset, rag_version, config_name, filepath)
    """
    csv_files = glob.glob(str(ROOT / "*.csv"))
    other_configs = []
    
    for csv_path in csv_files:
        filename = Path(csv_path).name
        
        # Skip known configurations
        if is_known_config(filename):
            continue
        
        # Parse filename: Dataset-RAGVersion-Config.csv
        parts = filename.replace(".csv", "").split("-")
        if len(parts) >= 3:
            dataset = parts[0]
            rag_version = parts[1]
            config_name = "-".join(parts[2:])  # Handle multi-part config names
            
            other_configs.append((dataset, rag_version, config_name, csv_path))
    
    return other_configs


def analyze_other_configs(dataset: str):
    """
    Analyze other discovered configurations for a specific dataset.
    """
    configs = discover_other_configs()
    
    # Filter for the specific dataset
    dataset_configs = [c for c in configs if c[0] == dataset]
    
    if not dataset_configs:
        return
    
    results = []
    
    # Add baseline first
    baseline_filename = f"{dataset}-DAHR-Qwen_4B.csv"
    baseline_path = ROOT / baseline_filename
    baseline_recall = calculate_recall_at_k(baseline_path, k=5)
    
    results.append({
        'Dataset': dataset,
        'RAG Version': 'DAHR',
        'Configuration': 'Baseline (Qwen_4B)',
        'Recall@5 (%)': baseline_recall
    })
    
    # Add other configurations for this dataset
    for _, rag_version, config_name, csv_path in dataset_configs:
        recall = calculate_recall_at_k(Path(csv_path), k=5)
        results.append({
            'Dataset': dataset,
            'RAG Version': rag_version,
            'Configuration': config_name,
            'Recall@5 (%)': recall
        })
    
    df = pd.DataFrame(results)
    print_table(df, f"{dataset} - Other Configurations - Recall@5 (%)")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_table(df: pd.DataFrame, title: str):
    """Print a formatted table with title. Use NA for NaNs."""
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}")
    if df.empty:
        print("No data available.")
    else:
        # Fill NaNs with "NA" for display
        display_df = df.fillna("NA")
        # Updated floatfmt from .2f to .3f for an extra decimal of precision
        print(display_df.to_markdown(floatfmt=".3f"))
    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    
    # Define order: MultihopRAG first, then HotpotQA
    ordered_datasets = ["MultihopRAG", "HotpotQA"]
    
    for dataset in ordered_datasets:
        print(f"\n{'#' * 80}")
        print(f"ANALYSIS FOR DATASET: {dataset}")
        print(f"{'#' * 80}")
        
        analyze_embeddings(dataset)
        analyze_retrieval_components(dataset)
        analyze_other_configs(dataset)
    


if __name__ == "__main__":
    main()
