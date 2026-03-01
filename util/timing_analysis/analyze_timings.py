import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import typing

# Define the files to analyze here. 
# These files must be located in the same directory as this script (util/timing_analysis).
FILES_TO_ANALYZE = [
    "HotpotQA_100.json",
    "HotpotQA_1k-DocAwareHybridRAG.json",
    "HotpotQA_Dev-DocAwareHybridRAG.json"
]

# Size in chunks
DATASET_TO_SIZE = {
    "HotpotQA_100.json": 231,
    "HotpotQA_Dev-DocAwareHybridRAG.json": 10141,
    "HotpotQA_1k-DocAwareHybridRAG.json": 2527
}

def analyze_file(file_path: Path) -> typing.Optional[typing.Dict[str, float]]:
    """
    Analyzes a single timing JSON file.
    Returns a dictionary of average times for each step, or None if failed.
    """
    print(f"\nAnalyzing: {file_path.name}")
    try:
        with file_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                print(f"Skipping {file_path.name}: Expected a list of records.")
                return None
    except Exception as e:
        print(f"Error reading {file_path.name}: {e}")
        return None

    if not data:
        print(f"No valid data points found in {file_path.name}.")
        return None

    # Aggregate timings
    aggregated_timings = defaultdict(list)
    total_queries = len(data)

    for record in data:
        timings = record.get("timings", {})
        for key, value in timings.items():
            aggregated_timings[key].append(value)

    # Calculate averages
    avg_stats = {}
    for key, values in aggregated_timings.items():
        avg_time = sum(values) / len(values)
        avg_stats[key] = avg_time
    
    # Print report
    print(f"Total Queries: {total_queries}")
    print("-" * 65)
    print(f"{'Step Name':<40} | {'Avg (s)':<10}")
    print("-" * 65)
    
    sorted_stats = sorted(avg_stats.items(), key=lambda x: x[1], reverse=True)
    for key, avg in sorted_stats:
        print(f"{key:<40} | {avg:<10.4f}")
    print("-" * 65)

    return avg_stats

def analyze_all_and_plot():
    current_dir = Path(__file__).parent
    print(f"Checking for files in: {current_dir}")
    
    # List of tuples: (size, filename, stats_dict)
    results = []

    for filename in FILES_TO_ANALYZE:
        file_path = current_dir / filename
        if file_path.exists():
            stats = analyze_file(file_path)
            if stats:
                # Get size from mapping, defaulting to 0 or some indicator if not found
                # Note: DATASET_TO_SIZE keys might need to match filename exactly or just the key
                # The original script had slightly different keys in DATASET_TO_SIZE than filenames
                # Dictionary keys in analyze_timings.py (original) did NOT have .json extension
                # But FILES_TO_ANALYZE has .json extension. 
                # I updated DATASET_TO_SIZE to include extensions for easier matching.
                size = DATASET_TO_SIZE.get(filename)
                if size is None:
                    # Fallback check without extension
                    size = DATASET_TO_SIZE.get(filename.replace(".json", ""))
                
                if size is not None:
                    results.append((size, filename, stats))
                else:
                    print(f"Warning: No size found for {filename} in DATASET_TO_SIZE. Skipping plot for this file.")
        else:
            print(f"⚠️ File not found: {filename}")

    if not results:
        print("No valid data collected for plotting.")
        return

    # Sort by database size
    results.sort(key=lambda x: x[0])

    sizes = [r[0] for r in results]
    # Collect all unique keys (steps) across all files
    all_keys = set()
    for _, _, stats in results:
        all_keys.update(stats.keys())
    
    # Remove 'total_search_time' if present, as it's the sum of others and we want stacked bar
    if 'total_search_time' in all_keys:
        all_keys.remove('total_search_time')
    
    sorted_keys = sorted(list(all_keys)) # Sort keys for consistent stacking order

    # Prepare data for plotting
    # data_for_plot is a dict: {step_name: [time_for_size_1, time_for_size_2, ...]}
    data_for_plot = defaultdict(list)
    
    for _, _, stats in results:
        for key in sorted_keys:
            data_for_plot[key].append(stats.get(key, 0.0))

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for stackplot
    # stackplot expects y as a list of arrays, one for each stack
    y_values = [data_for_plot[key] for key in sorted_keys]
    
    ax.stackplot(sizes, y_values, labels=sorted_keys, alpha=0.8)

    ax.set_xlabel('Dataset Size (Chunks)')
    ax.set_ylabel('Average Time (s)')
    ax.set_title('Retrieval Component Timings by Dataset Size')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_all_and_plot()
