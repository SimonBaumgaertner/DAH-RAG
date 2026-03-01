import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from common.analysis.log_analyzer import LogAnalyzer
import matplotlib.pyplot as plt
import numpy as np

# We decide on a list of datasets as well as a list of approaches
datasets = [
    "HotpotQA",
    "MultiHopRAG",
    "NovelQA",
    "PubMedQA"
]
approaches = [
    "No RAG",
    "BM25",
    "NaiveVectorDB",
    "RaptorRAG",
    "HippoRAG",
    "Document-Aware Hybrid-RAG",
]

# Mapping for datasets to file prefixes
dataset_file_map = {
    "PubMedQA": "PubMedQA_10k",
    "HotpotQA": "HotpotQA_1k",
    "MultiHopRAG": "MultiHopRAG",
    "NovelQA": "NovelQA",
}

# Mapping for approaches to file suffixes
approach_file_map = {
    "Document-Aware Hybrid-RAG": "DocAwareHybridRAG",
    "No RAG": "NoRAGGeneration"
}

# Map approaches to colors
approach_color_map = {
    "No RAG": "#bababa", # gray
    "BM25": "#f0c571", # gold
    "NaiveVectorDB": "#082a54", # dark blue 
    "RaptorRAG": "#a559aa", # purple
    "HippoRAG": "#e02b35", # red
    "Document-Aware Hybrid-RAG": "#59a89c" # Teal
}


# We load the statistics from the prime_studies folder and plot a matrix of the existing / missing experiments
# A name of a file looks e.g. like this: "HotpotQA_1k-BM25.csv"

def get_file_name(dataset, approach):
    d_name = dataset_file_map.get(dataset, dataset)
    a_name = approach_file_map.get(approach, approach)
    return f"{d_name}-{a_name}.csv"

def check_experiments():
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs_and_tracks", "prime_studies")
    
    # Check if path exists, if not try absolute path assumption or relative from project root
    if not os.path.exists(base_path):
        # Fallback to absolute path based on user environment if needed
        base_path = "/home/simon/PycharmProjects/MastersThesis/logs_and_tracks/prime_studies"
    
    if not os.path.exists(base_path):
        print(f"Error: Directory not found at {base_path}")
        return

    matrix = {}

    # Build the matrix data
    for dataset in datasets:
        matrix[dataset] = {}
        for approach in approaches:
            file_name = get_file_name(dataset, approach)
            file_path = os.path.join(base_path, file_name)
            exists = os.path.exists(file_path)
            matrix[dataset][approach] = "✅" if exists else "❌"

    # --- Console Output (Commented out to remove "thingy in console") ---
    # header = f"{'Dataset':<15} | " + " | ".join([f"{a[:10]:<10}" for a in approaches])
    # print("\nExperiment Matrix:")
    # print(header)
    # print("-" * len(header))
    
    # Prepare data for plotting
    plot_data = []
    
    for dataset in datasets:
        # row = f"{dataset:<15} | "
        plot_row = []
        for approach in approaches:
            status_icon = matrix[dataset][approach]
            # row += f"{status_icon:<10} | "
            # 1 for Found, 0 for Missing
            plot_row.append(1 if status_icon == "✅" else 0)
        # print(row) # Printing removed
        plot_data.append(plot_row)

    # Plotting
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(os.path.dirname(__file__), "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Setup the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Define colors: Red for missing (0), Green for found (1)
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['#FFCDD2', '#C8E6C9']) # pastel red, pastel green
        
        # Plot the heatmap
        im = ax.imshow(plot_data, cmap=cmap, vmin=0, vmax=1)
        
        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(approaches)))
        ax.set_yticks(np.arange(len(datasets)))
        ax.set_xticklabels(approaches, rotation=45, ha="right")
        ax.set_yticklabels(datasets)
        
        # Loop over data dimensions and create text annotations.
        for i in range(len(datasets)):
            for j in range(len(approaches)):
                # CHANGED: Use text instead of emojis to fix Glyph Error
                text_val = "Done" if plot_data[i][j] == 1 else "Missing"
                
                text = ax.text(j, i, text_val,
                               ha="center", va="center", color="black", fontsize=12)
                               
        ax.set_title("Comparative Study Matrix")
        fig.tight_layout()
        
        plt.show()
        

    except ImportError as e:
        print("\nCould not generate plot: matplotlib or numpy not found.")
        print(f"Error: {e}")


def plot_metric(metric_name, y_label, output_filename, metric_extractor, is_percentage=True):
    print(f"\nPlotting {metric_name}...")
    
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs_and_tracks", "prime_studies")
    
    # Check if path exists, if not try absolute path assumption or relative from project root
    if not os.path.exists(base_path):
        # Fallback to absolute path based on user environment if needed
        base_path = "/home/simon/PycharmProjects/MastersThesis/logs_and_tracks/prime_studies"
    
    if not os.path.exists(base_path):
        print(f"Error: Directory not found at {base_path}")
        return

    # Prepare data for plotting
    data_values = {} # dataset -> {approach -> value}
    
    for dataset in datasets:
        data_values[dataset] = {}
        for approach in approaches:
            file_name = get_file_name(dataset, approach)
            file_path = os.path.join(base_path, file_name)
            
            val = 0.0
            if os.path.exists(file_path):
                try:
                    analyzer = LogAnalyzer(csv_file=file_path)
                    val = metric_extractor(analyzer)
                except Exception as e:
                    print(f"Error reading {file_name}: {e}")
                    val = 0.0
            
            data_values[dataset][approach] = val

    # Plotting
    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Bar configuration
    bar_width = 0.12 # Depends on number of approaches (6 approaches * 0.12 = 0.72 width per group)
    x = np.arange(len(datasets))
    
    # Create grouped bars
    for i, approach in enumerate(approaches):
        values = [data_values[dataset][approach] for dataset in datasets]
        
        # Determine x positions for this approach's bars
        # Center the group around the tick: tick is at x.
        # Total group width is roughly len(approaches) * bar_width.
        # Start offset calculation ensuring centering.
        offset = (i - (len(approaches) - 1) / 2) * bar_width
        
        bars = ax.bar(x + offset, values, width=bar_width, label=approach, color=approach_color_map.get(approach, "#000000"))
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}' if is_percentage else f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, rotation=90)

    # Labels and Title
    ax.set_xlabel('Datasets')
    ax.set_ylabel(y_label)
    ax.set_title(f'{metric_name} by Dataset and Approach')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(title="Approaches", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if is_percentage:
        ax.set_ylim(0, 100) # Percentage
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    
    output_path = os.path.join(plots_dir, output_filename)
    plt.show()

def plot_accuracies():
    plot_metric(
        metric_name="QA Accuracy",
        y_label="Accuracy (%)",
        output_filename="accuracy_bargraph.png",
        metric_extractor=lambda analyzer: analyzer.qa_accuracy()
    )

def plot_index_scaling():
    plot_metric(
        metric_name="Indexing Scaling",
        y_label="Total Index Time (Minutes)",
        output_filename="indexing_scaling_bargraph.png",
        metric_extractor=lambda analyzer: analyzer.total_indexing_time() / 1000.0 / 60.0,
        is_percentage=False
    )

def plot_retrieval_scaling():
    plot_metric(
        metric_name="Retrieval Scaling",
        y_label="Avg Retrieval Time (ms)",
        output_filename="retrieval_scaling_bargraph.png",
        metric_extractor=lambda analyzer: analyzer.retrieval_time_avg(),
        is_percentage=False
    )

def plot_recall_at_5():
    plot_metric(
        metric_name="Recall@5",
        y_label="Recall@5 (%)",
        output_filename="recall_at_5_bargraph.png",
        metric_extractor=lambda analyzer: analyzer.recall_at_k(5)
    )

if __name__ == "__main__":
    check_experiments()
    plot_accuracies()
    plot_recall_at_5()
    plot_index_scaling()
    plot_retrieval_scaling()