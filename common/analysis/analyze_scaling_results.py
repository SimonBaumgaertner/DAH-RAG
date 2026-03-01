import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from common.data_classes.evaluation import EntryType

# Map approaches to colors
approach_color_map = {
    "BM25": "#f0c571",  # gold
    "VectorDB": "#082a54",  # dark blue 
    "Vector + Reranker": "#082a54", # dark blue (same as VectorDB)
    "RaptorRAG": "#a559aa",  # purple
    "HippoRAG": "#e02b35",  # red
    "HippoRAG2": "#e02b35",  # red (same as HippoRAG)
    "Document-Aware Hybrid-RAG": "#59a89c",  # Teal
    "Document-Aware Hybrid-RAG + Reranker": "#59a89c", # Teal (same as DAHR)

}

# Increase overall plot text sizes
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 16
})

# Map filename patterns to display names
approach_display_names = {
    "Scaling-BM25": "BM25",
    "NaiveVectorDB": "VectorDB",
    "RaptorRAG": "RaptorRAG",
    "HippoRAG": "HippoRAG2",
    "DocAwareHybridRAG": "Document-Aware Hybrid-RAG",
    "VectorDB+Rerank": "Vector + Reranker",
    "DAHR+Rerank": "Document-Aware Hybrid-RAG + Reranker"
}

desired_legend_order = [
    "BM25",
    "VectorDB",
    "Vector + Reranker",
    "RaptorRAG",
    "HippoRAG2",
    "Document-Aware Hybrid-RAG",
    "Document-Aware Hybrid-RAG + Reranker"
]

def sort_results(results_dict):
    def get_sort_key(item):
        filename = item[0]
        display_name = get_display_name(filename)
        try:
            return desired_legend_order.index(display_name)
        except ValueError:
            return len(desired_legend_order)
    return sorted(results_dict.items(), key=get_sort_key)

def get_display_name(filename: str) -> str:
    """Extract approach name from filename and map to display name."""
    filename_lower = filename.lower()
    for key, display_name in approach_display_names.items():
        if key.lower() in filename_lower:
            return display_name
    return filename  # fallback to original filename

def analyze_scaling_results(log_dir: str):
    """
    Analyzes all scaling experiment CSV logs in the directory.
    """
    csv_files = glob.glob(os.path.join(log_dir, "*.csv"))
    
    if not csv_files:
        print(f"⚠️ No CSV files found in {log_dir}")
        return

    print(f"📊 Found {len(csv_files)} logs in {log_dir}\n")

    results = {}

    # 1. Process each file and print tables
    for csv_path in sorted(csv_files):
        filename = os.path.basename(csv_path)
        print(f"--- 📄 Analyzing {filename} ---")
        
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"❌ Failed to read {filename}: {e}")
            continue

        # Filter for relevant tracks
        indexing_mask = df['entry_type'] == EntryType.SCALING_INDEXING_TRACK.value
        retrieval_mask = df['entry_type'] == EntryType.SCALING_RETRIEVAL_TRACK.value
        
        indexing_df = df[indexing_mask].copy()
        retrieval_df = df[retrieval_mask].copy()

        def process_indexing_df(sub_df):
            if sub_df.empty:
                return sub_df
            
            sub_df['step_count'] = pd.to_numeric(sub_df['identifier'], errors='coerce')
            sub_df['duration'] = pd.to_numeric(sub_df['value'], errors='coerce') / 3600000.0 # Convert ms to hours
            
            sub_df = sub_df.dropna(subset=['step_count', 'duration'])
            sub_df = sub_df.sort_values('step_count')
            return sub_df
        
        def process_retrieval_df(sub_df):
            if sub_df.empty:
                return sub_df
            
            sub_df = sub_df.copy()
            sub_df['step_count'] = pd.to_numeric(sub_df['identifier'], errors='coerce')
            sub_df['duration'] = pd.to_numeric(sub_df['value'], errors='coerce') / 1000.0 # Convert ms to s
            
            sub_df = sub_df.dropna(subset=['step_count', 'duration'])
            sub_df = sub_df.sort_values('step_count')
            return sub_df

        indexing_df = process_indexing_df(indexing_df)
        retrieval_df = process_retrieval_df(retrieval_df)

        # Helper to calculate Recall@5
        def calculate_recall_at_k(proof_df, k=5):
            if proof_df.empty:
                return 0.0
            
            # proof_track: identifier=question_id, value=rank
            ranks = {} # q_id -> list of ranks
            for _, row in proof_df.iterrows():
                qid = row['identifier']
                try:
                    rank = int(float(row['value'])) # parser might yield float string
                except:
                    continue
                if qid not in ranks:
                    ranks[qid] = []
                ranks[qid].append(rank)
            
            if not ranks:
                return 0.0
            
            question_recalls = []
            for q_ranks in ranks.values():
                total_proofs = len(q_ranks)
                if total_proofs == 0:
                    continue
                # found in top k (rank < k, 0-indexed assumed? LogAnalyzer uses 0 <= r < k)
                # LogAnalyzer says: r != -1 and 0 <= r < k
                found = sum(1 for r in q_ranks if r != -1 and 0 <= r < k)
                question_recalls.append(found / total_proofs)
            
            if not question_recalls:
                return 0.0
            
            return 100.0 * (sum(question_recalls) / len(question_recalls))

        # Get indices of the scaling tracks to define boundaries
        step_end_indices = df[df['entry_type'] == EntryType.SCALING_RETRIEVAL_TRACK.value].index
        
        recall_data = [] # (step_count, recall_at_5)
        generation_data = []
        
        prev_idx = 0
        for end_idx in step_end_indices:
            # Identifier of the track is the step count
            step_count_str = df.loc[end_idx, 'identifier']
            try:
                step_count = int(step_count_str)
            except:
                continue
                
            # Slice the dataframe for this step
            # -1 to exclude the scaling track itself if needed, but filtering by type handles it
            step_slice = df.iloc[prev_idx:end_idx] 
            
            proof_slice = step_slice[step_slice['entry_type'] == 'proof_track']
            recall = calculate_recall_at_k(proof_slice, k=5)
            
            recall_data.append({'step_count': step_count, 'recall_at_5': recall})
            
            gen_slice = step_slice[step_slice['entry_type'] == 'llm_retrieval_generation_time_track']
            if not gen_slice.empty:
                total_gen_ms = pd.to_numeric(gen_slice['value']).sum()
                generation_data.append({'step_count': step_count, 'total_gen_ms': total_gen_ms})
            
            prev_idx = end_idx + 1 # Start next step after this track
            
        recall_df = pd.DataFrame(recall_data)
        if not recall_df.empty:
            recall_df = recall_df.sort_values('step_count')

        gen_df = pd.DataFrame(generation_data)
        if not gen_df.empty:
            gen_df = gen_df.sort_values('step_count')

        # Compute normalized duration and overwrite duration
        if not retrieval_df.empty and not gen_df.empty:
            retrieval_df = pd.merge(retrieval_df, gen_df, on='step_count', how='left')
            retrieval_df['total_gen_ms'] = retrieval_df['total_gen_ms'].fillna(0)
            global_mean_total_gen_s = (retrieval_df['total_gen_ms'].mean()) / 1000.0
            retrieval_df['duration'] = retrieval_df['duration'] - (retrieval_df['total_gen_ms'] / 1000.0) + global_mean_total_gen_s

        # Merge recall into retrieval results for easier handling (optional)
        # But we'll just store it in results
        
        if not indexing_df.empty:
             indexing_df['cumulative_duration'] = indexing_df['duration'].cumsum()

        results[filename] = {
            'indexing': indexing_df,
            'retrieval': retrieval_df,
            'recall': recall_df
        }

        print("\n📈 Indexing Results:")
        if not indexing_df.empty:
            print(indexing_df[['step_count', 'duration', 'cumulative_duration']].to_markdown(index=False))
        else:
            print("No data.")

        print("\n📉 Retrieval Results:")
        if not retrieval_df.empty:
            # Merge recall if available
            if not recall_df.empty:
                display_df = pd.merge(retrieval_df, recall_df, on='step_count', how='left')
                print(display_df[['step_count', 'duration', 'recall_at_5']].to_markdown(index=False))
            else:
                print(retrieval_df[['step_count', 'duration']].to_markdown(index=False))
        else:
            print("No data.")
        print("\n" + "="*40 + "\n")

    # 2. Cumulative Comparing Plot
    plt.figure(figsize=(10, 6))
    for filename, data in sort_results(results):
        idx_df = data['indexing']
        if not idx_df.empty:
            display_name = get_display_name(filename)
            
            # Skip "Vector + Reranker" for indexing graph as it's the same as "VectorDB"
            if "Reranker" in display_name:
                continue
                
            color = approach_color_map.get(display_name, '#000000')
            marker_type = '^' if 'Reranker' in display_name else 'o'
            plt.plot(idx_df['step_count'], idx_df['cumulative_duration'], 
                    marker=marker_type, linestyle='-', color=color, label=display_name)
    
    plt.title('Cumulative Indexing Time vs Total Documents')
    plt.xlabel('Total Documents')
    plt.ylabel('Time (h)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3. Retrieval Comparison Plot
    plt.figure(figsize=(10, 6))
    for filename, data in sort_results(results):
        ret_df = data['retrieval']
        if not ret_df.empty:
            display_name = get_display_name(filename)
            color = approach_color_map.get(display_name, '#000000')
            marker_type = '^' if 'Reranker' in display_name else 'o'
            plt.plot(ret_df['step_count'], ret_df['duration'], 
                    marker=marker_type, linestyle='-', color=color, label=display_name)

    plt.title('Retrieval Time vs Total Documents')
    plt.xlabel('Total Documents')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 3.5 Recall Comparison Plot
    plt.figure(figsize=(10, 6))
    for filename, data in sort_results(results):
        rec_df = data.get('recall')
        if rec_df is not None and not rec_df.empty:
            display_name = get_display_name(filename)
            color = approach_color_map.get(display_name, '#000000')
            marker_type = '^' if 'Reranker' in display_name else 'o'
            plt.plot(rec_df['step_count'], rec_df['recall_at_5'], 
                    marker=marker_type, linestyle='-', color=color, label=display_name)

    plt.title('Recall@5 vs Total Documents')
    plt.xlabel('Total Documents')
    plt.ylabel('Recall@5 (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 4. Individual Plots (Indexing + Retrieval + Recall)
    for filename, data in results.items():
        idx_df = data['indexing']
        ret_df = data['retrieval']
        rec_df = data.get('recall')
        
        if idx_df.empty and ret_df.empty:
            continue
            
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        lines = []
        labels = []

        if not idx_df.empty:
            l1, = ax1.plot(idx_df['step_count'], idx_df['cumulative_duration'], 'b-', label='Cumulative Indexing')
            lines.append(l1)
            labels.append(l1.get_label())
            ax1.set_xlabel('Total Documents')
            ax1.set_ylabel('Cumulative Indexing Time (hours)', color='b')
            ax1.tick_params(axis='y', labelcolor='b')

        if not ret_df.empty:
            ax2 = ax1.twinx()
            l2, = ax2.plot(ret_df['step_count'], ret_df['duration'], 'r--', label='Retrieval Time (Normalized if HippoRAG)')
            lines.append(l2)
            labels.append(l2.get_label())
            
            ax2.set_ylabel('Retrieval Time (s)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
        if rec_df is not None and not rec_df.empty:
            ax3 = ax1.twinx()
            ax3.spines.right.set_position(("axes", 1.2))
            
            l3, = ax3.plot(rec_df['step_count'], rec_df['recall_at_5'], 'g:', marker='^', label='Recall@5')
            lines.append(l3)
            labels.append(l3.get_label())
            ax3.set_ylabel('Recall@5 (%)', color='g')
            ax3.tick_params(axis='y', labelcolor='g')
            
        plt.title(f'Scaling Performance: {filename}')
        plt.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)
        plt.grid(True)
        plt.tight_layout() # Adjust layout to make room for the third axis
        plt.show()
        
if __name__ == "__main__":
    LOG_DIR = "logs_and_tracks/scaling_studies/"
    analyze_scaling_results(LOG_DIR)
