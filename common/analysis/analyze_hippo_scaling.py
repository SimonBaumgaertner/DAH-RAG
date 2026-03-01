import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from common.data_classes.evaluation import EntryType

def analyze_generation_times(csv_path: str):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # A retrieval stage ends where entry_type == 'scaling_retrieval_track'
    step_end_indices = df[df['entry_type'] == EntryType.SCALING_RETRIEVAL_TRACK.value].index
    
    prev_idx = 0
    results = []
    
    all_gen_times = []
    
    for end_idx in step_end_indices:
        step_count_str = df.loc[end_idx, 'identifier']
        actual_retrieval_time_ms = float(df.loc[end_idx, 'value']) # Total retrieval time for this stage
        
        try:
            step_count = int(step_count_str)
        except:
            continue
            
        # The slice of dataframe for this iteration
        step_slice = df.iloc[prev_idx:end_idx]
        
        # Filter for generation times
        gen_slice = step_slice[step_slice['entry_type'] == 'llm_retrieval_generation_time_track'].copy()
        
        if not gen_slice.empty:
            # Convert values to numeric
            gen_slice['value'] = pd.to_numeric(gen_slice['value'])
            
            # Sum of generation times for this stage
            total_gen_time_ms = gen_slice['value'].sum()
            num_queries = len(gen_slice)
            avg_gen_time_ms = total_gen_time_ms / num_queries
            
            # Store all individual gen times to calculate a global mean later if needed,
            # or just calculate the global mean of the TOTAL generation times.
            
            results.append({
                'step_count': step_count, 
                'avg_generation_time': avg_gen_time_ms,
                'total_generation_time': total_gen_time_ms,
                'actual_retrieval_time_ms': actual_retrieval_time_ms,
                'queries': num_queries
            })
            print(f"Step {step_count}: {num_queries} queries, Total Gen: {total_gen_time_ms/1000:.2f}s, Actual Ret: {actual_retrieval_time_ms/1000:.2f}s")
        else:
             print(f"Step {step_count}: No queries found")
            
        prev_idx = end_idx + 1
        
    if not results:
        print("No generation times found in any stage.")
        return
        
    res_df = pd.DataFrame(results)
    
    # Calculate regular mean total generation time across all stages
    global_mean_total_gen_ms = res_df['total_generation_time'].mean()
    print(f"\nGlobal Mean Total Generation Time per stage: {global_mean_total_gen_ms/1000:.2f}s")
    
    # Normalize: Target is (Actual - Total Gen this step) + Global Mean Total Gen
    res_df['normalized_retrieval_time_ms'] = res_df['actual_retrieval_time_ms'] - res_df['total_generation_time'] + global_mean_total_gen_ms
    
    res_df = res_df.sort_values('step_count')
    
    # Convert to seconds for plotting
    res_df['actual_retrieval_s'] = res_df['actual_retrieval_time_ms'] / 1000.0
    res_df['normalized_retrieval_s'] = res_df['normalized_retrieval_time_ms'] / 1000.0
    res_df['total_gen_s'] = res_df['total_generation_time'] / 1000.0
    res_df['retrieval_minus_gen_s'] = (res_df['actual_retrieval_time_ms'] - res_df['total_generation_time']) / 1000.0
    
    # Print a small table
    print("\nSummary:")
    print(res_df[['step_count', 'actual_retrieval_s', 'total_gen_s', 'normalized_retrieval_s', 'retrieval_minus_gen_s']].to_string(index=False))

    # Plot
    plt.figure(figsize=(12, 7))
    plt.plot(res_df['step_count'], res_df['actual_retrieval_s'], marker='o', linestyle='-', color='#e02b35', label='Actual Retrieval Time')
    plt.plot(res_df['step_count'], res_df['normalized_retrieval_s'], marker='s', linestyle='--', color='#59a89c', label='Normalized (Constant Generation Time)')
    plt.plot(res_df['step_count'], res_df['retrieval_minus_gen_s'], marker='^', linestyle=':', color='#0077b6', label='Retrieval minus Generation (Pure Logic/DB)')
    
    plt.title('Retrieval Time vs Total Documents (HippoRAG2)')
    plt.xlabel('Total Documents')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    csv_file = os.path.join(os.path.dirname(__file__), '../../logs_and_tracks/scaling_studies/Scaling-HippoRAG.csv')
    analyze_generation_times(csv_file)
