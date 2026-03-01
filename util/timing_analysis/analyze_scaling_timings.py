import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# The first question in each block - used to identify block boundaries
FIRST_QUESTION = "Were Scott Derrickson and Ed Wood of the same nationality?"

def load_and_separate_blocks(file_path: Path):
    """
    Load the JSON file and separate it into 8 blocks based on the first question.
    Returns a list of 8 blocks, where each block is a list of query records.
    """
    print(f"Loading: {file_path.name}")
    
    with file_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Expected a list of records in the JSON file")
    
    # Separate into blocks
    blocks = []
    current_block = []
    
    for record in data:
        query = record.get("query", "")
        
        # If we encounter the first question and we already have data in current_block,
        # it means we're starting a new block
        if query == FIRST_QUESTION and current_block:
            blocks.append(current_block)
            current_block = []
        
        current_block.append(record)
    
    # Don't forget the last block
    if current_block:
        blocks.append(current_block)
    
    print(f"Found {len(blocks)} blocks")
    for i, block in enumerate(blocks):
        print(f"  Block {i+1}: {len(block)} queries")
    
    return blocks

def calculate_block_statistics(blocks):
    """
    Calculate average timings for each block.
    Returns a dict: {timing_key: [avg_for_block1, avg_for_block2, ...]}
    """
    block_stats = []
    
    for block_idx, block in enumerate(blocks):
        # Aggregate timings for this block
        aggregated_timings = defaultdict(list)
        
        for record in block:
            timings = record.get("timings", {})
            for key, value in timings.items():
                aggregated_timings[key].append(value)
        
        # Calculate averages for this block
        avg_stats = {}
        for key, values in aggregated_timings.items():
            avg_stats[key] = sum(values) / len(values) if values else 0.0
        
        block_stats.append(avg_stats)
    
    # Reorganize data: {timing_key: [avg_for_block1, avg_for_block2, ...]}
    timing_keys = set()
    for stats in block_stats:
        timing_keys.update(stats.keys())
    
    # Remove 'total_search_time' as it's the sum of others
    if 'total_search_time' in timing_keys:
        timing_keys.remove('total_search_time')
    
    result = {}
    for key in timing_keys:
        result[key] = [stats.get(key, 0.0) for stats in block_stats]
    
    # Also get total_search_time separately for reference
    result['total_search_time'] = [stats.get('total_search_time', 0.0) for stats in block_stats]
    
    return result

def plot_timing_development(timing_data, output_dir: Path):
    """
    Create comprehensive visualizations of timing development across blocks.
    """
    num_blocks = len(next(iter(timing_data.values())))
    block_labels = [f"Block {i+1}" for i in range(num_blocks)]
    block_numbers = np.arange(1, num_blocks + 1)
    
    # Get timing keys (excluding total)
    timing_keys = [k for k in timing_data.keys() if k != 'total_search_time']
    timing_keys = sorted(timing_keys)
    
    # Define a nice color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(timing_keys)))
    
    # ========== Plot 1: Line Plot for Each Timing Component ==========
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    
    for idx, key in enumerate(timing_keys):
        values = timing_data[key]
        ax1.plot(block_numbers, values, marker='o', linewidth=2, 
                label=key.replace('_', ' ').title(), color=colors[idx])
    
    ax1.set_xlabel('Block Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Timing Component Development Across Scaling Blocks', 
                  fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(block_numbers)
    ax1.set_xticklabels(block_labels, rotation=0)
    plt.tight_layout()
    
    # ========== Plot 2: Stacked Area Chart ==========
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    
    # Prepare data for stacked area
    y_values = [timing_data[key] for key in timing_keys]
    
    ax2.stackplot(block_numbers, y_values, labels=[k.replace('_', ' ').title() for k in timing_keys],
                  colors=colors, alpha=0.8)
    
    ax2.set_xlabel('Block Number', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Stacked Timing Components Across Scaling Blocks', 
                  fontsize=14, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(block_numbers)
    ax2.set_xticklabels(block_labels, rotation=0)
    plt.tight_layout()
    
    # ========== Plot 3: Individual Component Subplots ==========
    num_components = len(timing_keys)
    cols = 3
    rows = (num_components + cols - 1) // cols
    
    fig3, axes = plt.subplots(rows, cols, figsize=(16, rows * 3))
    axes = axes.flatten() if num_components > 1 else [axes]
    
    for idx, key in enumerate(timing_keys):
        ax = axes[idx]
        values = timing_data[key]
        
        ax.plot(block_numbers, values, marker='o', linewidth=2.5, 
               color=colors[idx], markersize=8)
        ax.fill_between(block_numbers, values, alpha=0.3, color=colors[idx])
        
        ax.set_title(key.replace('_', ' ').title(), fontweight='bold')
        ax.set_xlabel('Block Number', fontsize=9)
        ax.set_ylabel('Avg Time (s)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(block_numbers)
        ax.set_xticklabels(block_labels, rotation=45, ha='right', fontsize=8)
    
    # Hide unused subplots
    for idx in range(num_components, len(axes)):
        axes[idx].axis('off')
    
    fig3.suptitle('Individual Timing Component Development', 
                  fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # ========== Plot 4: Total Search Time Development ==========
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    
    total_times = timing_data['total_search_time']
    ax4.plot(block_numbers, total_times, marker='o', linewidth=3, 
            color='darkred', markersize=10, label='Total Search Time')
    ax4.fill_between(block_numbers, total_times, alpha=0.2, color='darkred')
    
    # Add value labels on points
    for i, (x, y) in enumerate(zip(block_numbers, total_times)):
        ax4.annotate(f'{y:.2f}s', (x, y), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')
    
    ax4.set_xlabel('Block Number', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Average Total Search Time (seconds)', fontsize=12, fontweight='bold')
    ax4.set_title('Total Search Time Development Across Scaling Blocks', 
                  fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(block_numbers)
    ax4.set_xticklabels(block_labels, rotation=0)
    ax4.legend(fontsize=10)
    plt.tight_layout()
    
    # ========== Plot 5: Percentage Breakdown by Block ==========
    fig5, ax5 = plt.subplots(figsize=(14, 8))
    
    # Calculate percentages
    percentages = []
    for block_idx in range(num_blocks):
        total = sum(timing_data[key][block_idx] for key in timing_keys)
        block_percentages = [timing_data[key][block_idx] / total * 100 if total > 0 else 0 
                            for key in timing_keys]
        percentages.append(block_percentages)
    
    # Transpose for stacking
    percentages_transposed = list(zip(*percentages))
    
    # Create stacked bar chart
    bottom = np.zeros(num_blocks)
    for idx, key in enumerate(timing_keys):
        values = percentages_transposed[idx]
        ax5.bar(block_numbers, values, bottom=bottom, label=key.replace('_', ' ').title(),
               color=colors[idx], alpha=0.8)
        bottom += values
    
    ax5.set_xlabel('Block Number', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Percentage of Total Time (%)', fontsize=12, fontweight='bold')
    ax5.set_title('Timing Component Percentage Breakdown by Block', 
                  fontsize=14, fontweight='bold')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax5.set_xticks(block_numbers)
    ax5.set_xticklabels(block_labels, rotation=0)
    ax5.set_ylim(0, 100)
    plt.tight_layout()
    
    # Show all plots
    plt.show()
    
    print("\n" + "="*70)
    print("TIMING STATISTICS SUMMARY")
    print("="*70)
    
    # Print statistics table
    print(f"\n{'Component':<45} | {'Block 1':<10} | {'Block 8':<10} | {'Change':<10}")
    print("-" * 85)
    
    for key in timing_keys:
        block1_val = timing_data[key][0]
        block8_val = timing_data[key][-1]
        change = ((block8_val - block1_val) / block1_val * 100) if block1_val > 0 else 0
        change_str = f"{change:+.1f}%"
        
        print(f"{key.replace('_', ' ').title():<45} | {block1_val:<10.4f} | {block8_val:<10.4f} | {change_str:<10}")
    
    print("-" * 85)
    total_block1 = timing_data['total_search_time'][0]
    total_block8 = timing_data['total_search_time'][-1]
    total_change = ((total_block8 - total_block1) / total_block1 * 100) if total_block1 > 0 else 0
    print(f"{'TOTAL SEARCH TIME':<45} | {total_block1:<10.4f} | {total_block8:<10.4f} | {total_change:+.1f}%")
    print("="*70)

def main():
    # File to analyze
    current_dir = Path(__file__).parent
    file_path = current_dir / "Scaling-DocAwareHybridRAG.json"
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    # Load and separate into blocks
    blocks = load_and_separate_blocks(file_path)
    
    if len(blocks) != 8:
        print(f"⚠️  Warning: Expected 8 blocks but found {len(blocks)}")
    
    # Calculate statistics for each block
    timing_data = calculate_block_statistics(blocks)
    
    # Create visualizations
    plot_timing_development(timing_data, current_dir)

if __name__ == "__main__":
    main()
