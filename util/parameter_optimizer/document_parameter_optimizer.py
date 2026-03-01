import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Optional

# ---------------------------------------------------------
# 1. Data Loading & Structure
# ---------------------------------------------------------

class RankingDataProcessor:
    def __init__(self, json_data: List[Dict]):
        self.raw_data = json_data
        self.feature_names = []
        self.scaler = StandardScaler()

    def process_documents(self):
        """
        Parses the JSON to create a dataset for Document Ranking.
        Returns:
            X (Tensor): Feature matrix
            pairs (List): List of tuples (positive_index, negative_index)
        """
        rows = []
        
        # 1. Flatten data into a DataFrame format
        # We need to identify valid feature keys dynamically from the first valid record
        if not self.raw_data:
            raise ValueError("No data provided")
            
        # identifying feature keys (excluding strings like 'document_id')
        # We look for the first valid entry with document scores
        sample_doc = None
        for entry in self.raw_data:
            if entry.get('document_scores'):
                sample_doc = entry['document_scores'][0]
                break
        
        if sample_doc is None:
            raise ValueError("No document scores found in any entry.")

        self.feature_names = [k for k, v in sample_doc.items() 
                              if isinstance(v, (int, float)) and k != 'score']
        
        print(f"Detected Features to Optimize: {self.feature_names}")

        all_features = []
        doc_indices = []
        
        # We need to store which row belongs to which question to form pairs
        groups = {} # {question_index: [row_indices]}
        labels = {} # {row_index: 1 (correct) or 0 (incorrect)}

        global_idx = 0

        for q_idx, entry in enumerate(self.raw_data):
            correct_ids = set(entry.get('correct_documents', []))
            
            # Skip entries with no correct documents or no scores
            if not correct_ids or not entry.get('document_scores'):
                continue

            groups[q_idx] = []
            
            for doc in entry['document_scores']:
                # Extract features in specific order
                feats = [doc.get(f, 0.0) for f in self.feature_names]
                all_features.append(feats)
                
                # Determine label
                is_correct = 1 if doc['document_id'] in correct_ids else 0
                labels[global_idx] = is_correct
                
                groups[q_idx].append(global_idx)
                global_idx += 1

        # 2. Normalize Data
        # Optimization works much better if lexical (0-100) and dense (0-1) are on similar scales
        X = np.array(all_features)
        
        if len(X) == 0:
             raise ValueError("No valid feature data found to process.")

        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # 3. Generate Pairs (Positive vs Negative)
        # We want to maximize Score(Pos) - Score(Neg)
        idx_pos = []
        idx_neg = []

        for q_idx, indices in groups.items():
            pos_indices = [i for i in indices if labels[i] == 1]
            neg_indices = [i for i in indices if labels[i] == 0]

            # Create cartesian product of Correct vs Incorrect for this question
            for p in pos_indices:
                for n in neg_indices:
                    idx_pos.append(p)
                    idx_neg.append(n)

        return X_tensor, torch.tensor(idx_pos), torch.tensor(idx_neg)

# ---------------------------------------------------------
# 2. The Linear Ranking Model
# ---------------------------------------------------------

class LinearRanker(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # We use a Linear layer with NO bias. 
        # Ranking is purely about the weighted sum of features.
        self.weights = nn.Linear(num_features, 1, bias=False)
        
        # Initialize with positive weights explicitly if desired
        nn.init.uniform_(self.weights.weight, 0.1, 1.0)

    def forward(self, x):
        return self.weights(x)

# ---------------------------------------------------------
# 3. Optimization Logic
# ---------------------------------------------------------

def optimize_weights(json_data):
    try:
        processor = RankingDataProcessor(json_data)
        
        # Prepare data
        print("Processing data...")
        X, idx_pos, idx_neg = processor.process_documents()
    except ValueError as e:
        print(f"Skipping optimization: {e}")
        return None, None

    if len(idx_pos) == 0:
        print("No valid pairs found (e.g., all documents are correct or all are incorrect).")
        return None, None

    # Model Setup
    model = LinearRanker(num_features=X.shape[1])
    
    # MarginRankingLoss: 
    # Loss is low if (Input1 - Input2) > margin
    # We want PosScore > NegScore by at least 'margin'
    criterion = nn.MarginRankingLoss(margin=0.1) 
    
    # Optimizer (Adam is generally robust)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print(f"Training on {len(idx_pos)} pairwise comparisons...")

    # Training Loop
    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Get scores for all docs
        all_scores = model(X)
        
        # Extract scores for the pairs
        score_pos = all_scores[idx_pos]
        score_neg = all_scores[idx_neg]
        
        # Target is 1, meaning we want score_pos > score_neg
        target = torch.ones_like(score_pos)
        
        loss = criterion(score_pos, score_neg, target)
        
        loss.backward()
        optimizer.step()
        
        # Optional: Constraint to keep weights positive 
        # (Negative weights usually don't make sense for 'similarity' scores)
        with torch.no_grad():
            model.weights.weight.clamp_(min=0.0)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")

    # ---------------------------------------------------------
    # 4. Extract and Display Results
    # ---------------------------------------------------------
    learned_weights = model.weights.weight.detach().numpy()[0]
    
    # Because we scaled inputs, we must un-scale weights to apply them to raw data
    # Raw_Weight = Learned_Weight / Scale
    # (This is an approximation to get back to original magnitude relative to each other)
    raw_weights = learned_weights / processor.scaler.scale_
    
    # Normalize so the largest weight is roughly 10 or 1.0 for readability
    raw_weights = raw_weights / np.max(raw_weights) * 10.0

    print("\n" + "="*30)
    print("OPTIMIZED WEIGHTS")
    print("="*30)
    results = dict(zip(processor.feature_names, raw_weights))
    
    for k, v in results.items():
        print(f"{k:.<30} {v:.4f}")
        
    return results, processor


# ---------------------------------------------------------
# 5. Feature Impact Analysis
# ---------------------------------------------------------

def compute_feature_contributions(json_data: List[Dict], optimized_weights: Dict[str, float], feature_names: List[str]) -> Dict[str, float]:
    """
    Computes the total contribution of each feature across the complete dataset.
    For each document, calculates feature_value * weight for each feature,
    then sums these contributions across all documents.
    
    Args:
        json_data: Complete dataset with document scores
        optimized_weights: Dictionary mapping feature names to their optimized weights
        feature_names: List of feature names in the correct order
        
    Returns:
        Dictionary mapping feature names to their total contribution
    """
    feature_contributions: Dict[str, float] = {feature: 0.0 for feature in feature_names}
    
    total_documents = 0
    
    for entry in json_data:
        if not entry.get('document_scores'):
            continue
            
        for doc in entry['document_scores']:
            total_documents += 1
            
            for feature_name in feature_names:
                feature_value = doc.get(feature_name, 0.0)
                weight = optimized_weights.get(feature_name, 0.0)
                contribution = feature_value * weight
                feature_contributions[feature_name] += contribution
    
    print(f"\nProcessed {total_documents} documents for impact analysis")
    return feature_contributions


def visualize_feature_impact(feature_contributions: Dict[str, float], output_path: Optional[str] = None) -> None:
    """
    Creates a pie chart showing the relative impact of each feature on the dataset.
    
    Args:
        feature_contributions: Dictionary mapping feature names to their total contributions
        output_path: Optional path to save the figure. If None, displays interactively.
    """
    # Filter out features with zero or negative contributions
    positive_contributions = {k: max(0.0, v) for k, v in feature_contributions.items() if v > 0}
    
    if not positive_contributions:
        print("No positive contributions found. Cannot create pie chart.")
        return
    
    # Sort by contribution value (descending)
    sorted_features = sorted(positive_contributions.items(), key=lambda x: x[1], reverse=True)
    feature_names = [name.replace('_', ' ').title() for name, _ in sorted_features]
    contributions = [contrib for _, contrib in sorted_features]
    
    # Calculate percentages
    total = sum(contributions)
    percentages = [100 * contrib / total for contrib in contributions]
    
    # Create pie chart
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(range(len(feature_names)))
    
    wedges, texts, autotexts = plt.pie(
        contributions,
        labels=feature_names,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 10}
    )
    
    # Enhance text visibility
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    
    plt.title('Feature Impact on Dataset\n(Total Contribution: Weight × Score across all documents)', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.axis('equal')
    
    # Add legend with absolute values
    legend_labels = [f'{name}: {contrib:.2f}' for name, contrib in zip(feature_names, contributions)]
    plt.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPie chart saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


# ---------------------------------------------------------
# Example Usage
# ---------------------------------------------------------

if __name__ == "__main__":
    # Base directory containing the score subfolders
    base_scores_dir = os.path.join(os.path.dirname(__file__), "scores")
    
    # Specify which subdirectories to utilize. 
    # Add folder names here to include them in the optimization.
    # If you want to use ALL subfolders found in 'scores', you can set:
    # sub_dirs = [d for d in os.listdir(base_scores_dir) if os.path.isdir(os.path.join(base_scores_dir, d))]
    sub_dirs = [
        #"Local_HotpotQA_1k",
         #"HotpotQA_1k",
         # "MultihopRAG_old_config",
         #"Hotpot_1k_BFS",
         #"Dev_set_rankings"
         #"Scaling-DocAwareHybridRAG"
         "MultiHopRAG-Rankings"
    ]
    
    all_data = []
    total_files = 0
    
    print(f"Base Scores Directory: {base_scores_dir}")
    print(f"Selected Subdirectories: {sub_dirs}")

    for sub_dir in sub_dirs:
        dir_path = os.path.join(base_scores_dir, sub_dir)
        if not os.path.exists(dir_path):
            print(f"Warning: Directory not found: {dir_path}")
            continue

        # Find all json files in this subdirectory
        json_pattern = os.path.join(dir_path, "*.json")
        file_list = glob.glob(json_pattern)
        
        print(f"  -> Found {len(file_list)} JSON files in '{sub_dir}'")
        total_files += len(file_list)

        for file_path in file_list:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    # If the json is a list of items, extend. If it's a single object, append.
                    if isinstance(content, list):
                        all_data.extend(content)
                    else:
                        all_data.append(content)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    print(f"Total files processed: {total_files}")


    if all_data:
        print(f"Successfully loaded {len(all_data)} records. Starting optimization...")
        results, processor = optimize_weights(all_data)
        
        if results and processor:
            print("\n" + "="*30)
            print("COMPUTING FEATURE IMPACT ON COMPLETE DATASET")
            print("="*30)
            
            feature_contributions = compute_feature_contributions(
                all_data, 
                results, 
                processor.feature_names
            )
            
            print("\nFeature Contributions (Total: Score × Weight across all documents):")
            for feature, contribution in sorted(feature_contributions.items(), 
                                                  key=lambda x: x[1], 
                                                  reverse=True):
                print(f"{feature:.<30} {contribution:.2f}")
            
            # Create pie chart
            visualize_feature_impact(feature_contributions)
    else:
        print("No data loaded. Check directory path and file contents.")