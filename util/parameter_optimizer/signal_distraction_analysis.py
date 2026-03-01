import json
from pathlib import Path
from collections import defaultdict

# CONFIGURATION
DATASETS = [
    "PubMedQA_10k-Rankings",
    "HotpotQA_1k-Rankings",
    "NovelQA-Rankings",
    "MultiHopRAG-Rankings"
]

OPTIMIZED_WEIGHTS_PER_DATASET = {
    "PubMedQA_10k-Rankings": {
        "filter_lexical_weight": 5.0827,
        "filter_dense_weight": 0.0000,
        "filter_entity_weight": 0.0000,
        "rank_lexical_weight": 0.0128,
        "rank_dense_weight": 10.0000,
        "rank_ppr_weight": 0.0081
    },
    "HotpotQA_1k-Rankings": {
        "filter_lexical_weight": 1.6815,
        "filter_dense_weight": 10.0000,
        "filter_entity_weight": 0.0000,
        "rank_lexical_weight": 0.2245,
        "rank_dense_weight": 10.0000,
        "rank_ppr_weight": 0.6989
    },
    "NovelQA-Rankings": {
        "filter_lexical_weight": 0.7104,
        "filter_dense_weight": 1.9198,
        "filter_entity_weight": 1.2719,
        "rank_lexical_weight": 0.3234,
        "rank_dense_weight": 10.0000,
        "rank_ppr_weight": 0.3259
    },
    "MultiHopRAG-Rankings": {
        "filter_lexical_weight": 0.2929,
        "filter_dense_weight": 2.9763,
        "filter_entity_weight": 0.1593,
        "rank_lexical_weight": 0.1766,
        "rank_dense_weight": 10.0000,
        "rank_ppr_weight": 0.0744
    }
}

def analyze_dataset(folder_name, use_optimized_weights=False):
    base_dir = Path(__file__).parent / "scores"
    directory_path = base_dir / folder_name
    
    if not directory_path.is_dir():
        # print(f"Warning: {directory_path} is not a directory.")
        return None

    json_files = list(directory_path.glob("*.json"))
    if not json_files:
        return None

    doc_contributions = defaultdict(list)
    chunk_contributions = defaultdict(list)

    doc_mapping = {
        "filter_lexical_score": "filter_lexical_weight",
        "filter_dense_score": "filter_dense_weight",
        "filter_entity_score": "filter_entity_weight"
    }

    chunk_mapping = {
        "lexical_score": "rank_lexical_weight",
        "dense_score": "rank_dense_weight",
        "ppr_score": "rank_ppr_weight"
    }

    for json_file in json_files:
        with open(json_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
            if use_optimized_weights:
                weights = OPTIMIZED_WEIGHTS_PER_DATASET.get(folder_name, {})
            else:
                weights = data.get("weights", {})
            
            # Document Analysis
            document_scores = data.get("document_scores", [])
            for doc in document_scores:
                total_weighted_sum = sum(doc.get(score_key, 0) * weights.get(weight_key, 0) 
                                         for score_key, weight_key in doc_mapping.items())
                
                if total_weighted_sum > 0:
                    for score_key, weight_key in doc_mapping.items():
                        weighted_val = doc.get(score_key, 0) * weights.get(weight_key, 0)
                        doc_contributions[score_key].append(weighted_val / total_weighted_sum)

            # Chunk Analysis (Top 5)
            chunk_scores = data.get("chunk_scores", [])
            sorted_chunks = sorted(chunk_scores, key=lambda x: x.get("score", 0), reverse=True)
            top_5_chunks = sorted_chunks[:5]

            for chunk in top_5_chunks:
                total_weighted_sum = sum(chunk.get(score_key, 0) * weights.get(weight_key, 0) 
                                         for score_key, weight_key in chunk_mapping.items())
                
                if total_weighted_sum > 0:
                    for score_key, weight_key in chunk_mapping.items():
                        weighted_val = chunk.get(score_key, 0) * weights.get(weight_key, 0)
                        chunk_contributions[score_key].append(weighted_val / total_weighted_sum)

    results = {}
    if doc_contributions:
        results['doc'] = {k: sum(v)/len(v) for k, v in doc_contributions.items()}
    if chunk_contributions:
        results['chunk'] = {k: sum(v)/len(v) for k, v in chunk_contributions.items()}
    
    return results

def print_table(all_results):
    print("\n" + "="*80)
    print(f"{'Dataset':<25} | {'Doc Lex':<7} | {'Doc Dense':<7} | {'Doc Ent':<7} || {'Chu Lex':<7} | {'Chu Dense':<7} | {'Chu PPR':<7}")
    print("-" * 80)
    
    for dataset, res in all_results.items():
        doc = res.get('doc', {})
        chunk = res.get('chunk', {})
        
        d_lex = f"{doc.get('filter_lexical_score', 0)*100:5.1f}%"
        d_den = f"{doc.get('filter_dense_score', 0)*100:5.1f}%"
        d_ent = f"{doc.get('filter_entity_score', 0)*100:5.1f}%"
        
        c_lex = f"{chunk.get('lexical_score', 0)*100:5.1f}%"
        c_den = f"{chunk.get('dense_score', 0)*100:5.1f}%"
        c_ppr = f"{chunk.get('ppr_score', 0)*100:5.1f}%"
        
        print(f"{dataset:<25} | {d_lex:<7} | {d_den:<7} | {d_ent:<7} || {c_lex:<7} | {c_den:<7} | {c_ppr:<7}")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    all_results_original = {}
    all_results_optimized = {}

    for dataset in DATASETS:
        print(f"Analyzing {dataset}...")
        res_orig = analyze_dataset(dataset, use_optimized_weights=False)
        if res_orig:
            all_results_original[dataset] = res_orig
            
        res_opt = analyze_dataset(dataset, use_optimized_weights=True)
        if res_opt:
            all_results_optimized[dataset] = res_opt
    
    if all_results_original:
        print("=== USING ORIGINAL WEIGHTS (FROM JSON) ===")
        print_table(all_results_original)
    else:
        print("No results found.")
        
    if all_results_optimized:
        print("=== USING OPTIMIZED WEIGHTS (PER DATASET) ===")
        print_table(all_results_optimized)
