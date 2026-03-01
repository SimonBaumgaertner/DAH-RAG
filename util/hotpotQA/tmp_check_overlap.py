import json
import os

try:
    with open('data/HotpotQA_10k/QA.json', 'r') as f:
        qa_10k = json.load(f)
    print(f"Loaded {len(qa_10k)} from 10k set")
    
    ids_10k = set(q['_id'] if '_id' in q else q['question_id'] for q in qa_10k)
    print(f"Unique IDs in 10k: {len(ids_10k)}")

    path_dev = 'util/hotpotQA/hotpot_dev_distractor_v1.json'
    if os.path.exists(path_dev):
        with open(path_dev, 'r') as f:
            dev_full = json.load(f)
        print(f"Loaded {len(dev_full)} from official dev set")
        
        ids_dev = set(q['_id'] if '_id' in q else q['question_id'] for q in dev_full)
        
        overlap = ids_10k.intersection(ids_dev)
        print(f"Overlap count: {len(overlap)}")
        
        if len(overlap) > 0:
            print("Conclusion: 10k set IS a subset of official Dev")
        else:
            print("Conclusion: 10k set is DISJOINT from official Dev (probably from Train)")
    else:
        print("Official dev set not found at expected path")

except Exception as e:
    print(f"Error: {e}")
