#!/usr/bin/env python3
"""
Batch script to process all HotpotQA datasets with the QA parser.

This script will update the QA.json files in all HotpotQA dataset directories
with the correct supporting facts from the distractor JSON.
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Base directory containing all HotpotQA datasets
    base_dir = Path("../../data")
    
    # Find all HotpotQA dataset directories
    hotpot_dirs = []
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.startswith("HotpotQA_"):
            qa_file = item / "QA.json"
            if qa_file.exists():
                hotpot_dirs.append(item)
    
    if not hotpot_dirs:
        print("❌ No HotpotQA dataset directories found in", base_dir)
        sys.exit(1)
    
    print(f"🔍 Found {len(hotpot_dirs)} HotpotQA datasets:")
    for dir_path in hotpot_dirs:
        print(f"  - {dir_path.name}")
    
    print("\n🚀 Processing all datasets...")
    
    # Process each dataset
    for i, dataset_dir in enumerate(hotpot_dirs, 1):
        print(f"\n📁 [{i}/{len(hotpot_dirs)}] Processing {dataset_dir.name}")
        
        try:
            # Run the QA parser
            result = subprocess.run([
                sys.executable, "qa_parser.py", str(dataset_dir)
            ], capture_output=True, text=True, cwd=Path(__file__).parent)
            
            if result.returncode == 0:
                print(f"✅ Successfully processed {dataset_dir.name}")
                # Print the summary from the output
                for line in result.stdout.split('\n'):
                    if line.startswith('📊 Summary:'):
                        print(f"   {line}")
            else:
                print(f"❌ Error processing {dataset_dir.name}:")
                print(f"   {result.stderr}")
                
        except Exception as e:
            print(f"❌ Exception processing {dataset_dir.name}: {e}")
    
    print(f"\n🎉 Finished processing all {len(hotpot_dirs)} datasets!")

if __name__ == "__main__":
    main()
