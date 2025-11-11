#!/usr/bin/env python3
"""
FIX THE FALSE POSITIVE PROBLEM

Root cause: GNN learned "before vs after" instead of "buggy vs clean"
Solution: Shuffle training data to break temporal pairing

This script:
1. Loads your existing dataset
2. Shuffles all graphs randomly (breaks before/after pairing)
3. Saves the shuffled dataset for retraining
"""

import torch
import random
from pathlib import Path
from typing import List

def fix_dataset(input_path: Path, output_path: Path, seed: int = 42):
    """
    Load dataset, shuffle to break pairing, and save.
    
    Args:
        input_path: Path to original dataset.pt
        output_path: Path to save shuffled dataset.pt
        seed: Random seed for reproducibility
    """
    print("=" * 80)
    print("🔧 FIXING TRAINING DATA PAIRING PROBLEM")
    print("=" * 80)
    
    # Load dataset
    print(f"\n📦 Loading dataset from: {input_path}")
    data = torch.load(input_path, weights_only=False)
    
    if 'samples' in data:
        graphs = data['samples']
    else:
        graphs = data
    
    print(f"✅ Loaded {len(graphs)} graphs")
    
    # Analyze label distribution BEFORE shuffling
    print("\n📊 BEFORE SHUFFLING:")
    label_0_count = sum(1 for g in graphs if int(g.y.item()) == 0)
    label_1_count = sum(1 for g in graphs if int(g.y.item()) == 1)
    print(f"   Label 0 (before_code): {label_0_count} ({label_0_count/len(graphs)*100:.1f}%)")
    print(f"   Label 1 (after_code): {label_1_count} ({label_1_count/len(graphs)*100:.1f}%)")
    
    # Check if data is paired (every 2 consecutive graphs are from same lesson)
    paired_count = 0
    for i in range(0, len(graphs) - 1, 2):
        if hasattr(graphs[i], 'sample_meta') and hasattr(graphs[i+1], 'sample_meta'):
            if graphs[i].sample_meta['lesson'] == graphs[i+1].sample_meta['lesson']:
                paired_count += 1
    
    pairing_ratio = paired_count / (len(graphs) // 2) if len(graphs) > 1 else 0
    print(f"\n🔗 Pairing analysis:")
    print(f"   {paired_count}/{len(graphs)//2} consecutive pairs from same lesson ({pairing_ratio*100:.1f}%)")
    
    if pairing_ratio > 0.8:
        print(f"   ❌ HIGHLY PAIRED DATA DETECTED!")
        print(f"   This is why your GNN learned 'before vs after' instead of 'buggy vs clean'")
    
    # Shuffle to break pairing
    print(f"\n🔀 Shuffling {len(graphs)} graphs (seed={seed})...")
    random.seed(seed)
    random.shuffle(graphs)
    print(f"✅ Shuffled successfully!")
    
    # Verify pairing is broken
    paired_count_after = 0
    for i in range(0, len(graphs) - 1, 2):
        if hasattr(graphs[i], 'sample_meta') and hasattr(graphs[i+1], 'sample_meta'):
            if graphs[i].sample_meta['lesson'] == graphs[i+1].sample_meta['lesson']:
                paired_count_after += 1
    
    pairing_ratio_after = paired_count_after / (len(graphs) // 2) if len(graphs) > 1 else 0
    print(f"\n📊 AFTER SHUFFLING:")
    print(f"   {paired_count_after}/{len(graphs)//2} consecutive pairs from same lesson ({pairing_ratio_after*100:.1f}%)")
    
    if pairing_ratio_after < 0.2:
        print(f"   ✅ PAIRING SUCCESSFULLY BROKEN!")
        print(f"   GNN will now learn actual bug patterns, not version differences")
    else:
        print(f"   ⚠️ WARNING: Some pairing still remains (expected <20%, got {pairing_ratio_after*100:.1f}%)")
    
    # Verify label distribution unchanged
    label_0_count_after = sum(1 for g in graphs if int(g.y.item()) == 0)
    label_1_count_after = sum(1 for g in graphs if int(g.y.item()) == 1)
    print(f"\n📊 Label distribution (should be unchanged):")
    print(f"   Label 0: {label_0_count_after} ({label_0_count_after/len(graphs)*100:.1f}%)")
    print(f"   Label 1: {label_1_count_after} ({label_1_count_after/len(graphs)*100:.1f}%)")
    
    assert label_0_count == label_0_count_after, "Label 0 count changed!"
    assert label_1_count == label_1_count_after, "Label 1 count changed!"
    print(f"   ✅ Label distribution verified (unchanged)")
    
    # Save shuffled dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n💾 Saving shuffled dataset to: {output_path}")
    torch.save({'samples': graphs}, output_path)
    print(f"✅ Saved successfully!")
    
    # Instructions
    print("\n" + "=" * 80)
    print("📋 NEXT STEPS")
    print("=" * 80)
    print(f"""
1. Retrain your GNN with the shuffled dataset:
   
   python3 -m nerion_digital_physicist.training.train_multitask_ewc \\
       --dataset {output_path} \\
       --epochs 50 \\
       --batch-size 32

2. The GNN will now learn:
   - Actual bug patterns (not before/after differences)
   - Structural features that indicate bugs
   - Semantic patterns from GraphCodeBERT

3. Expected improvements:
   - No more false positives on real code
   - Better generalization to unseen bugs
   - Actual 90%+ accuracy on bug detection (not version classification)

4. Validation:
   - Test on obviously buggy code (should predict label 0)
   - Test on obviously clean code (should predict label 1)
   - Model should NOT predict everything as one class
""")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix training data pairing problem")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to original dataset.pt (with paired before/after)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save shuffled dataset.pt"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)"
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"❌ Input dataset not found: {args.input}")
        print("\nPlease provide the path to your training dataset.")
        print("Example: experiments/datasets/gnn/final_complete/supervised/*/dataset.pt")
        return
    
    fix_dataset(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()

