#!/usr/bin/env python3
"""
FIX DATA LEAKAGE PROBLEM

Root cause: Random split puts lesson_1_before in train and lesson_1_after in val
Solution: Split by lesson, not by individual graphs

This ensures:
1. If lesson_1_before is in train, lesson_1_after is ALSO in train
2. Validation set contains COMPLETELY UNSEEN lessons
3. GNN learns bug patterns, not lesson recognition
"""

import torch
import random
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

def fix_data_leakage(input_path: Path, output_dir: Path, val_ratio: float = 0.15, seed: int = 42):
    """
    Load dataset, split by lesson (not by graph), shuffle, and save.
    
    Args:
        input_path: Path to original dataset.pt
        output_dir: Directory to save train.pt and val.pt
        val_ratio: Fraction of lessons for validation (default: 0.15)
        seed: Random seed for reproducibility
    """
    print("=" * 80)
    print("🔧 FIXING DATA LEAKAGE PROBLEM")
    print("=" * 80)
    
    # Load dataset
    print(f"\n📦 Loading dataset from: {input_path}")
    data = torch.load(input_path, weights_only=False)
    
    if 'samples' in data:
        graphs = data['samples']
    else:
        graphs = data
    
    print(f"✅ Loaded {len(graphs)} graphs")
    
    # Step 1: Group graphs by lesson
    print(f"\n📊 Grouping graphs by lesson...")
    lessons: Dict[str, List] = defaultdict(list)
    
    for graph in graphs:
        if hasattr(graph, 'sample_meta') and 'lesson' in graph.sample_meta:
            lesson_id = graph.sample_meta['lesson']
            lessons[lesson_id].append(graph)
        else:
            # Fallback: If no lesson metadata, treat each graph as separate lesson
            # This prevents leakage but is suboptimal
            print(f"   ⚠️ WARNING: Graph missing lesson metadata, treating as separate lesson")
            lessons[f"unknown_{id(graph)}"].append(graph)
    
    print(f"✅ Found {len(lessons)} unique lessons")
    
    # Analyze lesson sizes
    lesson_sizes = [len(graphs) for graphs in lessons.values()]
    avg_size = sum(lesson_sizes) / len(lesson_sizes)
    print(f"   Average graphs per lesson: {avg_size:.1f}")
    print(f"   Min: {min(lesson_sizes)}, Max: {max(lesson_sizes)}")
    
    # Check for leakage in original split (if we had one)
    print(f"\n🔍 Checking for data leakage in original dataset...")
    # We can't check the original split without knowing how it was done,
    # but we can warn if lessons have both before and after
    lessons_with_both = sum(1 for graphs in lessons.values() if len(graphs) >= 2)
    print(f"   {lessons_with_both}/{len(lessons)} lessons have 2+ graphs (before/after pairs)")
    print(f"   If these were split randomly, there's {lessons_with_both/len(lessons)*100:.1f}% chance of leakage")
    
    # Step 2: Split lessons into train/val
    print(f"\n🔀 Splitting lessons into train/val ({1-val_ratio:.0%}/{val_ratio:.0%})...")
    random.seed(seed)
    lesson_ids = list(lessons.keys())
    random.shuffle(lesson_ids)
    
    val_size = int(len(lesson_ids) * val_ratio)
    train_size = len(lesson_ids) - val_size
    
    train_lesson_ids = lesson_ids[:train_size]
    val_lesson_ids = lesson_ids[train_size:]
    
    print(f"✅ Split complete:")
    print(f"   Train: {train_size} lessons")
    print(f"   Val: {val_size} lessons")
    
    # Step 3: Create train/val datasets
    print(f"\n📦 Creating train/val datasets...")
    train_graphs = []
    for lesson_id in train_lesson_ids:
        train_graphs.extend(lessons[lesson_id])
    
    val_graphs = []
    for lesson_id in val_lesson_ids:
        val_graphs.extend(lessons[lesson_id])
    
    print(f"✅ Datasets created:")
    print(f"   Train: {len(train_graphs)} graphs from {len(train_lesson_ids)} lessons")
    print(f"   Val: {len(val_graphs)} graphs from {len(val_lesson_ids)} lessons")
    
    # Verify no leakage
    print(f"\n🔒 Verifying no data leakage...")
    train_lesson_set = set(train_lesson_ids)
    val_lesson_set = set(val_lesson_ids)
    overlap = train_lesson_set & val_lesson_set
    
    if overlap:
        print(f"   ❌ ERROR: {len(overlap)} lessons appear in BOTH train and val!")
        print(f"   This should never happen. Aborting.")
        return
    else:
        print(f"   ✅ ZERO overlap between train and val lessons")
        print(f"   Validation set contains COMPLETELY UNSEEN lessons")
    
    # Step 4: Shuffle within each split
    print(f"\n🔀 Shuffling graphs within train/val splits...")
    random.shuffle(train_graphs)
    random.shuffle(val_graphs)
    print(f"✅ Shuffled successfully!")
    
    # Analyze label distribution
    print(f"\n📊 Label distribution:")
    train_label_0 = sum(1 for g in train_graphs if int(g.y.item()) == 0)
    train_label_1 = sum(1 for g in train_graphs if int(g.y.item()) == 1)
    val_label_0 = sum(1 for g in val_graphs if int(g.y.item()) == 0)
    val_label_1 = sum(1 for g in val_graphs if int(g.y.item()) == 1)
    
    print(f"   Train: {train_label_0} label 0 ({train_label_0/len(train_graphs)*100:.1f}%), "
          f"{train_label_1} label 1 ({train_label_1/len(train_graphs)*100:.1f}%)")
    print(f"   Val:   {val_label_0} label 0 ({val_label_0/len(val_graphs)*100:.1f}%), "
          f"{val_label_1} label 1 ({val_label_1/len(val_graphs)*100:.1f}%)")
    
    # Step 5: Save train/val datasets
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.pt"
    val_path = output_dir / "val.pt"
    
    print(f"\n💾 Saving datasets...")
    torch.save({'samples': train_graphs}, train_path)
    print(f"   ✅ Train: {train_path}")
    
    torch.save({'samples': val_graphs}, val_path)
    print(f"   ✅ Val: {val_path}")
    
    # Save metadata
    metadata = {
        'total_lessons': len(lessons),
        'train_lessons': len(train_lesson_ids),
        'val_lessons': len(val_lesson_ids),
        'train_graphs': len(train_graphs),
        'val_graphs': len(val_graphs),
        'val_ratio': val_ratio,
        'seed': seed,
        'no_leakage': True,
    }
    
    import json
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ Metadata: {metadata_path}")
    
    # Instructions
    print("\n" + "=" * 80)
    print("📋 NEXT STEPS")
    print("=" * 80)
    print(f"""
1. Retrain your GNN with the fixed datasets:
   
   python3 -m nerion_digital_physicist.training.train_multitask_ewc \\
       --train-dataset {train_path} \\
       --val-dataset {val_path} \\
       --epochs 50 \\
       --batch-size 32

2. The GNN will now:
   - Learn from COMPLETELY SEPARATE lessons in train vs val
   - NOT recognize validation lessons (they're unseen)
   - Learn actual bug patterns (not lesson recognition)
   - Generalize to real production code

3. Expected results:
   - Validation accuracy will DROP (maybe 60-75%)
   - This is GOOD - it means the task is harder (no cheating)
   - But the model will ACTUALLY WORK on real code
   - No more false positives

4. Validation:
   - Test on production code (truly unseen)
   - Model should discriminate buggy vs clean
   - NOT predict everything as one class
""")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix data leakage by splitting at lesson level")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to original dataset.pt (with leakage)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to save train.pt and val.pt"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of lessons for validation (default: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting (default: 42)"
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"❌ Input dataset not found: {args.input}")
        print("\nPlease provide the path to your training dataset.")
        return
    
    fix_data_leakage(args.input, args.output, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()

