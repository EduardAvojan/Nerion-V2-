#!/usr/bin/env python3
"""
Verify that dataset has proper metadata for lesson-level splitting.

This script checks:
1. Do graphs have sample_meta['lesson'] field?
2. How many graphs per lesson?
3. Are there exactly 2 graphs per lesson (before/after)?
4. What's the label distribution?
"""

import torch
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

def verify_dataset(dataset_path: Path):
    """Verify dataset has proper metadata for lesson-level splitting."""
    
    print("=" * 80)
    print("🔍 VERIFYING DATASET METADATA")
    print("=" * 80)
    
    # Load dataset
    print(f"\n📦 Loading dataset from: {dataset_path}")
    try:
        data = torch.load(dataset_path, weights_only=False)
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False
    
    if 'samples' in data:
        graphs = data['samples']
    else:
        graphs = data
    
    print(f"✅ Loaded {len(graphs)} graphs")
    
    # Check metadata on first graph
    print(f"\n📊 Checking first graph metadata...")
    if len(graphs) == 0:
        print(f"❌ Dataset is empty!")
        return False
    
    first_graph = graphs[0]
    
    if not hasattr(first_graph, 'sample_meta'):
        print(f"❌ CRITICAL: Graphs have NO sample_meta attribute!")
        print(f"   The fix_data_leakage.py script will NOT work.")
        print(f"   You need to rebuild the dataset with dataset_builder.py")
        return False
    
    print(f"✅ Graphs have sample_meta attribute")
    print(f"   First graph metadata: {first_graph.sample_meta}")
    
    if 'lesson' not in first_graph.sample_meta:
        print(f"❌ CRITICAL: sample_meta missing 'lesson' field!")
        print(f"   The fix_data_leakage.py script will NOT work.")
        print(f"   You need to rebuild the dataset with dataset_builder.py")
        return False
    
    print(f"✅ Graphs have sample_meta['lesson'] field")
    
    # Group graphs by lesson
    print(f"\n📊 Analyzing lesson distribution...")
    lessons: Dict[str, List] = defaultdict(list)
    missing_metadata_count = 0
    
    for i, graph in enumerate(graphs):
        if hasattr(graph, 'sample_meta') and 'lesson' in graph.sample_meta:
            lesson_id = graph.sample_meta['lesson']
            lessons[lesson_id].append(graph)
        else:
            missing_metadata_count += 1
            print(f"   ⚠️ Graph {i} missing metadata")
    
    if missing_metadata_count > 0:
        print(f"\n⚠️ WARNING: {missing_metadata_count}/{len(graphs)} graphs missing metadata")
        print(f"   These will be treated as separate lessons (suboptimal)")
    
    print(f"\n✅ Found {len(lessons)} unique lessons")
    
    # Analyze graphs per lesson
    lesson_sizes = [len(graphs) for graphs in lessons.values()]
    
    print(f"\n📊 Graphs per lesson:")
    print(f"   Average: {sum(lesson_sizes) / len(lesson_sizes):.2f}")
    print(f"   Min: {min(lesson_sizes)}")
    print(f"   Max: {max(lesson_sizes)}")
    
    # Count lessons by size
    size_distribution = defaultdict(int)
    for size in lesson_sizes:
        size_distribution[size] += 1
    
    print(f"\n📊 Lesson size distribution:")
    for size in sorted(size_distribution.keys()):
        count = size_distribution[size]
        percentage = count / len(lessons) * 100
        print(f"   {size} graphs: {count} lessons ({percentage:.1f}%)")
    
    # Check if most lessons have exactly 2 graphs (before/after)
    lessons_with_2_graphs = size_distribution.get(2, 0)
    if lessons_with_2_graphs / len(lessons) > 0.8:
        print(f"\n✅ GOOD: {lessons_with_2_graphs}/{len(lessons)} lessons have exactly 2 graphs (before/after)")
    else:
        print(f"\n⚠️ WARNING: Only {lessons_with_2_graphs}/{len(lessons)} lessons have 2 graphs")
        print(f"   Expected most lessons to have exactly 2 graphs (before/after)")
    
    # Check label distribution
    print(f"\n📊 Label distribution:")
    label_0_count = sum(1 for g in graphs if int(g.y.item()) == 0)
    label_1_count = sum(1 for g in graphs if int(g.y.item()) == 1)
    
    print(f"   Label 0 (before_code): {label_0_count} ({label_0_count/len(graphs)*100:.1f}%)")
    print(f"   Label 1 (after_code): {label_1_count} ({label_1_count/len(graphs)*100:.1f}%)")
    
    if abs(label_0_count - label_1_count) / len(graphs) < 0.1:
        print(f"   ✅ GOOD: Labels are roughly balanced")
    else:
        print(f"   ⚠️ WARNING: Labels are imbalanced")
    
    # Sample a few lessons to show structure
    print(f"\n📊 Sample lessons (first 5):")
    for i, (lesson_id, lesson_graphs) in enumerate(list(lessons.items())[:5]):
        print(f"\n   Lesson {i+1}: {lesson_id}")
        print(f"      Graphs: {len(lesson_graphs)}")
        for j, graph in enumerate(lesson_graphs):
            label = int(graph.y.item())
            sample_type = graph.sample_meta.get('sample_type', 'unknown')
            print(f"         Graph {j+1}: label={label}, type={sample_type}, nodes={graph.num_nodes}")
    
    # Final verdict
    print(f"\n" + "=" * 80)
    print("📋 VERDICT")
    print("=" * 80)
    
    all_good = True
    
    if missing_metadata_count > 0:
        print(f"⚠️ {missing_metadata_count} graphs missing metadata (suboptimal)")
        all_good = False
    else:
        print(f"✅ All graphs have proper metadata")
    
    if lessons_with_2_graphs / len(lessons) > 0.8:
        print(f"✅ Most lessons have exactly 2 graphs (before/after)")
    else:
        print(f"⚠️ Unexpected lesson structure (not all have 2 graphs)")
        all_good = False
    
    if abs(label_0_count - label_1_count) / len(graphs) < 0.1:
        print(f"✅ Labels are balanced")
    else:
        print(f"⚠️ Labels are imbalanced")
    
    print(f"\n" + "=" * 80)
    if all_good:
        print(f"✅ DATASET IS READY FOR fix_data_leakage.py")
        print(f"\nNext step:")
        print(f"   python3 fix_data_leakage.py \\")
        print(f"       --input {dataset_path} \\")
        print(f"       --output experiments/datasets/gnn/fixed \\")
        print(f"       --val-ratio 0.15")
    else:
        print(f"⚠️ DATASET HAS ISSUES - Consider rebuilding")
        print(f"\nRecommended:")
        print(f"   python3 -m nerion_digital_physicist.training.dataset_builder \\")
        print(f"       --db out/learning/curriculum.sqlite \\")
        print(f"       --output-dir experiments/datasets/gnn/rebuilt \\")
        print(f"       --mode supervised")
    print("=" * 80)
    
    return all_good


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify dataset has proper metadata")
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to dataset.pt file to verify"
    )
    
    args = parser.parse_args()
    
    if not args.dataset.exists():
        print(f"❌ Dataset not found: {args.dataset}")
        print("\nUsage:")
        print(f"   python3 verify_dataset_metadata.py <path_to_dataset.pt>")
        print("\nExample:")
        print(f"   python3 verify_dataset_metadata.py experiments/datasets/gnn/raw/supervised/*/dataset.pt")
        return
    
    verify_dataset(args.dataset)


if __name__ == "__main__":
    main()

