#!/usr/bin/env python3
"""Test GraphCodeBERT integration with GNN training pipeline."""

from pathlib import Path
import sys

print("=" * 70)
print("Testing GraphCodeBERT Integration")
print("=" * 70)

# Test 1: Load GraphCodeBERT embeddings
print("\n1. Testing GraphCodeBERT loader...")
try:
    from nerion_digital_physicist.agent.graphcodebert_loader import load_graphcodebert_embeddings, get_lesson_embedding

    embeddings, id_map, name_map = load_graphcodebert_embeddings()
    print(f"   ✅ Loaded {len(id_map)} lesson embeddings")
    print(f"   ✅ Embedding dimension: {embeddings['dimension']}")
    print(f"   ✅ Model: {embeddings['model']}")

    # Test retrieval
    test_emb = get_lesson_embedding(lesson_id=1, sample_type="before")
    if test_emb and len(test_emb) == 768:
        print(f"   ✅ Retrieved embedding for lesson ID 1: {len(test_emb)}-dimensional vector")
    else:
        print(f"   ❌ Failed to retrieve embedding for lesson ID 1")
        sys.exit(1)

except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Test 2: Check dataset builder updates
print("\n2. Testing dataset builder...")
try:
    from nerion_digital_physicist.training.dataset_builder import _annotate_graph
    from torch_geometric.data import Data
    import torch

    # Create dummy graph
    x = torch.randn(10, 32)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    dummy_graph = Data(x=x, edge_index=edge_index)

    # Annotate with GraphCodeBERT embedding
    annotated = _annotate_graph(dummy_graph, label=0, lesson="idempotent_api_request_handling", sample_type="before")

    if hasattr(annotated, 'graphcodebert_embedding'):
        if annotated.graphcodebert_embedding.shape[0] == 768:
            print(f"   ✅ GraphCodeBERT embedding attached: {annotated.graphcodebert_embedding.shape}")
        else:
            print(f"   ❌ Wrong embedding dimension: {annotated.graphcodebert_embedding.shape}")
            sys.exit(1)
    else:
        print(f"   ❌ GraphCodeBERT embedding not attached to graph")
        sys.exit(1)

except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check GNN model accepts GraphCodeBERT
print("\n3. Testing GNN models...")
try:
    from nerion_digital_physicist.agent.brain import build_gnn
    import torch

    model = build_gnn(
        architecture="sage",
        num_node_features=32,
        hidden_channels=64,
        num_classes=2,
        use_graphcodebert=True,
    )

    # Create dummy batch
    x = torch.randn(20, 32)
    edge_index = torch.tensor([[0, 1, 2, 10, 11], [1, 2, 3, 11, 12]], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    graphcodebert_emb = torch.randn(2, 768)  # 2 graphs in batch

    # Forward pass
    out = model(x, edge_index, batch, graphcodebert_embedding=graphcodebert_emb)

    if out.shape == (2, 2):  # 2 graphs, 2 classes
        print(f"   ✅ Model forward pass successful: output shape {out.shape}")
    else:
        print(f"   ❌ Unexpected output shape: {out.shape}")
        sys.exit(1)

except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Check training config
print("\n4. Testing training configuration...")
try:
    from nerion_digital_physicist.training.run_training import TrainingConfig
    from pathlib import Path

    config = TrainingConfig(
        dataset_path=Path("dummy.pt"),
        output_dir=Path("out"),
        use_graphcodebert=True,
    )

    if config.use_graphcodebert:
        print(f"   ✅ TrainingConfig supports use_graphcodebert flag")
    else:
        print(f"   ❌ TrainingConfig use_graphcodebert not set")
        sys.exit(1)

except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print("\nGraphCodeBERT integration is ready for training.")
print("\nNext steps:")
print("1. Regenerate dataset with GraphCodeBERT embeddings:")
print("   python3 -m nerion_digital_physicist.training.dataset_builder")
print()
print("2. Train with GraphCodeBERT embeddings:")
print("   python3 -m nerion_digital_physicist.training.run_training \\")
print("       --dataset experiments/datasets/gnn/*/dataset.pt \\")
print("       --architecture sage \\")
print("       --use-graphcodebert \\")
print("       --epochs 50")
print("=" * 70)
