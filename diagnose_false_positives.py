#!/usr/bin/env python3
"""
Diagnose why the 91.8% GNN produces false positives on real code.

This script tests the trained model on:
1. Obviously buggy code (should predict label 0/1 for buggy)
2. Obviously clean code (should predict label 0/1 for clean)
3. Real production code (should NOT predict everything as one class)
"""

import torch
from pathlib import Path
from nerion_digital_physicist.agent.data import create_graph_data_from_source
from nerion_digital_physicist.agent.semantics import get_global_embedder
from nerion_digital_physicist.agent.brain import MultiTaskCodeGraphSAGE
from torch_geometric.data import Batch

# Test cases: obviously buggy vs obviously clean code
TEST_CASES = [
    {
        "name": "Division by zero (BUGGY)",
        "code": """
def divide(a, b):
    return a / b  # BUG: No check for b == 0

result = divide(10, 0)  # Will crash
""",
        "expected": "buggy",
    },
    {
        "name": "Division by zero (FIXED)",
        "code": """
def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

result = divide(10, 2)  # Safe
""",
        "expected": "clean",
    },
    {
        "name": "SQL injection (BUGGY)",
        "code": """
def get_user(username):
    query = f"SELECT * FROM users WHERE name = '{username}'"  # BUG: SQL injection
    return db.execute(query)
""",
        "expected": "buggy",
    },
    {
        "name": "SQL injection (FIXED)",
        "code": """
def get_user(username):
    query = "SELECT * FROM users WHERE name = ?"
    return db.execute(query, (username,))  # Parameterized query
""",
        "expected": "clean",
    },
    {
        "name": "Null pointer (BUGGY)",
        "code": """
def process_user(user):
    return user.name.upper()  # BUG: user might be None
""",
        "expected": "buggy",
    },
    {
        "name": "Null pointer (FIXED)",
        "code": """
def process_user(user):
    if user is None:
        return None
    return user.name.upper()
""",
        "expected": "clean",
    },
    {
        "name": "Simple clean function",
        "code": """
def add(a, b):
    return a + b
""",
        "expected": "clean",
    },
    {
        "name": "Production-quality code",
        "code": """
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def fetch_data(url: str, timeout: int = 30) -> Optional[dict]:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch data from {url}: {e}")
        return None
""",
        "expected": "clean",
    },
]


def load_model(model_path: Path) -> MultiTaskCodeGraphSAGE:
    """Load the trained GNN model."""
    checkpoint = torch.load(model_path, weights_only=False)
    
    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        config = checkpoint.get('config', {})
    else:
        state_dict = checkpoint
        config = {}
    
    # Infer model configuration from state dict
    # Check first SAGE layer to get num_node_features
    first_sage_key = 'sage_layers.0.lin_l.weight'
    if first_sage_key in state_dict:
        num_node_features = state_dict[first_sage_key].shape[1]
    else:
        num_node_features = 800  # Default: 32 structural + 768 GraphCodeBERT
    
    # Check hidden channels from first layer output
    hidden_channels = state_dict['sage_layers.0.lin_l.weight'].shape[0]
    
    print(f"📊 Model configuration:")
    print(f"   num_node_features: {num_node_features}")
    print(f"   hidden_channels: {hidden_channels}")
    print(f"   use_graphcodebert: {num_node_features == 800}")
    
    # Create model
    model = MultiTaskCodeGraphSAGE(
        num_node_features=num_node_features,
        hidden_channels=hidden_channels,
        num_layers=4,
        residual=False,
        dropout=0.2,
        use_graphcodebert=(num_node_features == 800),
        freeze_backbone=False,
    )
    
    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model


def predict(model: MultiTaskCodeGraphSAGE, code: str) -> tuple[int, float]:
    """
    Predict if code is buggy or clean.
    
    Returns:
        (predicted_label, confidence)
        - predicted_label: 0 or 1
        - confidence: probability of predicted class
    """
    embedder = get_global_embedder()
    
    # Create graph
    graph = create_graph_data_from_source(code, embedder=embedder)
    
    # Batch it (required by PyG)
    batch = Batch.from_data_list([graph])
    
    # Extract GraphCodeBERT embedding if model uses it
    graphcodebert_embedding = None
    if hasattr(graph, 'graphcodebert_embedding') and graph.graphcodebert_embedding is not None:
        graphcodebert_embedding = graph.graphcodebert_embedding.unsqueeze(0)  # Add batch dim
    
    # Forward pass
    with torch.no_grad():
        logits = model(
            batch.x,
            batch.edge_index,
            batch.batch,
            task='bug',
            graphcodebert_embedding=graphcodebert_embedding
        )
    
    # Convert logits to probabilities
    if logits.dim() == 1:
        # Single output (BCEWithLogitsLoss format)
        prob = torch.sigmoid(logits).item()
        predicted_label = 1 if prob > 0.5 else 0
        confidence = prob if predicted_label == 1 else (1 - prob)
    else:
        # Two outputs (CrossEntropyLoss format)
        probs = torch.softmax(logits, dim=-1)
        predicted_label = probs.argmax(dim=-1).item()
        confidence = probs[0, predicted_label].item()
    
    return predicted_label, confidence


def main():
    print("=" * 80)
    print("🔬 DIAGNOSING FALSE POSITIVE PROBLEM")
    print("=" * 80)
    
    # Load model
    model_path = Path("models/digital_physicist_brain.pt")
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Please provide the path to your trained model.")
        return
    
    print(f"\n📦 Loading model from: {model_path}")
    model = load_model(model_path)
    print(f"✅ Model loaded successfully!\n")
    
    # Test on all cases
    print("=" * 80)
    print("🧪 TESTING ON OBVIOUS BUGS AND CLEAN CODE")
    print("=" * 80)
    
    results = []
    for test_case in TEST_CASES:
        name = test_case["name"]
        code = test_case["code"]
        expected = test_case["expected"]
        
        predicted_label, confidence = predict(model, code)
        
        # Interpret label (depends on training setup)
        # CRITICAL: We need to figure out what label 0 and label 1 mean
        predicted_class = "label_" + str(predicted_label)
        
        correct = "✅" if (
            (expected == "buggy" and predicted_label == 0) or
            (expected == "clean" and predicted_label == 1)
        ) else "❌"
        
        results.append({
            "name": name,
            "expected": expected,
            "predicted_label": predicted_label,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "correct": correct,
        })
        
        print(f"\n{correct} {name}")
        print(f"   Expected: {expected}")
        print(f"   Predicted: {predicted_class} (confidence: {confidence:.2%})")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    
    label_0_count = sum(1 for r in results if r["predicted_label"] == 0)
    label_1_count = sum(1 for r in results if r["predicted_label"] == 1)
    
    print(f"\nLabel distribution:")
    print(f"   Label 0: {label_0_count}/{len(results)} ({label_0_count/len(results)*100:.1f}%)")
    print(f"   Label 1: {label_1_count}/{len(results)} ({label_1_count/len(results)*100:.1f}%)")
    
    # Detect false positive problem
    if label_0_count == len(results):
        print(f"\n❌ FALSE POSITIVE PROBLEM DETECTED!")
        print(f"   Model predicts EVERYTHING as label 0")
        print(f"   This means it learned to always predict one class.")
    elif label_1_count == len(results):
        print(f"\n❌ FALSE POSITIVE PROBLEM DETECTED!")
        print(f"   Model predicts EVERYTHING as label 1")
        print(f"   This means it learned to always predict one class.")
    else:
        print(f"\n✅ Model shows some discrimination between classes")
    
    # Explain the root cause
    print("\n" + "=" * 80)
    print("🔍 ROOT CAUSE ANALYSIS")
    print("=" * 80)
    print("""
The training setup has a fundamental flaw:

CURRENT TRAINING:
- Label 0 = before_code (buggy version)
- Label 1 = after_code (fixed version)
- Task: "Is this the before or after version?"

PROBLEM:
- The GNN learned to distinguish "before vs after" by looking at:
  * Code length (after is usually longer)
  * Complexity (after has more error handling)
  * GraphCodeBERT embeddings (semantic difference)
- But this is NOT the same as "is this code buggy?"

WHEN TESTING ON REAL CODE:
- There's no "before/after" pair
- The code is just... code
- The GNN defaults to one class because it has no reference point

SOLUTION:
- Shuffle training data to break before/after pairing
- Train on mixed buggy/clean code in random order
- GNN learns actual bug patterns, not version differences
""")


if __name__ == "__main__":
    main()

