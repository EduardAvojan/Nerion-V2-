"""Phase 2 training loop: fit the GNN brain on toy graph data."""

from __future__ import annotations

import os
from tempfile import NamedTemporaryFile

import torch
from torch_geometric.data import Data

from brain_v2 import CodeGraphNN
from data_pipeline import create_graph_data_object
from environment_v2 import EnvironmentV2, Action
from semantics import get_global_embedder


EMBEDDER = get_global_embedder()


def _graph_with_label(file_path: str, label: int) -> Data:
    """Create a PyG graph with an attached label and batch vector."""
    graph = create_graph_data_object(file_path, embedder=EMBEDDER)
    graph.y = torch.tensor([label], dtype=torch.long)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    return graph


def generate_dataset() -> list[Data]:
    """Construct a minimal dataset containing passing and failing code graphs."""
    print("Generating dataset...")

    base_dir = os.path.dirname(__file__)
    source_path = os.path.join(base_dir, "logic_v2.py")

    print("  - Generating 'good' code graph...")
    good_graph = _graph_with_label(source_path, label=0)

    print("  - Generating 'bad' code graph...")
    env = EnvironmentV2(file_to_modify="logic_v2.py", embedder=EMBEDDER)
    broken_source = env._apply_action(Action.CHANGE_OPERATOR_MULTIPLY_TO_ADD, verbose=False)
    env._restore_file()

    with NamedTemporaryFile("w", suffix="_logic_v2_bad.py", dir=base_dir, delete=False) as temp_file:
        temp_file.write(broken_source)
        temp_path = temp_file.name

    try:
        bad_graph = _graph_with_label(temp_path, label=1)
    finally:
        os.remove(temp_path)

    print("Dataset generated.")
    return [good_graph, bad_graph]


def main():
    dataset = generate_dataset()

    feature_dim = dataset[0].x.shape[1]
    model = CodeGraphNN(num_node_features=feature_dim, hidden_channels=64, num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    print("\n🚀 Starting Training Loop...")
    model.train()
    epochs = 100
    for epoch in range(epochs + 1):
        total_loss = 0.0
        for data in dataset:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 20 == 0:
            avg_loss = total_loss / len(dataset)
            print(f"Epoch {epoch:03d}, Loss: {avg_loss:.4f}")

    print("✅ Training Complete.\n")

    print("🕵️‍♀️ Evaluating trained model...")
    model.eval()
    with torch.no_grad():
        good_out = model(dataset[0].x, dataset[0].edge_index, dataset[0].batch)
        good_pred = good_out.argmax(dim=1)
        print(
            "  - Prediction for good_graph:",
            "Correct (Predicts Pass)" if good_pred.item() == 0 else "Incorrect",
        )

        bad_out = model(dataset[1].x, dataset[1].edge_index, dataset[1].batch)
        bad_pred = bad_out.argmax(dim=1)
        print(
            "  - Prediction for bad_graph:",
            "Correct (Predicts Fail)" if bad_pred.item() == 1 else "Incorrect",
        )


if __name__ == "__main__":
    main()
