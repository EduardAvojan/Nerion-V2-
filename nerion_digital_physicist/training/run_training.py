"""
This script trains the Digital Physicist's GNN brain on the curriculum database.
"""
import sqlite3
from pathlib import Path
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader

# Add project root to path to allow imports from other packages
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from nerion_digital_physicist.agent.brain import CodeGraphNN
from nerion_digital_physicist.agent.data import create_graph_data_from_source
from nerion_digital_physicist.agent.semantics import get_global_embedder

DB_PATH = Path("out/learning/curriculum.sqlite")
MODEL_SAVE_PATH = "digital_physicist_brain.pt"

# Training Hyperparameters
LEARNING_RATE = 0.001
EPOCHS = 50 # Increase for better training, decrease for faster runs
BATCH_SIZE = 16

def load_lessons_from_db():
    """Loads all lessons from the curriculum database."""
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}. No data to train on.")
        return []
    
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        cursor = con.cursor()
        cursor.execute("SELECT before_code, after_code FROM lessons")
        return cursor.fetchall()

def main():
    """Main function to run the training loop."""
    print("--- Starting GNN Training Session ---")

    # 1. Load Data
    lessons = load_lessons_from_db()
    if not lessons:
        print("No lessons found in the database. Halting training.")
        return

    print(f"Loaded {len(lessons)} lessons from the database.")

    # 2. Prepare Data
    print("Preparing graph data for training...")
    embedder = get_global_embedder()
    dataset = []
    for lesson in lessons:
        try:
            # 'before' code is a bad example (label 0)
            before_graph = create_graph_data_from_source(lesson['before_code'], embedder=embedder)
            before_graph.y = torch.tensor([0], dtype=torch.long)
            dataset.append(before_graph)

            # 'after' code is a good example (label 1)
            after_graph = create_graph_data_from_source(lesson['after_code'], embedder=embedder)
            after_graph.y = torch.tensor([1], dtype=torch.long)
            dataset.append(after_graph)
        except Exception as e:
            print(f" - Skipping lesson due to data preparation error: {e}")

    if not dataset:
        print("No valid data could be prepared for training. Halting.")
        return

    # PyG DataLoader handles batching of graph data
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Prepared {len(dataset)} total graphs for training.")

    # 3. Initialize Model and Optimizer
    # The number of features is taken from the first graph in the dataset
    num_node_features = dataset[0].num_node_features
    model = CodeGraphNN(
        num_node_features=num_node_features,
        hidden_channels=256,
        num_classes=2 # 0 for 'bad' code, 1 for 'good' code
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"Initialized GNN model with {num_node_features} node features.")

    # 4. Training Loop
    print(f"Starting training for {EPOCHS} epochs...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            # Use global_mean_pool to get a single embedding for the whole graph (or batch of graphs)
            graph_embedding = global_mean_pool(out, data.batch)
            loss = F.cross_entropy(graph_embedding, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f"  - Epoch {epoch + 1}/{EPOCHS}, Average Loss: {avg_loss:.4f}")

    print("Training complete.")

    # 5. Save the Model
    try:
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Successfully saved trained model to: {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"ERROR: Could not save model: {e}")

    print("--- GNN Training Session Complete ---")

if __name__ == "__main__":
    main()
