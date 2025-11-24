"""
Nerion's Heart: Multi-Objective GNN Oracle (Model Architecture)

Core neural architecture for autonomous code repair:
- Test Oracle (Siamese): Predicts code-test correctness via Triplet Loss
- Performance Oracle (Regression): Predicts optimization needs via MSE Loss

This module contains ONLY the model definitions. Training is in train_nerion_heart.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class GNNBackbone(nn.Module):
    """
    Shared GraphSAGE backbone that extracts powerful graph representations.

    This is the "body" that learns universal code understanding.
    Input features combine GraphCodeBERT embeddings + AST structural features.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        """
        Args:
            in_channels: Input feature dimension (e.g., 800 = 768 GraphCodeBERT + 32 AST)
            hidden_channels: Hidden dimension for intermediate layers
            out_channels: Final embedding dimension
        """
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor):
        """
        Forward pass through the GNN backbone.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch index mapping nodes to graphs [num_nodes]

        Returns:
            (x_graph, x_nodes): Graph-level embedding and node-level embeddings
        """
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x_nodes = self.conv4(x, edge_index)
        x_graph = global_mean_pool(x_nodes, batch)  # [batch_size, out_channels]

        return x_graph, x_nodes


class MultiObjectiveGNNOracle(nn.Module):
    """
    Nerion's Heart: Multi-Objective GNN Oracle

    A unified architecture with two task-specific heads:
    1. Test Oracle Head: Learns code-test alignment via Triplet Loss
    2. Performance Oracle Head: Learns optimization needs via MSE Loss

    The magic happens in the training loop where both losses are combined.
    """

    def __init__(
        self,
        in_channels: int = 800,
        hidden_channels: int = 256,
        embedding_dim: int = 128,
    ):
        """
        Args:
            in_channels: Input feature dimension (GraphCodeBERT + AST)
            hidden_channels: Hidden dimension for intermediate layers
            embedding_dim: Final embedding dimension for both heads
        """
        super().__init__()

        # === SHARED BACKBONE ===
        # This GNN learns the universal code representation
        self.backbone = GNNBackbone(in_channels, hidden_channels, embedding_dim)

        # === HEAD 1: TEST ORACLE (Siamese/Triplet) ===
        # Projects backbone embedding to triplet loss space
        # Used to compare code vs. test via cosine distance
        self.triplet_head = nn.Linear(embedding_dim, embedding_dim)

        # === HEAD 2: PERFORMANCE ORACLE (Regression) ===
        # Predicts 3 continuous optimization scores: [speed, memory, complexity]
        # Each score is normalized to [0, 1] representing optimization need percentile
        self.regression_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 3),  # Output: [speed, memory, complexity]
        )

    def forward(self, data):
        """
        Single forward pass that computes outputs for BOTH tasks.

        Args:
            data: torch_geometric.data.Batch object with x, edge_index, batch,
                  and graphcodebert_embedding.

        Returns:
            (triplet_output, performance_scores):
                - triplet_output: [batch_size, embedding_dim] for Triplet Loss
                - performance_scores: [batch_size, 3] for MSE Loss
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        graphcodebert_embedding = data.graphcodebert_embedding

        # The DataLoader flattens the graph-level embeddings, so we reshape it.
        # It becomes [batch_size, embedding_dim] e.g., [32, 768]
        batch_size = data.num_graphs
        reshaped_graph_embedding = graphcodebert_embedding.view(batch_size, -1)

        # === FEATURE ENGINEERING: Combine Node and Graph Embeddings ===
        # Broadcast the graph-level GraphCodeBERT embedding to each node in the graph.
        # reshaped_graph_embedding is [batch_size, 768]
        # batch is [num_nodes_in_batch]
        # This operation creates [num_nodes_in_batch, 768]
        graph_embedding_broadcast = reshaped_graph_embedding[batch]

        # Concatenate with original node features to create rich input features
        # x is [num_nodes_in_batch, 32]
        # combined_x becomes [num_nodes_in_batch, 800]
        combined_x = torch.cat([x, graph_embedding_broadcast], dim=1)

        # === SHARED BACKBONE ===
        # Extract graph-level embedding from the GNN backbone
        graph_embedding, _ = self.backbone(combined_x, edge_index, batch)

        # === HEAD 1: Test Oracle Output ===
        # This vector will be used in triplet loss to compare codes
        triplet_output = self.triplet_head(graph_embedding)  # [batch_size, embedding_dim]

        # === HEAD 2: Performance Oracle Output ===
        # These 3 values predict the optimization needs
        performance_scores = self.regression_head(graph_embedding)  # [batch_size, 3]

        return triplet_output, performance_scores
