import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class CodeGraphNN(torch.nn.Module):
    """
    The "Specialist-Grade" GNN brain for Nerion, featuring a deep and
    wide architecture with regularization.
    """

    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(CodeGraphNN, self).__init__()
        torch.manual_seed(42)

        # Wider and deeper architecture
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels * 2)
        self.conv3 = GCNConv(hidden_channels * 2, hidden_channels * 2)
        self.conv4 = GCNConv(hidden_channels * 2, hidden_channels)

        # A linear layer for the final classification
        self.linear = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # Deeper processing pipeline with dropout for regularization
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv3(x, edge_index).relu()

        x = self.conv4(x, edge_index).relu()

        # The final linear layer is applied to each node individually.
        x = self.linear(x)

        return x


def main():
    """Demonstrates creating an instance of the new, scaled-up brain."""
    NUM_NODE_FEATURES = 5  # Based on our data pipeline
    NUM_CLASSES = 2
    HIDDEN_CHANNELS = 256  # A strong, wide base layer

    model = CodeGraphNN(
        num_node_features=NUM_NODE_FEATURES,
        hidden_channels=HIDDEN_CHANNELS,
        num_classes=NUM_CLASSES,
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("🧠 Specialist-Grade GNN Brain Instantiated Successfully!")
    print(f"   - Hidden Channels: {HIDDEN_CHANNELS}")
    print(f"   - Total Trainable Parameters: {num_params:,}")
    print("\nModel Architecture:")
    print(model)


if __name__ == "__main__":
    main()
