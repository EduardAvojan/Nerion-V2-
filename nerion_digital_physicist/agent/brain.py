"""Graph neural network architectures for the Digital Physicist."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv, global_mean_pool


class _StackedGraphModel(nn.Module):
    """Shared stack used by all graph variants.

    The model keeps the output dimensionality stable across layers so residual
    connections and global pooling strategies remain valid regardless of the
    chosen convolution primitive.
    """

    def __init__(
        self,
        conv_factory: Callable[[int, int], nn.Module],
        num_node_features: int,
        hidden_channels: int,
        num_classes: int,
        *,
        num_layers: int = 4,
        residual: bool = False,
        dropout: float = 0.4,  # Increased from 0.2 to 0.4 for regularization
        use_batch_norm: bool = True,
        use_graphcodebert: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.residual = residual
        self.dropout = float(dropout)
        self.use_graphcodebert = use_graphcodebert

        layers = []
        norms = []
        in_channels = num_node_features
        for _ in range(num_layers):
            layers.append(conv_factory(in_channels, hidden_channels))
            if use_batch_norm:
                norms.append(nn.BatchNorm1d(hidden_channels))
            in_channels = hidden_channels

        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms) if norms else None

        # If using GraphCodeBERT, concatenate it with pooled graph features
        head_input_dim = hidden_channels
        if use_graphcodebert:
            head_input_dim = hidden_channels + 768  # 768 is GraphCodeBERT dimension

        self.head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x, edge_index, batch, *, use_dropout: bool = False, graphcodebert_embedding=None, edge_attr=None):  # noqa: D401 - PyG style signature
        features = x
        for idx, conv in enumerate(self.layers):
            # Pass edge_attr if the layer supports it (GAT, GATv2, EdgeConv, etc.)
            try:
                out = conv(features, edge_index, edge_attr=edge_attr)
            except TypeError:
                # Layer doesn't support edge_attr (GCN, SAGE, GIN)
                out = conv(features, edge_index)
            if self.norms is not None:
                out = self.norms[idx](out)
            out = F.relu(out)
            if self.residual and out.shape == features.shape:
                out = out + features
            # The `use_dropout` flag allows forcing dropout even in eval mode.
            out = F.dropout(out, p=self.dropout, training=self.training or use_dropout)
            features = out

        # Global pooling for graph-level prediction
        pooled_features = global_mean_pool(features, batch)

        # Concatenate GraphCodeBERT embedding if available
        if self.use_graphcodebert and graphcodebert_embedding is not None:
            # graphcodebert_embedding shape: (batch_size, 768)
            combined_features = torch.cat([pooled_features, graphcodebert_embedding], dim=1)
            return self.head(combined_features)
        elif self.use_graphcodebert:
            # If use_graphcodebert is True but embedding is missing, pad with zeros
            batch_size = pooled_features.size(0)
            zero_embedding = torch.zeros(batch_size, 768, device=pooled_features.device)
            combined_features = torch.cat([pooled_features, zero_embedding], dim=1)
            return self.head(combined_features)
        else:
            return self.head(pooled_features)


def _make_mlp(in_channels: int, hidden_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_channels, hidden_channels),
        nn.ReLU(),
        nn.Linear(hidden_channels, hidden_channels),
        nn.ReLU(),
    )


class CodeGraphGCN(_StackedGraphModel):
    """GCN stack with optional residual connections."""

    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int,
        num_classes: int,
        *,
        num_layers: int = 4,
        residual: bool = False,
        dropout: float = 0.2,
        use_graphcodebert: bool = False,
    ) -> None:
        super().__init__(
            lambda in_ch, out_ch: GCNConv(in_ch, out_ch),
            num_node_features,
            hidden_channels,
            num_classes,
            num_layers=num_layers,
            residual=residual,
            dropout=dropout,
            use_batch_norm=True,
            use_graphcodebert=use_graphcodebert,
        )

    def predict_health(self, graph_data) -> float:
        """
        Predict the health score of a code graph.
        
        Args:
            graph_data: PyG Data object containing the code graph
            
        Returns:
            float: Health score between 0.0 (healthy) and 1.0 (buggy)
        """
        self.eval()
        with torch.no_grad():
            # Add batch dimension if missing
            if not hasattr(graph_data, 'batch') or graph_data.batch is None:
                graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long)
            
            # Forward pass
            logits = self.forward(
                graph_data.x, 
                graph_data.edge_index, 
                graph_data.batch,
                edge_attr=getattr(graph_data, 'edge_attr', None)
            )
            
            # Convert to probability (sigmoid)
            probs = torch.sigmoid(logits)
            
            # Return probability of "bug" class (assuming class 1 is bug, or single output)
            # If multi-class, we might need to adjust. 
            # Based on MultiTaskCodeGraphSAGE, bug head outputs 1 dim.
            # But CodeGraphGCN has num_classes.
            
            if probs.dim() > 1 and probs.size(1) > 1:
                # Multi-class: return max probability or probability of specific "bug" class
                # Assuming class 0 is healthy, 1-4 are error types.
                # So health score = sum(probs[1:]) or 1 - prob[0]
                return 1.0 - probs[0, 0].item()
            else:
                return probs.item()


class CodeGraphSAGE(_StackedGraphModel):
    """GraphSAGE stack with residual support."""

    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int,
        num_classes: int,
        *,
        num_layers: int = 4,
        residual: bool = False,
        dropout: float = 0.2,
        use_graphcodebert: bool = False,
    ) -> None:
        super().__init__(
            lambda in_ch, out_ch: SAGEConv(in_ch, out_ch),
            num_node_features,
            hidden_channels,
            num_classes,
            num_layers=num_layers,
            residual=residual,
            dropout=dropout,
            use_batch_norm=True,
            use_graphcodebert=use_graphcodebert,
        )


class CodeGraphGIN(_StackedGraphModel):
    """GIN stack with per-layer MLPs and batch normalisation."""

    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int,
        num_classes: int,
        *,
        num_layers: int = 4,
        residual: bool = False,
        dropout: float = 0.2,
        use_graphcodebert: bool = False,
    ) -> None:
        super().__init__(
            lambda in_ch, out_ch: GINConv(_make_mlp(in_ch, out_ch)),
            num_node_features,
            hidden_channels,
            num_classes,
            num_layers=num_layers,
            residual=residual,
            dropout=dropout,
            use_batch_norm=True,
            use_graphcodebert=use_graphcodebert,
        )


class CodeGraphGAT(_StackedGraphModel):
    """Graph Attention Network stack with configurable heads and edge awareness."""

    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int,
        num_classes: int,
        *,
        num_layers: int = 4,
        residual: bool = False,
        dropout: float = 0.2,
        heads: int = 4,
        use_graphcodebert: bool = False,
        edge_dim: int = 6,  # Number of edge types (sequence, call, shared_symbol, control_flow, data_flow, contains)
    ) -> None:

        def _conv(in_ch: int, out_ch: int) -> nn.Module:
            return GATConv(
                in_ch,
                out_ch,
                heads=heads,
                concat=False,
                dropout=dropout,
                edge_dim=edge_dim,  # Enable edge-aware attention
            )

        super().__init__(
            _conv,
            num_node_features,
            hidden_channels,
            num_classes,
            num_layers=num_layers,
            residual=residual,
            dropout=dropout,
            use_batch_norm=True,
            use_graphcodebert=use_graphcodebert,
        )


class MultiTaskCodeGraphSAGE(nn.Module):
    """
    Multi-task GraphSAGE for transfer learning from CodeNet to bug detection.

    Architecture:
    - Shared SAGE backbone (can be frozen to preserve CodeNet knowledge)
    - Task-specific heads for bug detection and optional CodeNet auxiliary task
    - Supports EWC regularization to prevent catastrophic forgetting

    Usage:
        # Stage 1: Freeze backbone, train only bug head
        model = MultiTaskCodeGraphSAGE(freeze_backbone=True)
        model.load_codenet_weights('checkpoint.pt')

        # Stage 2: Unfreeze last layer + bug head
        model.unfreeze_last_layer()
    """

    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int,
        *,
        num_layers: int = 4,
        residual: bool = False,
        dropout: float = 0.2,
        use_graphcodebert: bool = False,
        freeze_backbone: bool = False,
        num_codenet_classes: int = 2,  # Optional: for auxiliary CodeNet task
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.residual = residual
        self.dropout = float(dropout)
        self.use_graphcodebert = use_graphcodebert
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        # Build SAGE backbone (shared across tasks)
        self.sage_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_channels = num_node_features

        for _ in range(num_layers):
            self.sage_layers.append(SAGEConv(in_channels, hidden_channels))
            self.norms.append(nn.BatchNorm1d(hidden_channels))
            in_channels = hidden_channels

        # Freeze backbone if requested (preserves CodeNet knowledge)
        if freeze_backbone:
            self.freeze_backbone()

        # Task-specific heads
        head_input_dim = hidden_channels
        if use_graphcodebert:
            head_input_dim = hidden_channels + 768  # 768 = GraphCodeBERT dimension

        # Bug detection head (primary task) - Graph Attention for 90%+ accuracy
        # Operates on graph structure BEFORE pooling to learn WHERE bugs are
        self.bug_head = nn.ModuleList([
            # First GAT layer: Learn bug-relevant node features
            GATConv(
                in_channels=head_input_dim,  # SAGE + GraphCodeBERT features
                out_channels=256,
                heads=4,
                dropout=self.dropout,
                concat=True  # Concatenate heads → 256*4 = 1024 dims
            ),
            nn.BatchNorm1d(256 * 4),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            
            # Second GAT layer: Refine bug detection patterns
            GATConv(
                in_channels=256 * 4,
                out_channels=128,
                heads=4,
                dropout=self.dropout,
                concat=True  # 128*4 = 512 dims
            ),
            nn.BatchNorm1d(128 * 4),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            
            # Third GAT layer: Final bug prediction per node
            GATConv(
                in_channels=128 * 4,
                out_channels=1,
                heads=1,
                dropout=self.dropout,
                concat=False  # Single output per node
            )
        ])
        
        # Global pooling to aggregate node predictions
        self.global_pool = global_mean_pool

        # CodeNet auxiliary head (optional, for multi-task regularization)
        self.codenet_head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(hidden_channels, num_codenet_classes),
        )

    def freeze_backbone(self):
        """Freeze all SAGE layers and norms to preserve CodeNet knowledge."""
        for param in self.sage_layers.parameters():
            param.requires_grad = False
        for param in self.norms.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all SAGE layers for full fine-tuning."""
        for param in self.sage_layers.parameters():
            param.requires_grad = True
        for param in self.norms.parameters():
            param.requires_grad = True

    def unfreeze_last_layer(self):
        """Unfreeze only the last SAGE layer (Stage 2 training)."""
        for param in self.sage_layers[-1].parameters():
            param.requires_grad = True
        for param in self.norms[-1].parameters():
            param.requires_grad = True

    def load_codenet_weights(self, checkpoint_path: str):
        """Load pretrained CodeNet weights into the backbone."""
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        # Extract state dict (handle both direct state dict and nested checkpoint)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Map pretrained weights to our structure
        # Pretrained uses: layers.0, layers.1, norms.0, norms.1
        # We use: sage_layers.0, sage_layers.1, norms.0, norms.1
        backbone_dict = {}
        for key, value in state_dict.items():
            if key.startswith('layers.'):
                # Rename layers -> sage_layers
                new_key = key.replace('layers.', 'sage_layers.')
                backbone_dict[new_key] = value
            elif key.startswith('norms.'):
                # Keep norms as is
                backbone_dict[key] = value

        # Load into current model (ignore head - we have new task-specific heads)
        missing_keys, unexpected_keys = self.load_state_dict(backbone_dict, strict=False)
        print(f"✓ Loaded CodeNet weights from {checkpoint_path}")
        print(f"  Loaded {len(backbone_dict)} backbone parameters")
        if missing_keys:
            print(f"  Missing keys (expected - new task heads): {len([k for k in missing_keys if 'head' in k])}")
        if unexpected_keys:
            print(f"  Unexpected keys: {unexpected_keys}")

    def forward(self, x, edge_index, batch, *, task='bug', use_dropout: bool = False, graphcodebert_embedding=None, edge_attr=None):
        """
        Forward pass with task selection.

        Args:
            task: 'bug' for bug detection, 'codenet' for auxiliary task
        """
        # Shared SAGE backbone
        features = x
        for idx, (conv, norm) in enumerate(zip(self.sage_layers, self.norms)):
            try:
                out = conv(features, edge_index, edge_attr=edge_attr)
            except TypeError:
                out = conv(features, edge_index)

            out = norm(out)
            out = F.relu(out)

            if self.residual and out.shape == features.shape:
                out = out + features

            out = F.dropout(out, p=self.dropout, training=self.training or use_dropout)
            features = out

        # Concatenate GraphCodeBERT embeddings to node features BEFORE attention
        # This allows the GAT to learn how GraphCodeBERT features relate to graph structure
        if self.use_graphcodebert and graphcodebert_embedding is not None:
            # graphcodebert_embedding is [batch_size, 768], need to expand to nodes
            # Use batch index to repeat for each node in the graph
            node_gcb = graphcodebert_embedding[batch]  # This broadcasts correctly
            features = torch.cat([features, node_gcb], dim=1)
        elif self.use_graphcodebert:
            # Pad with zeros if GraphCodeBERT missing
            zero_gcb = torch.zeros(features.size(0), 768, device=features.device)
            features = torch.cat([features, zero_gcb], dim=1)

        # Task-specific head
        if task == 'bug':
            # Apply GAT layers sequentially on graph structure
            x = features
            for i, layer in enumerate(self.bug_head):
                if isinstance(layer, GATConv):
                    # GAT layers need edge_index to propagate attention
                    x = layer(x, edge_index)
                else:
                    # BatchNorm, ReLU, Dropout operate on node features
                    x = layer(x)
            
            # Pool node predictions to graph-level prediction
            out = self.global_pool(x, batch).squeeze()
            return out  # Return logits (BCEWithLogitsLoss will apply sigmoid)
            
        elif task == 'codenet':
            # CodeNet head still uses pooled features (simpler task)
            pooled_features = global_mean_pool(features, batch)
            return self.codenet_head(pooled_features)
        else:
            raise ValueError(f"Unknown task: {task}")


MODEL_REGISTRY = {
    "gcn": CodeGraphGCN,
    "sage": CodeGraphSAGE,
    "gin": CodeGraphGIN,
    "gat": CodeGraphGAT,
}


def build_gnn(
    architecture: str,
    num_node_features: int,
    hidden_channels: int,
    num_classes: int,
    *,
    num_layers: int = 4,
    residual: bool = False,
    dropout: float = 0.2,
    attention_heads: int = 4,
    use_graphcodebert: bool = False,
    edge_dim: int = 6,  # Number of edge types for edge-aware models
) -> nn.Module:
    """Return the requested GNN architecture."""

    key = architecture.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{architecture}'. Available: {', '.join(sorted(MODEL_REGISTRY))}"
        )

    model_cls = MODEL_REGISTRY[key]
    kwargs = {
        "num_layers": num_layers,
        "residual": residual,
        "dropout": dropout,
        "use_graphcodebert": use_graphcodebert,
    }
    if key == "gat":
        kwargs["heads"] = attention_heads
        kwargs["edge_dim"] = edge_dim  # GAT supports edge features
    return model_cls(
        num_node_features,
        hidden_channels,
        num_classes,
        **kwargs,
    )


# Backwards compatibility alias used across the codebase.
CodeGraphNN = CodeGraphGCN


__all__ = [
    "CodeGraphGCN",
    "CodeGraphSAGE",
    "CodeGraphGIN",
    "CodeGraphGAT",
    "MultiTaskCodeGraphSAGE",
    "CodeGraphNN",
    "build_gnn",
]
