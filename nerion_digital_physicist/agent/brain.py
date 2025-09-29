"""Graph neural network architectures for the Digital Physicist."""

from __future__ import annotations

from typing import Callable

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv


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
        dropout: float = 0.2,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.residual = residual
        self.dropout = float(dropout)

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
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x, edge_index, batch):  # noqa: D401 - PyG style signature
        features = x
        for idx, conv in enumerate(self.layers):
            out = conv(features, edge_index)
            if self.norms is not None:
                out = self.norms[idx](out)
            out = F.relu(out)
            if self.residual and out.shape == features.shape:
                out = out + features
            out = F.dropout(out, p=self.dropout, training=self.training)
            features = out
        return self.head(features)


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
        )


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
        )


class CodeGraphGAT(_StackedGraphModel):
    """Graph Attention Network stack with configurable heads."""

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
    ) -> None:

        def _conv(in_ch: int, out_ch: int) -> nn.Module:
            return GATConv(
                in_ch,
                out_ch,
                heads=heads,
                concat=False,
                dropout=dropout,
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
        )


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
    }
    if key == "gat":
        kwargs["heads"] = attention_heads
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
    "CodeGraphNN",
    "build_gnn",
]
