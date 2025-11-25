#!/usr/bin/env python3
"""
Phase 1: Contrastive Pretraining for Code Graphs

Self-supervised pretraining on unlabeled code to learn general code representations
before fine-tuning on bug detection. This improves data efficiency and generalization.

Key Approach:
1. Collect unlabeled code from training_ground (real OSS projects)
2. Create positive pairs via semantic-preserving augmentations
3. Use NT-Xent loss to learn embeddings where similar code is close
4. Transfer learned representations to GNN backbone

Based on:
- SimCLR: "A Simple Framework for Contrastive Learning"
- GraphCL: "Graph Contrastive Learning with Augmentations"
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import sys
import copy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import SAGEConv, global_mean_pool


# ============================================================================
# Code Augmentation (simplified from augmentation.py to avoid import issues)
# ============================================================================

class AugmentationType(Enum):
    """Types of code augmentations"""
    VARIABLE_RENAME = "variable_rename"
    COMMENT_REMOVAL = "comment_removal"
    WHITESPACE_CHANGE = "whitespace_change"
    DOCSTRING_REMOVAL = "docstring_removal"
    TYPE_HINT_REMOVAL = "type_hint_removal"


class CodeAugmentor:
    """Applies semantic-preserving transformations to code."""

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.variable_names = ['var', 'val', 'item', 'data', 'result', 'temp', 'x', 'y', 'z']

    def augment(
        self,
        code: str,
        num_augmentations: int = 2,
        allowed_types: Optional[List[AugmentationType]] = None
    ) -> 'AugmentationResult':
        """Apply augmentations to code."""
        if allowed_types is None:
            allowed_types = list(AugmentationType)

        augmented = code
        applied_types = []

        try:
            ast.parse(code)
        except SyntaxError:
            return AugmentationResult(code, code, [], True, False)

        available_types = allowed_types.copy()
        random.shuffle(available_types)

        for aug_type in available_types[:num_augmentations]:
            try:
                if aug_type == AugmentationType.COMMENT_REMOVAL:
                    augmented = self._remove_comments(augmented)
                elif aug_type == AugmentationType.WHITESPACE_CHANGE:
                    augmented = self._change_whitespace(augmented)
                elif aug_type == AugmentationType.DOCSTRING_REMOVAL:
                    augmented = self._remove_docstrings(augmented)
                elif aug_type == AugmentationType.TYPE_HINT_REMOVAL:
                    augmented = self._remove_type_hints(augmented)
                elif aug_type == AugmentationType.VARIABLE_RENAME:
                    augmented = self._rename_variables(augmented)
                applied_types.append(aug_type)
            except Exception:
                continue

        try:
            ast.parse(augmented)
            ast_valid = True
        except SyntaxError:
            augmented = code
            ast_valid = True

        return AugmentationResult(code, augmented, applied_types, True, ast_valid)

    def _remove_comments(self, code: str) -> str:
        """Remove inline comments"""
        lines = code.split('\n')
        result = []
        for line in lines:
            if '#' in line:
                in_string = False
                cleaned = []
                for char in line:
                    if char in ['"', "'"]:
                        in_string = not in_string
                    if char == '#' and not in_string:
                        break
                    cleaned.append(char)
                result.append(''.join(cleaned).rstrip())
            else:
                result.append(line)
        return '\n'.join(result)

    def _change_whitespace(self, code: str) -> str:
        """Change whitespace (remove consecutive blank lines)"""
        lines = code.split('\n')
        result = []
        prev_blank = False
        for line in lines:
            is_blank = line.strip() == ''
            if not (is_blank and prev_blank):
                result.append(line)
            prev_blank = is_blank
        return '\n'.join(result)

    def _remove_docstrings(self, code: str) -> str:
        """Remove docstrings from functions/classes"""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if (node.body and isinstance(node.body[0], ast.Expr) and
                        isinstance(node.body[0].value, ast.Constant) and
                        isinstance(node.body[0].value.value, str)):
                        node.body = node.body[1:] or [ast.Pass()]
            return ast.unparse(tree)
        except Exception:
            return code

    def _remove_type_hints(self, code: str) -> str:
        """Remove type hints from function signatures"""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    node.returns = None
                    for arg in node.args.args:
                        arg.annotation = None
            return ast.unparse(tree)
        except Exception:
            return code

    def _rename_variables(self, code: str) -> str:
        """Simple variable renaming (limited scope)"""
        # Skip complex renaming for now - just return original
        return code


@dataclass
class AugmentationResult:
    """Result of code augmentation"""
    original_code: str
    augmented_code: str
    augmentation_types: List[AugmentationType]
    preserved_semantics: bool
    ast_valid: bool


# ============================================================================
# Simple Graph Data Creation (standalone version)
# ============================================================================

def create_simple_graph_data(code: str) -> Optional[Data]:
    """
    Create a simple graph representation of code.

    This is a standalone version that doesn't require the full data pipeline.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    # Collect functions and their metadata
    nodes = []
    node_features = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nodes.append(node.name)

            # Basic features
            features = [
                float(len(node.name)),  # Name length
                float(len(node.args.args)),  # Arg count
                float(len(node.body)),  # Body statement count
                1.0 if node.returns else 0.0,  # Has return annotation
                1.0 if ast.get_docstring(node) else 0.0,  # Has docstring
            ]

            # Count different node types in body
            for_count = 0
            if_count = 0
            call_count = 0
            return_count = 0

            for child in ast.walk(node):
                if isinstance(child, ast.For):
                    for_count += 1
                elif isinstance(child, ast.If):
                    if_count += 1
                elif isinstance(child, ast.Call):
                    call_count += 1
                elif isinstance(child, ast.Return):
                    return_count += 1

            features.extend([
                float(for_count),
                float(if_count),
                float(call_count),
                float(return_count),
            ])

            # Pad to fixed size
            while len(features) < 32:
                features.append(0.0)

            node_features.append(features[:32])

    if not nodes:
        # Create dummy node for files with no functions
        node_features.append([0.0] * 32)
        nodes.append("__module__")

    # Create edges (sequential for now)
    edge_index = []
    for i in range(len(nodes) - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])  # Bidirectional

    if not edge_index:
        edge_index = [[0, 0]]  # Self-loop for single node

    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


# ============================================================================
# Contrastive Pretraining Components
# ============================================================================

@dataclass
class ContrastivePretrainConfig:
    """Configuration for contrastive pretraining"""
    # Data
    training_ground_path: str = "training_ground"
    min_file_size: int = 100
    max_file_size: int = 50000
    exclude_tests: bool = False

    # Model
    hidden_channels: int = 256
    num_layers: int = 4
    projection_dim: int = 128

    # Training
    batch_size: int = 32
    learning_rate: float = 0.0003
    weight_decay: float = 1e-5
    epochs: int = 100
    temperature: float = 0.07

    # Augmentation
    num_augmentations: int = 2
    augmentation_types: List[AugmentationType] = field(default_factory=lambda: [
        AugmentationType.COMMENT_REMOVAL,
        AugmentationType.DOCSTRING_REMOVAL,
        AugmentationType.WHITESPACE_CHANGE,
        AugmentationType.TYPE_HINT_REMOVAL,
    ])

    # Checkpointing
    save_every: int = 10
    output_dir: str = "models"


class GraphContrastiveEncoder(nn.Module):
    """Graph encoder for contrastive learning with transferable backbone."""

    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int = 256,
        num_layers: int = 4,
        projection_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # SAGE backbone (transferable to main GNN)
        self.sage_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_channels = num_node_features
        for _ in range(num_layers):
            self.sage_layers.append(SAGEConv(in_channels, hidden_channels))
            self.norms.append(nn.BatchNorm1d(hidden_channels))
            in_channels = hidden_channels

        # Projection head (discarded after pretraining)
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, projection_dim),
        )
        self.hidden_channels = hidden_channels

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        return_embedding: bool = False,
    ) -> torch.Tensor:
        h = x
        for sage, norm in zip(self.sage_layers, self.norms):
            h = sage(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        graph_embedding = global_mean_pool(h, batch)

        if return_embedding:
            return graph_embedding

        z = self.projection(graph_embedding)
        return F.normalize(z, dim=1)

    def get_backbone_state_dict(self) -> Dict[str, torch.Tensor]:
        """Extract SAGE backbone weights for transfer."""
        state_dict = {}
        for i, (sage, norm) in enumerate(zip(self.sage_layers, self.norms)):
            for name, param in sage.named_parameters():
                state_dict[f"layers.{i}.{name}"] = param.data.clone()
            for name, param in norm.named_parameters():
                state_dict[f"norms.{i}.{name}"] = param.data.clone()
        return state_dict


class NTXentLoss(nn.Module):
    """NT-Xent loss for contrastive learning"""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)
        sim_matrix = torch.mm(z, z.T) / self.temperature

        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).to(z.device)

        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

        return F.cross_entropy(sim_matrix, labels)


class CodeGraphDataset(Dataset):
    """Dataset that creates augmented graph pairs for contrastive learning."""

    def __init__(
        self,
        code_files: List[Path],
        augmentor: CodeAugmentor,
        config: ContrastivePretrainConfig,
    ):
        self.code_files = code_files
        self.augmentor = augmentor
        self.config = config

        self.valid_samples: List[Tuple[str, Path]] = []
        print(f"Loading {len(code_files)} code files...")

        for path in code_files:
            try:
                code = path.read_text(encoding='utf-8', errors='ignore')
                ast.parse(code)
                self.valid_samples.append((code, path))
            except Exception:
                continue

        print(f"Loaded {len(self.valid_samples)} valid Python files")

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Tuple[Data, Data]:
        code, path = self.valid_samples[idx]

        aug1 = self.augmentor.augment(
            code,
            num_augmentations=self.config.num_augmentations,
            allowed_types=self.config.augmentation_types,
        )
        aug2 = self.augmentor.augment(
            code,
            num_augmentations=self.config.num_augmentations,
            allowed_types=self.config.augmentation_types,
        )

        graph1 = create_simple_graph_data(aug1.augmented_code)
        graph2 = create_simple_graph_data(aug2.augmented_code)

        if graph1 is None:
            graph1 = create_simple_graph_data(code)
        if graph2 is None:
            graph2 = create_simple_graph_data(code)

        # Fallback to dummy graph
        if graph1 is None:
            graph1 = Data(x=torch.zeros(1, 32), edge_index=torch.tensor([[0], [0]]))
        if graph2 is None:
            graph2 = Data(x=torch.zeros(1, 32), edge_index=torch.tensor([[0], [0]]))

        return graph1, graph2


def collate_pairs(batch: List[Tuple[Data, Data]]) -> Tuple[Batch, Batch]:
    """Custom collate function for graph pairs."""
    graphs1, graphs2 = zip(*batch)
    return Batch.from_data_list(graphs1), Batch.from_data_list(graphs2)


def collect_code_files(
    training_ground: Path,
    config: ContrastivePretrainConfig,
) -> List[Path]:
    """Collect Python files from training ground."""
    files = []
    excluded_dirs = {'.git', '__pycache__', '.venv', 'venv', 'env', 'node_modules'}

    for py_file in training_ground.rglob("*.py"):
        if any(part in excluded_dirs for part in py_file.parts):
            continue
        if config.exclude_tests and ('test' in py_file.name.lower() or 'tests' in py_file.parts):
            continue
        try:
            size = py_file.stat().st_size
            if config.min_file_size <= size <= config.max_file_size:
                files.append(py_file)
        except OSError:
            continue

    return files


def pretrain(config: ContrastivePretrainConfig, device: torch.device) -> GraphContrastiveEncoder:
    """Run contrastive pretraining."""

    training_ground = Path(config.training_ground_path)
    if not training_ground.exists():
        raise FileNotFoundError(f"Training ground not found: {training_ground}")

    code_files = collect_code_files(training_ground, config)
    print(f"Found {len(code_files)} code files")

    if len(code_files) < config.batch_size:
        raise ValueError(f"Not enough files ({len(code_files)}) for batch size {config.batch_size}")

    augmentor = CodeAugmentor()
    dataset = CodeGraphDataset(code_files, augmentor, config)

    if len(dataset) < config.batch_size:
        raise ValueError(f"Not enough valid samples ({len(dataset)}) for batch size {config.batch_size}")

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_pairs,
        num_workers=0,
        drop_last=True,
    )

    # Determine input dimension from first sample
    sample_graph = dataset[0][0]
    num_node_features = sample_graph.x.size(1)
    print(f"Node feature dimension: {num_node_features}")

    model = GraphContrastiveEncoder(
        num_node_features=num_node_features,
        hidden_channels=config.hidden_channels,
        num_layers=config.num_layers,
        projection_dim=config.projection_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.learning_rate / 100,
    )

    criterion = NTXentLoss(temperature=config.temperature)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting contrastive pretraining for {config.epochs} epochs...")
    print(f"Batch size: {config.batch_size}, LR: {config.learning_rate}")
    print(f"Temperature: {config.temperature}")
    print("-" * 50)

    history = {'epoch': [], 'loss': [], 'lr': []}
    best_loss = float('inf')

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch1, batch2 in dataloader:
            batch1 = batch1.to(device)
            batch2 = batch2.to(device)

            optimizer.zero_grad()

            z1 = model(batch1.x, batch1.edge_index, batch1.batch)
            z2 = model(batch2.x, batch2.edge_index, batch2.batch)

            loss = criterion(z1, z2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        current_lr = scheduler.get_last_lr()[0]

        history['epoch'].append(epoch + 1)
        history['loss'].append(avg_loss)
        history['lr'].append(current_lr)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}/{config.epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

        if (epoch + 1) % config.save_every == 0:
            checkpoint_path = output_dir / f"contrastive_epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'backbone_state_dict': model.get_backbone_state_dict(),
                'loss': avg_loss,
            }, output_dir / "contrastive_best.pt")

    final_path = output_dir / "contrastive_pretrained.pt"
    torch.save({
        'epoch': config.epochs,
        'model_state_dict': model.state_dict(),
        'backbone_state_dict': model.get_backbone_state_dict(),
        'loss': avg_loss,
        'history': history,
        'num_files': len(code_files),
        'timestamp': datetime.now().isoformat(),
    }, final_path)

    print("-" * 50)
    print(f"Pretraining complete! Final loss: {avg_loss:.4f}, Best: {best_loss:.4f}")
    print(f"Saved: {final_path}")

    with open(output_dir / "contrastive_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    return model


def main():
    parser = argparse.ArgumentParser(description="Contrastive Pretraining for Code Graphs")
    parser.add_argument("--training-ground", type=str, default="training_ground")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--hidden-channels", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    config = ContrastivePretrainConfig(
        training_ground_path=args.training_ground,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        temperature=args.temperature,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        output_dir=args.output_dir,
    )

    pretrain(config, device)


if __name__ == "__main__":
    main()
