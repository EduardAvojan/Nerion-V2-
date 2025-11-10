"""
Multi-task training with EWC for bug detection transfer learning.

Implements the strategy to reach 85-90% accuracy:
1. Load CodeNet pretrained SAGE (79.2% general code understanding)
2. Stage 1: Freeze backbone, train bug head only on curriculum (5 epochs)
3. Stage 2: Unfreeze last layer, train with EWC + balanced sampling (15 epochs)
4. Optional Stage 3: Add GraphCodeBERT embeddings (5 epochs)

This preserves CodeNet knowledge while specializing for bug detection.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, WeightedRandomSampler, random_split
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_adj

from nerion_digital_physicist.agent.brain import MultiTaskCodeGraphSAGE
from nerion_digital_physicist.training.online_learner import OnlineLearner, EWCConfig


class OnlineEWC:
    """
    Simplified EWC for transfer learning with PyG DataLoader.

    Computes Fisher Information Matrix to identify important parameters
    for CodeNet task, then applies EWC penalty during curriculum training.
    """

    def __init__(self, model: nn.Module, lambda_ewc: float = 1000.0):
        self.lambda_ewc = lambda_ewc
        self.fisher_dict = {}
        self.optimal_params = {}

    def compute_fisher(self, model: nn.Module, data_loader: DataLoader, device: torch.device, num_samples: int = 5000):
        """
        Compute Fisher Information Matrix using PyG DataLoader.

        Fisher diagonal: F_i = E[(∂log p(y|x,θ)/∂θ_i)²]
        """
        model.eval()  # Use eval mode for Fisher computation

        # Initialize Fisher accumulators
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.fisher_dict[name] = torch.zeros_like(param.data)

        # Save optimal parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.clone()

        # Accumulate gradients squared over batches
        num_processed = 0
        total_grad_norm = 0
        for batch_idx, batch in enumerate(data_loader):
            if num_processed >= num_samples:
                break

            batch = batch.to(device)
            model.zero_grad()

            # Forward pass (bug detection task)
            # Note: Skip GraphCodeBERT for Fisher computation (CodeNet data may not batch it correctly)
            # Fisher measures SAGE backbone importance only, which is what we want
            out = model(
                batch.x,
                batch.edge_index,
                batch.batch,
                task='bug',
                graphcodebert_embedding=None  # Force None for Fisher on CodeNet
            ).squeeze()

            # Sample from model's distribution (not ground truth labels)
            # This is the key to Fisher - we use model's beliefs, not data labels
            probs = torch.sigmoid(out)
            sampled_labels = torch.bernoulli(probs).detach()

            # Compute log likelihood
            loss = F.binary_cross_entropy_with_logits(out, sampled_labels)
            loss.backward()

            # Debug: Check gradients on first batch
            if batch_idx == 0:
                print(f"\n🔍 FISHER COMPUTATION DEBUG (Batch 0):")
                print(f"   batch.y shape: {batch.y.shape}")
                print(f"   out shape: {out.shape}")
                print(f"   loss: {loss.item():.6f}")
                print(f"   sampled_labels mean: {sampled_labels.float().mean():.4f}")
                grad_count = 0
                total_norm = 0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        total_norm += grad_norm
                        grad_count += 1
                        if grad_norm > 0:
                            print(f"   ✓ {name}: grad_norm={grad_norm:.8f}")
                print(f"   Total parameters with gradients: {grad_count}")
                print(f"   Average gradient norm: {total_norm/grad_count if grad_count>0 else 0:.8f}")
                if total_norm < 1e-8:
                    print(f"   ❌ NO MEANINGFUL GRADIENTS!")
                print(f"🔍 END FISHER DEBUG\n")

            # Accumulate squared gradients
            batch_grad_norm = 0
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher_dict[name] += param.grad.data.pow(2)
                    batch_grad_norm += param.grad.norm().item()

            num_processed += batch.y.size(0)
            total_grad_norm += batch_grad_norm

        print(f"   Average gradient norm across all batches: {total_grad_norm / (batch_idx + 1):.8f}")

        # Average over samples
        print(f"   Averaging Fisher over {num_processed} samples...")
        for name in self.fisher_dict:
            self.fisher_dict[name] /= num_processed
            # Debug: Check first parameter's Fisher value
            if name == 'sage_layers.0.lin_l.weight':
                sample_fisher = self.fisher_dict[name].mean().item()
                print(f"   Sample parameter {name}: Fisher={sample_fisher:.12f}")

        return self.fisher_dict

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC penalty: λ/2 * Σ_i F_i * (θ_i - θ*_i)²
        
        CRITICAL: Only apply to backbone (SAGE layers), NOT bug head!
        The bug head is new and needs to learn freely.
        """
        if not self.fisher_dict:
            return torch.tensor(0.0)

        penalty = 0.0
        backbone_params = 0
        for name, param in model.named_parameters():
            # Skip bug head - it should learn freely!
            if 'bug_head' in name:
                continue
                
            if name in self.fisher_dict and name in self.optimal_params:
                penalty += (
                    self.fisher_dict[name] *
                    (param - self.optimal_params[name]).pow(2)
                ).sum()
                backbone_params += 1

        total_penalty = self.lambda_ewc / 2 * penalty
        print(f"   EWC penalty applied to {backbone_params} backbone parameters: {total_penalty:.6f}")
        return total_penalty


@dataclass
class MultiTaskConfig:
    codenet_checkpoint: Path  # 79.2% SAGE checkpoint
    curriculum_dataset: Path  # Bug detection curriculum
    output_dir: Path

    # Architecture
    hidden_channels: int = 512
    num_layers: int = 4
    dropout: float = 0.2
    use_graphcodebert: bool = True

    # Training stages
    stage1_epochs: int = 5  # Freeze backbone
    stage2_epochs: int = 15  # Unfreeze last layer + EWC
    batch_size: int = 16

    # EWC hyperparameters
    ewc_lambda: float = 1000.0  # Regularization strength
    ewc_samples: int = 5000  # Samples for Fisher computation

    # Balanced sampling
    curriculum_weight: float = 100.0  # Weight curriculum samples higher

    # Graph augmentation
    edge_dropout_prob: float = 0.1
    node_mask_prob: float = 0.05

    val_ratio: float = 0.15
    seed: int = 42


class GraphSampleDataset(Dataset):
    """Thin Dataset wrapper around a list of Data samples."""

    def __init__(self, samples: List[object]):
        self._samples = list(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        return self._samples[idx]


def augment_graph(data, edge_dropout_prob: float = 0.1, node_mask_prob: float = 0.05):
    """
    Graph augmentation to bridge CodeNet-to-curriculum domain gap.

    Makes clean CodeNet graphs look more like buggy curriculum graphs:
    - Randomly drop edges (simulates incomplete parsing)
    - Randomly mask node features (simulates semantic gaps)
    """
    # Clone to avoid modifying original
    data = data.clone()

    # Drop edges randomly
    if edge_dropout_prob > 0:
        edge_index, edge_attr = dropout_adj(
            data.edge_index,
            edge_attr=data.edge_attr if hasattr(data, 'edge_attr') else None,
            p=edge_dropout_prob,
            training=True
        )
        data.edge_index = edge_index
        if hasattr(data, 'edge_attr') and edge_attr is not None:
            data.edge_attr = edge_attr

    # Mask node features randomly
    if node_mask_prob > 0:
        node_mask = torch.rand(data.x.size(0)) > node_mask_prob
        data.x = data.x * node_mask.float().unsqueeze(1)

    return data


def load_curriculum_dataset(path: Path, verify_graphcodebert: bool = True) -> Tuple[List[object], dict]:
    """Load curriculum dataset (bug detection task)."""
    if not path.exists():
        raise FileNotFoundError(f"Curriculum dataset not found: {path}")

    payload = torch.load(path, weights_only=False)
    samples = payload.get("samples")
    metadata = payload.get("metadata", {})

    if not isinstance(samples, list) or not samples:
        raise RuntimeError(f"Dataset does not contain valid samples")

    # Verify GraphCodeBERT embeddings exist
    if verify_graphcodebert:
        print(f"🔍 Verifying GraphCodeBERT embeddings...")
        missing_gcb = 0
        for i, sample in enumerate(samples[:10]):  # Check first 10
            if not hasattr(sample, 'graphcodebert_embedding'):
                missing_gcb += 1
                print(f"   ❌ Sample {i} missing GraphCodeBERT embedding")

        if missing_gcb > 0:
            print(f"\n❌ ERROR: {missing_gcb}/10 samples missing GraphCodeBERT embeddings!")
            print(f"   Run: python3 -m nerion_digital_physicist.agent.semantics --precompute-gcb")
            print(f"   Or disable with --use-graphcodebert=False")
            sys.exit(1)
        else:
            print(f"   ✓ All samples have GraphCodeBERT embeddings (shape: {samples[0].graphcodebert_embedding.shape})")

    print(f"✓ Loaded curriculum: {len(samples):,} samples")
    return samples, metadata


def compute_fisher_on_codenet(
    model: MultiTaskCodeGraphSAGE,
    codenet_checkpoint_path: Path,
    num_samples: int = 5000,
    device: torch.device = None
) -> dict:
    """
    Compute Fisher Information Matrix on CodeNet task.

    This tells us which parameters are important for CodeNet,
    so we can penalize changes to them during curriculum training.
    """
    if device is None:
        device = torch.device('cpu')

    print(f"\n🧮 Computing Fisher Information Matrix on CodeNet...")
    print(f"   Using {num_samples} samples to estimate parameter importance")

    # Load a subset of CodeNet for Fisher computation
    codenet_dir = Path("experiments/datasets/gnn/batches")
    if not codenet_dir.exists():
        print(f"   ❌ ERROR: CodeNet batches directory not found: {codenet_dir}")
        print(f"      Expected location: experiments/datasets/gnn/batches/")
        sys.exit(1)

    batch_files = sorted(codenet_dir.glob("codenet_graphcodebert_batch_0[12].pt"))[:2]
    if not batch_files:
        print(f"   ❌ ERROR: No CodeNet batch files found in {codenet_dir}")
        print(f"      Expected files like: codenet_graphcodebert_batch_01.pt")
        sys.exit(1)

    print(f"   Loading from {len(batch_files)} CodeNet batches...")
    codenet_samples = []
    for batch_file in batch_files:
        print(f"      Loading {batch_file.name}...", end=" ")
        payload = torch.load(batch_file, weights_only=False)
        samples = payload.get("samples", [])
        print(f"{len(samples):,} samples")
        codenet_samples.extend(samples[:num_samples // 2])  # Split across 2 batches
        if len(codenet_samples) >= num_samples:
            break

    codenet_samples = codenet_samples[:num_samples]
    print(f"   ✓ Loaded {len(codenet_samples):,} CodeNet samples for Fisher computation")

    # Create dataloader
    dataset = GraphSampleDataset(codenet_samples)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Compute Fisher using EWC
    print(f"   Computing Fisher matrix (this may take a few minutes)...")

    # Move model to device for Fisher computation
    model_for_fisher = model.to(device)

    ewc = OnlineEWC(model_for_fisher, lambda_ewc=0.0)
    fisher_dict = ewc.compute_fisher(model_for_fisher, loader, device, num_samples=len(codenet_samples))

    # Verify Fisher is non-zero
    if fisher_dict:
        sample_fisher_value = list(fisher_dict.values())[0].mean().item()
        print(f"   ✓ Fisher matrix computed for {len(fisher_dict)} parameters")
        print(f"   Sample Fisher value: {sample_fisher_value:.10f}")  # More precision

        if sample_fisher_value < 1e-10:
            print(f"   ❌ WARNING: Fisher values are near-zero! EWC will NOT work!")
            print(f"   This means gradients aren't flowing during Fisher computation.")
            print(f"   Continuing without EWC regularization...")
        else:
            print(f"   ✓ Fisher values are non-zero (EWC will work!)")
    else:
        print(f"   ❌ ERROR: Fisher dictionary is empty!")
        sys.exit(1)

    return fisher_dict, ewc.optimal_params


def train_stage1_frozen_backbone(
    model: MultiTaskCodeGraphSAGE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: MultiTaskConfig,
    device: torch.device
) -> None:
    """
    Stage 1: Freeze backbone, train only bug detection head.

    This quickly adapts the head to bug patterns without touching
    the general code understanding learned from CodeNet.
    """
    print(f"\n{'='*70}")
    print(f"🎯 STAGE 1: Train bug head only (backbone frozen)")
    print(f"{'='*70}")
    print(f"   Epochs: {config.stage1_epochs}")
    print(f"   Only the bug_head will be trained")
    print(f"   SAGE layers are FROZEN (preserving CodeNet knowledge)")

    model.freeze_backbone()

    # DEBUG: Verify bug head is actually trainable
    print(f"\n🔍 DEBUG: Checking bug head trainability after freeze_backbone()")
    for name, param in model.bug_head.named_parameters():
        print(f"   {name}: requires_grad={param.requires_grad}")
    
    # Optimizer: Only bug head parameters
    trainable_params = [p for p in model.bug_head.parameters() if p.requires_grad]
    print(f"   Found {len(trainable_params)} trainable parameter groups in bug head")
    
    if not trainable_params:
        print(f"   ❌ ERROR: No trainable parameters found in bug head!")
        print(f"   This means freeze_backbone() froze everything or bug_head wasn't created properly")
        sys.exit(1)
    
    # Calculate class weights for balanced loss (handle class imbalance)
    pos_weight = torch.tensor([3.0]).to(device)  # Increased from 2.0 → 3.0 for stronger bug focus
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.Adam(trainable_params, lr=1e-4, weight_decay=1e-4)  # Add L2 regularization

    print(f"   Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"   Bug head architecture: 6 layers with BatchNorm")
    print(f"   Using weighted BCE (pos_weight={pos_weight.item():.1f}) for class imbalance")
    print(f"   Learning rate: 1e-4 (100x smaller for stability)")
    print(f"   Epochs: {config.stage1_epochs}")

    for epoch in range(1, config.stage1_epochs + 1):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()

            # GRAPHCODEBERT FIX: PyG flattens embeddings, reshape them efficiently
            graphcodebert_embeddings = None
            if hasattr(batch, 'graphcodebert_embedding') and batch.graphcodebert_embedding is not None:
                # PyG batches graph-level attributes by flattening: (batch_size * 768,)
                # Reshape to (batch_size, 768) - much faster than manual stacking
                num_graphs = batch.num_graphs
                graphcodebert_embeddings = batch.graphcodebert_embedding.view(num_graphs, -1).to(device)

            # DEBUG: Check first batch
            if epoch == 1 and batch_idx == 0:
                print(f"\n🔍 FIRST BATCH DEBUG:")
                print(f"   batch.x shape: {batch.x.shape}")
                print(f"   batch.y shape: {batch.y.shape}")
                print(f"   batch.y min/max: {batch.y.min()}/{batch.y.max()}")
                print(f"   batch.y mean: {batch.y.float().mean():.4f}")
                if graphcodebert_embeddings is not None:
                    print(f"   GraphCodeBERT shape: {graphcodebert_embeddings.shape}")
                    print(f"   GraphCodeBERT mean: {graphcodebert_embeddings.mean():.4f}")
                    print(f"   GraphCodeBERT std: {graphcodebert_embeddings.std():.4f}")
                    print(f"   GraphCodeBERT max: {graphcodebert_embeddings.abs().max():.4f}")
                    # Check for zeros or NaN
                    if graphcodebert_embeddings.abs().max() < 1e-6:
                        print(f"   ❌ ERROR: GraphCodeBERT embeddings are ALL ZEROS!")
                    if torch.isnan(graphcodebert_embeddings).any():
                        print(f"   ❌ ERROR: GraphCodeBERT embeddings have NaN values!")
                else:
                    print(f"   ⚠️  GraphCodeBERT is None!")

            out = model(
                batch.x,
                batch.edge_index,
                batch.batch,
                task='bug',
                graphcodebert_embedding=graphcodebert_embeddings
            ).squeeze()

            # DEBUG: Check shapes and values
            if epoch == 1 and batch_idx == 0:
                print(f"\n🔍 LOSS COMPUTATION DEBUG:")
                print(f"   out shape={out.shape}, batch.y shape={batch.y.shape}")
                print(f"   out sample={out[:3].detach().cpu().numpy()}")
                print(f"   batch.y sample={batch.y[:3].detach().cpu().numpy()}")
                print(f"   out mean={out.mean():.4f}, y mean={batch.y.float().mean():.4f}")
                print(f"   out min/max={out.min():.4f}/{out.max():.4f}")
                print(f"   y min/max={batch.y.min():.4f}/{batch.y.max():.4f}")

            loss = criterion(out, batch.y.float())
            
            # DEBUG: Check loss value
            if epoch == 1 and batch_idx == 0:
                print(f"   loss={loss.item():.6f}")
                print(f"🔍 END LOSS DEBUG\n")

            loss.backward()
            optimizer.step()

            # DEBUG: Check gradients on first batch
            if epoch == 1 and batch_idx == 0:
                print(f"   Loss: {loss.item():.4f}")
                # Check if any gradients exist
                has_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None and param.grad.abs().sum() > 0:
                        print(f"   ✓ Gradient flowing for: {name} (norm: {param.grad.norm():.6f})")
                        has_grad = True
                        break
                if not has_grad:
                    print(f"   ❌ NO GRADIENTS FLOWING!")
                print(f"🔍 END FIRST BATCH DEBUG\n")

            train_loss += loss.item()
            pred = (torch.sigmoid(out) > 0.5).long()
            train_correct += (pred == batch.y).sum().item()
            train_total += batch.y.size(0)

        train_acc = train_correct / train_total
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                # GRAPHCODEBERT FIX: PyG flattens embeddings, reshape them efficiently
                graphcodebert_embeddings = None
                if hasattr(batch, 'graphcodebert_embedding') and batch.graphcodebert_embedding is not None:
                    num_graphs = batch.num_graphs
                    graphcodebert_embeddings = batch.graphcodebert_embedding.view(num_graphs, -1).to(device)

                out = model(
                    batch.x,
                    batch.edge_index,
                    batch.batch,
                    task='bug',
                    graphcodebert_embedding=graphcodebert_embeddings
                ).squeeze()

                loss = criterion(out, batch.y.float())
                val_loss += loss.item()
                pred = (torch.sigmoid(out) > 0.5).long()
                val_correct += (pred == batch.y).sum().item()
                val_total += batch.y.size(0)

        val_acc = val_correct / val_total
        val_loss /= len(val_loader)

        print(f"   Epoch {epoch}/{config.stage1_epochs}: "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.1%} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.1%}")


def train_stage2_ewc(
    model: MultiTaskCodeGraphSAGE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    fisher_dict: dict,
    optimal_params: dict,
    config: MultiTaskConfig,
    device: torch.device
) -> dict:
    """
    Stage 2: Unfreeze last layer, train with EWC regularization.

    EWC prevents catastrophic forgetting by penalizing changes to
    parameters important for CodeNet task (identified by Fisher).
    """
    print(f"\n{'='*70}")
    print(f"🎯 STAGE 2: Unfreeze last layer + EWC regularization")
    print(f"{'='*70}")
    print(f"   Epochs: {config.stage2_epochs}")
    print(f"   Training: bug_head + last SAGE layer")
    print(f"   EWC lambda: {config.ewc_lambda}")

    model.unfreeze_last_layer()

    # Optimizer: Bug head + last SAGE layer
    optimizer = torch.optim.Adam([
        {'params': model.bug_head.parameters(), 'lr': 1e-3},
        {'params': model.sage_layers[-1].parameters(), 'lr': 5e-4},
        {'params': model.norms[-1].parameters(), 'lr': 5e-4},
    ])

    # Setup EWC
    ewc = OnlineEWC(model, lambda_ewc=config.ewc_lambda)
    ewc.fisher_dict = fisher_dict
    ewc.optimal_params = optimal_params  # CRITICAL: Transfer optimal params for penalty computation

    # Verify Fisher loaded
    print(f"   Fisher parameters: {len(fisher_dict)}")
    print(f"   Optimal params: {len(optimal_params)}")
    if fisher_dict:
        sample_fisher = list(fisher_dict.values())[0].mean().item()
        print(f"   Sample Fisher value: {sample_fisher:.6f}")

    # Loss function for bug detection (with pos_weight for class imbalance)
    pos_weight = torch.tensor([3.0]).to(device)  # Match Stage 1
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None
    patience = 5
    epochs_without_improvement = 0

    for epoch in range(1, config.stage2_epochs + 1):
        # Training
        model.train()
        train_loss = 0
        train_bug_loss = 0
        train_ewc_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Extract GraphCodeBERT embeddings (efficient reshape, no manual stacking)
            graphcodebert_embeddings = None
            if hasattr(batch, 'graphcodebert_embedding') and batch.graphcodebert_embedding is not None:
                num_graphs = batch.num_graphs
                graphcodebert_embeddings = batch.graphcodebert_embedding.view(num_graphs, -1).to(device)

            # Forward
            out = model(
                batch.x,
                batch.edge_index,
                batch.batch,
                task='bug',
                graphcodebert_embedding=graphcodebert_embeddings
            ).squeeze()

            # Bug detection loss
            bug_loss = criterion(out, batch.y.float())

            # EWC penalty (prevents forgetting CodeNet)
            ewc_penalty = ewc.penalty(model)

            # Total loss
            total_loss = bug_loss + ewc_penalty
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            train_bug_loss += bug_loss.item()
            train_ewc_loss += ewc_penalty.item()

            pred = (torch.sigmoid(out) > 0.5).long()
            train_correct += (pred == batch.y).sum().item()
            train_total += batch.y.size(0)

        train_acc = train_correct / train_total
        train_loss /= len(train_loader)
        train_bug_loss /= len(train_loader)
        train_ewc_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                # Extract GraphCodeBERT embeddings (efficient reshape, no manual stacking)
                graphcodebert_embeddings = None
                if hasattr(batch, 'graphcodebert_embedding') and batch.graphcodebert_embedding is not None:
                    num_graphs = batch.num_graphs
                    graphcodebert_embeddings = batch.graphcodebert_embedding.view(num_graphs, -1).to(device)

                out = model(
                    batch.x,
                    batch.edge_index,
                    batch.batch,
                    task='bug',
                    graphcodebert_embedding=graphcodebert_embeddings
                ).squeeze()

                loss = criterion(out, batch.y.float())
                val_loss += loss.item()
                pred = (torch.sigmoid(out) > 0.5).long()
                val_correct += (pred == batch.y).sum().item()
                val_total += batch.y.size(0)

        val_acc = val_correct / val_total
        val_loss /= len(val_loader)

        # Track best model and early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0

            # Save best checkpoint
            checkpoint_path = config.output_dir / "best_stage2_checkpoint.pt"
            torch.save({
                'model_state_dict': best_model_state,
                'epoch': epoch,
                'val_acc': val_acc,
                'config': config,
            }, checkpoint_path)
            print(f"   💾 Saved best checkpoint (val_acc={val_acc:.1%})")
        else:
            epochs_without_improvement += 1

        print(f"   Epoch {epoch}/{config.stage2_epochs}: "
              f"train_acc={train_acc:.1%} val_acc={val_acc:.1%} | "
              f"bug_loss={train_bug_loss:.4f} ewc_loss={train_ewc_loss:.4f}")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\n   ⚠️  Early stopping: No improvement for {patience} epochs")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n   ✓ Loaded best model from epoch {best_epoch}")

    print(f"\n   ✓ Best validation accuracy: {best_val_acc:.1%} (epoch {best_epoch})")

    return {
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-task EWC training for bug detection")
    parser.add_argument("--codenet-checkpoint", type=Path, required=True,
                        help="Path to CodeNet pretrained SAGE checkpoint (79.2%)")
    parser.add_argument("--curriculum-dataset", type=Path, required=True,
                        help="Path to curriculum dataset (.pt file)")
    parser.add_argument("--output-dir", type=Path, default=Path("out/training_runs/multitask_ewc"),
                        help="Output directory for checkpoints")
    parser.add_argument("--stage1-epochs", type=int, default=5,
                        help="Epochs for Stage 1 (frozen backbone)")
    parser.add_argument("--stage2-epochs", type=int, default=15,
                        help="Epochs for Stage 2 (EWC + last layer)")
    parser.add_argument("--ewc-lambda", type=float, default=1000.0,
                        help="EWC regularization strength")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--hidden-channels", type=int, default=512,
                        help="Hidden dimension size (default: 512)")

    args = parser.parse_args()

    config = MultiTaskConfig(
        codenet_checkpoint=args.codenet_checkpoint,
        curriculum_dataset=args.curriculum_dataset,
        output_dir=args.output_dir,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        ewc_lambda=args.ewc_lambda,
        batch_size=args.batch_size,
        hidden_channels=args.hidden_channels,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Multi-Task EWC Training for Bug Detection")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"CodeNet checkpoint: {config.codenet_checkpoint}")
    print(f"Curriculum dataset: {config.curriculum_dataset}")
    print(f"Output directory: {config.output_dir}")
    print(f"{'='*70}\n")

    # Load curriculum dataset
    curriculum_samples, metadata = load_curriculum_dataset(
        config.curriculum_dataset,
        verify_graphcodebert=config.use_graphcodebert
    )
    num_node_features = metadata.get('num_node_features', 32)

    # DEBUG: Inspect curriculum labels and embeddings
    print(f"\n📊 CURRICULUM LABEL INSPECTION:")
    print(f"   Total samples: {len(curriculum_samples)}")
    if len(curriculum_samples) > 0:
        print(f"   Sample 0: y={curriculum_samples[0].y}")
        if hasattr(curriculum_samples[0], 'graphcodebert_embedding') and curriculum_samples[0].graphcodebert_embedding is not None:
            print(f"   Sample 0: graphcodebert_embedding shape={curriculum_samples[0].graphcodebert_embedding.shape}")
            print(f"   Sample 0: graphcodebert_embedding mean={curriculum_samples[0].graphcodebert_embedding.mean():.4f}")
            print(f"   Sample 0: graphcodebert_embedding std={curriculum_samples[0].graphcodebert_embedding.std():.4f}")
        else:
            print(f"   Sample 0: graphcodebert_embedding=None")
        if len(curriculum_samples) > 1:
            print(f"   Sample 1: y={curriculum_samples[1].y}")
        if len(curriculum_samples) > 2:
            print(f"   Sample 2: y={curriculum_samples[2].y}")
        
        # Check first 100 samples for label distribution
        n_check = min(100, len(curriculum_samples))
        labels = [int(sample.y) for sample in curriculum_samples[:n_check]]
        positive = sum(labels)
        print(f"   First {n_check} samples: {positive} bugs, {n_check-positive} clean")
        if positive == 0:
            print(f"   ❌ ERROR: NO BUGS FOUND IN FIRST {n_check} SAMPLES!")
            print(f"   This means your curriculum has no positive labels!")
        elif positive == n_check:
            print(f"   ❌ ERROR: ALL SAMPLES ARE BUGS IN FIRST {n_check} SAMPLES!")
            print(f"   This means your curriculum has no negative labels!")
        else:
            print(f"   ✓ Label distribution looks reasonable")

    # Split into train/val
    dataset = GraphSampleDataset(curriculum_samples)
    val_size = int(len(dataset) * config.val_ratio)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator)

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")

    # Calculate class weights for balanced sampling (handles 31% bugs, 69% clean)
    print(f"\n📊 COMPUTING CLASS WEIGHTS FOR BALANCED SAMPLING:")
    train_labels = torch.tensor([dataset[i].y.item() for i in train_dataset.indices])
    num_bugs = train_labels.sum().item()
    num_clean = len(train_labels) - num_bugs
    print(f"   Training set: {num_bugs} bugs ({num_bugs/len(train_labels)*100:.1f}%), {num_clean} clean ({num_clean/len(train_labels)*100:.1f}%)")

    # Inverse frequency weighting: rare class gets higher weight
    class_weights = torch.tensor([
        len(train_labels) / (2 * num_clean),  # Weight for clean (class 0)
        len(train_labels) / (2 * num_bugs)     # Weight for bugs (class 1)
    ])
    print(f"   Class weights: clean={class_weights[0]:.4f}, bugs={class_weights[1]:.4f}")

    # Create sample weights: each sample gets weight of its class
    sample_weights = class_weights[train_labels.long()]

    # WeightedRandomSampler for balanced batches
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # Allow oversampling of minority class
    )
    print(f"   ✓ WeightedRandomSampler created (balanced batches)")

    # Create dataloaders (note: sampler and shuffle are mutually exclusive)
    train_loader = DataLoader(GraphSampleDataset(train_dataset), batch_size=config.batch_size, sampler=sampler)
    val_loader = DataLoader(GraphSampleDataset(val_dataset), batch_size=config.batch_size, shuffle=False)

    # Verify CodeNet checkpoint exists
    if not config.codenet_checkpoint.exists():
        print(f"❌ ERROR: CodeNet checkpoint not found: {config.codenet_checkpoint}")
        print(f"   Expected 79.2% SAGE checkpoint from previous training")
        sys.exit(1)

    # Initialize multi-task model
    print(f"\n🏗️  Building MultiTaskCodeGraphSAGE...")
    print(f"   Architecture: {config.num_layers} layers, {config.hidden_channels} hidden channels")
    print(f"   Dropout: {config.dropout}")
    print(f"   GraphCodeBERT: {config.use_graphcodebert}")

    model = MultiTaskCodeGraphSAGE(
        num_node_features=num_node_features,
        hidden_channels=config.hidden_channels,
        num_layers=config.num_layers,
        dropout=config.dropout,
        use_graphcodebert=config.use_graphcodebert,
        freeze_backbone=False,  # Start unfrozen for Fisher computation
    )

    # Load CodeNet pretrained weights
    print(f"\n📦 Loading CodeNet pretrained weights...")
    model.load_codenet_weights(str(config.codenet_checkpoint))

    # Count parameters BEFORE freezing
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable (before freezing): {trainable_params_before:,}")

    model = model.to(device)

    # Compute Fisher on CodeNet BEFORE freezing (needs gradients)
    fisher_dict, optimal_params = compute_fisher_on_codenet(model, config.codenet_checkpoint, config.ewc_samples, device)

    # Save Fisher to disk (can be reused)
    fisher_path = config.output_dir / "codenet_fisher.pt"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'fisher_dict': fisher_dict, 'optimal_params': optimal_params}, fisher_path)
    print(f"   💾 Saved Fisher to {fisher_path}")

    # NOW freeze backbone for Stage 1
    print(f"\n🔒 Freezing backbone for Stage 1...")
    model.freeze_backbone()
    trainable_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable (after freezing): {trainable_params_after:,} (bug head only)")

    # Stage 1: Freeze backbone, train bug head
    train_stage1_frozen_backbone(model, train_loader, val_loader, config, device)

    # Stage 2: Unfreeze last layer, train with EWC (or evaluate final Stage 1 results)
    if config.stage2_epochs > 0:
        results = train_stage2_ewc(model, train_loader, val_loader, fisher_dict, optimal_params, config, device)
    else:
        # Evaluate final Stage 1 model
        print(f"\n📊 Evaluating final Stage 1 model...")
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                # Extract GraphCodeBERT embeddings (efficient reshape, no manual stacking)
                graphcodebert_embeddings = None
                if hasattr(batch, 'graphcodebert_embedding') and batch.graphcodebert_embedding is not None:
                    num_graphs = batch.num_graphs
                    graphcodebert_embeddings = batch.graphcodebert_embedding.view(num_graphs, -1).to(device)

                out = model(
                    batch.x,
                    batch.edge_index,
                    batch.batch,
                    task='bug',
                    graphcodebert_embedding=graphcodebert_embeddings
                ).squeeze()

                pred = (torch.sigmoid(out) > 0.5).long()
                val_correct += (pred == batch.y).sum().item()
                val_total += batch.y.size(0)

        final_val_acc = val_correct / val_total
        results = {'best_val_acc': final_val_acc, 'best_epoch': config.stage1_epochs}
        print(f"   Final Stage 1 validation accuracy: {final_val_acc:.1%}")

    # Save final model
    final_checkpoint_path = config.output_dir / "multitask_ewc_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'results': results,
    }, final_checkpoint_path)

    print(f"\n{'='*70}")
    print(f"✓ Training complete!")
    print(f"   Final validation accuracy: {results['best_val_acc']:.1%}")
    print(f"   Saved to: {final_checkpoint_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
