"""
Self-Supervised Contrastive Learning for Code Embeddings

Learns code representations through contrastive learning:
- Positive pairs: Augmented versions of same code
- Negative pairs: Different code snippets
- Loss: NT-Xent (Normalized Temperature-scaled Cross Entropy)

Inspired by SimCLR and MoCo papers.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from pathlib import Path

from .augmentation import CodeAugmentor, AugmentationType


@dataclass
class ContrastiveTrainingConfig:
    """Configuration for contrastive learning"""
    # Model
    embedding_dim: int = 256
    hidden_dim: int = 512
    projection_dim: int = 128

    # Training
    batch_size: int = 64
    learning_rate: float = 0.001
    temperature: float = 0.07
    epochs: int = 100

    # Augmentation
    num_augmentations: int = 2
    augmentation_types: List[AugmentationType] = field(default_factory=lambda: [
        AugmentationType.VARIABLE_RENAME,
        AugmentationType.COMMENT_REMOVAL,
        AugmentationType.DOCSTRING_REMOVAL,
        AugmentationType.WHITESPACE_CHANGE
    ])

    # Optimization
    weight_decay: float = 0.0001
    momentum: float = 0.9
    warmup_epochs: int = 10


@dataclass
class ContrastiveLoss:
    """NT-Xent loss for contrastive learning"""
    temperature: float = 0.07

    def __call__(
        self,
        z_i: torch.Tensor,  # (batch, projection_dim)
        z_j: torch.Tensor,  # (batch, projection_dim)
    ) -> torch.Tensor:
        """
        Compute NT-Xent loss.

        Args:
            z_i: Projections from first augmentation
            z_j: Projections from second augmentation

        Returns:
            Loss value
        """
        batch_size = z_i.shape[0]

        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate both views
        z = torch.cat([z_i, z_j], dim=0)  # (2*batch, projection_dim)

        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.T)  # (2*batch, 2*batch)
        sim_matrix = sim_matrix / self.temperature

        # Create labels: positive pairs are (i, i+batch) and (i+batch, i)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).long()

        if z_i.is_cuda:
            labels = labels.cuda()

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool)
        if z_i.is_cuda:
            mask = mask.cuda()
        sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))

        # Cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.

    Maps embeddings to a lower-dimensional space for contrastive loss.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContrastiveEncoder(nn.Module):
    """
    Encoder for code embeddings with contrastive learning.

    Architecture:
    1. Embedding layer (code → hidden representation)
    2. Projection head (hidden → contrastive space)

    During training: Use projection head for contrastive loss
    During inference: Use hidden representation (before projection)
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        projection_dim: int
    ):
        super().__init__()

        # Main encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # Projection head (for contrastive loss only)
        self.projection = ProjectionHead(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim
        )

    def forward(
        self,
        x: torch.Tensor,
        return_projection: bool = True
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features
            return_projection: If True, return projection for contrastive loss.
                             If False, return embedding for downstream tasks.

        Returns:
            Embeddings or projections
        """
        h = self.encoder(x)

        if return_projection:
            z = self.projection(h)
            return z
        else:
            return h

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings (not projections) for downstream tasks"""
        return self.forward(x, return_projection=False)


class CodeContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning on code.

    Each sample returns two augmented views of the same code.
    """

    def __init__(
        self,
        code_samples: List[str],
        feature_extractor: Any,  # Converts code → feature vector
        augmentor: CodeAugmentor,
        num_augmentations: int = 2,
        augmentation_types: Optional[List[AugmentationType]] = None
    ):
        """
        Initialize dataset.

        Args:
            code_samples: List of code strings
            feature_extractor: Function/object that converts code to features
            augmentor: Code augmentor
            num_augmentations: Number of augmentations per view
            augmentation_types: Allowed augmentation types
        """
        self.code_samples = code_samples
        self.feature_extractor = feature_extractor
        self.augmentor = augmentor
        self.num_augmentations = num_augmentations
        self.augmentation_types = augmentation_types

    def __len__(self) -> int:
        return len(self.code_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get two augmented views of the same code.

        Returns:
            (view1_features, view2_features)
        """
        code = self.code_samples[idx]

        # Create two augmented views
        view1 = self.augmentor.augment(
            code,
            num_augmentations=self.num_augmentations,
            allowed_types=self.augmentation_types
        ).augmented_code

        view2 = self.augmentor.augment(
            code,
            num_augmentations=self.num_augmentations,
            allowed_types=self.augmentation_types
        ).augmented_code

        # Convert to features
        features1 = self.feature_extractor(view1)
        features2 = self.feature_extractor(view2)

        return features1, features2


class ContrastiveLearner:
    """
    Self-supervised contrastive learning for code embeddings.

    Usage:
        >>> learner = ContrastiveLearner(input_dim=768, config=config)
        >>> learner.train(code_samples, feature_extractor)
        >>> embedding = learner.get_embedding(code_features)
    """

    def __init__(
        self,
        input_dim: int,
        config: Optional[ContrastiveTrainingConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize contrastive learner.

        Args:
            input_dim: Dimension of input features
            config: Training configuration
            device: Device to use ('cuda' or 'cpu')
        """
        self.config = config or ContrastiveTrainingConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model
        self.model = ContrastiveEncoder(
            input_dim=input_dim,
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            projection_dim=self.config.projection_dim
        ).to(self.device)

        # Loss function
        self.criterion = ContrastiveLoss(temperature=self.config.temperature)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Training history
        self.train_losses: List[float] = []

        # Augmentor
        self.augmentor = CodeAugmentor()

    def train(
        self,
        code_samples: List[str],
        feature_extractor: Any,
        val_samples: Optional[List[str]] = None
    ) -> Dict[str, List[float]]:
        """
        Train contrastive model.

        Args:
            code_samples: Training code samples
            feature_extractor: Converts code to features
            val_samples: Optional validation samples

        Returns:
            Training history
        """
        # Create dataset
        dataset = CodeContrastiveDataset(
            code_samples=code_samples,
            feature_extractor=feature_extractor,
            augmentor=self.augmentor,
            num_augmentations=self.config.num_augmentations,
            augmentation_types=self.config.augmentation_types
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0  # Single-threaded for simplicity
        )

        # Training loop
        self.model.train()

        for epoch in range(self.config.epochs):
            epoch_losses = []

            for batch_idx, (view1, view2) in enumerate(dataloader):
                view1 = view1.to(self.device)
                view2 = view2.to(self.device)

                # Forward pass
                z1 = self.model(view1, return_projection=True)
                z2 = self.model(view2, return_projection=True)

                # Compute loss
                loss = self.criterion(z1, z2)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())

            # Log epoch statistics
            avg_loss = np.mean(epoch_losses)
            self.train_losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config.epochs} - Loss: {avg_loss:.4f}")

        return {
            'train_loss': self.train_losses
        }

    def get_embedding(self, features: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for downstream tasks.

        Args:
            features: Input features (batch, input_dim)

        Returns:
            Embeddings (batch, embedding_dim)
        """
        self.model.eval()
        with torch.no_grad():
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float32)

            features = features.to(self.device)
            embeddings = self.model.get_embedding(features)

        return embeddings

    def compute_similarity(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor
    ) -> float:
        """
        Compute similarity between two code samples.

        Args:
            features1: Features for first sample
            features2: Features for second sample

        Returns:
            Similarity score (0 to 1)
        """
        emb1 = self.get_embedding(features1)
        emb2 = self.get_embedding(features2)

        # Cosine similarity
        emb1 = F.normalize(emb1, dim=-1)
        emb2 = F.normalize(emb2, dim=-1)

        similarity = torch.dot(emb1.squeeze(), emb2.squeeze()).item()

        # Map from [-1, 1] to [0, 1]
        return (similarity + 1.0) / 2.0

    def save(self, path: Path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses
        }, path)
        print(f"[ContrastiveLearner] Saved checkpoint to {path}")

    def load(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        print(f"[ContrastiveLearner] Loaded checkpoint from {path}")


class SimpleFeatureExtractor:
    """
    Simple feature extractor for code (fallback when CodeBERT unavailable).

    Extracts basic syntactic features from code.
    """

    def __init__(self, output_dim: int = 768):
        self.output_dim = output_dim

    def __call__(self, code: str) -> torch.Tensor:
        """Extract features from code"""
        features = np.zeros(self.output_dim)

        # Basic syntactic features
        features[0] = len(code)
        features[1] = code.count('\n')
        features[2] = code.count('def ')
        features[3] = code.count('class ')
        features[4] = code.count('import ')
        features[5] = code.count('return ')
        features[6] = code.count('if ')
        features[7] = code.count('for ')
        features[8] = code.count('while ')
        features[9] = code.count('try ')
        features[10] = code.count('except ')

        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        return torch.tensor(features, dtype=torch.float32)


# Example usage
if __name__ == "__main__":
    # Sample code snippets
    code_samples = [
        """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
""",
        """
def factorial(x):
    if x <= 1:
        return 1
    return x * factorial(x-1)
""",
        """
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
""",
    ]

    # Create feature extractor
    feature_extractor = SimpleFeatureExtractor(output_dim=768)

    # Create config
    config = ContrastiveTrainingConfig(
        embedding_dim=128,
        hidden_dim=256,
        projection_dim=64,
        batch_size=2,
        epochs=5,
        learning_rate=0.001
    )

    # Create learner
    learner = ContrastiveLearner(
        input_dim=768,
        config=config,
        device='cpu'
    )

    print("=== Training Contrastive Model ===")
    history = learner.train(code_samples * 10, feature_extractor)

    print("\n=== Computing Similarities ===")
    f1 = feature_extractor(code_samples[0])
    f2 = feature_extractor(code_samples[1])
    f3 = feature_extractor(code_samples[2])

    sim_12 = learner.compute_similarity(f1, f2)
    sim_13 = learner.compute_similarity(f1, f3)
    sim_23 = learner.compute_similarity(f2, f3)

    print(f"Fibonacci vs Factorial: {sim_12:.3f}")
    print(f"Fibonacci vs IsPrime: {sim_13:.3f}")
    print(f"Factorial vs IsPrime: {sim_23:.3f}")
