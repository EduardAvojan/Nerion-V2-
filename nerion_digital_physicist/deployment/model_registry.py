"""
Model Registry & Rollback System

Manages GNN model versions with semantic versioning and safe deployment.

Features:
- Semantic versioning (MAJOR.MINOR.PATCH)
- Performance tracking per version
- Automatic rollback on degradation
- Canary deployment (gradual rollout)
- A/B testing support
- Model comparison and diff

Integration:
- Continuous learning produces new model versions
- Daemon loads model from registry
- Auto-rollback if validation fails
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn


class DeploymentStage(Enum):
    """Model deployment stage"""
    DEVELOPMENT = "development"      # In training
    STAGING = "staging"              # Validation in progress
    CANARY = "canary"                # 10% production traffic
    PRODUCTION = "production"        # 100% production traffic
    DEPRECATED = "deprecated"        # Old version, marked for removal
    ROLLED_BACK = "rolled_back"     # Rolled back due to issues


@dataclass
class ModelVersion:
    """A registered model version"""
    version: str                     # Semantic version (e.g., "1.2.3")
    model_path: Path                 # Path to model weights
    created_at: str
    stage: DeploymentStage

    # Performance metrics
    validation_accuracy: float
    old_task_accuracy: float         # Forgetting metric
    test_accuracy: Optional[float] = None

    # Metadata
    architecture: str = "GraphSAGE"  # GCN, GraphSAGE, GIN, GAT
    num_parameters: int = 0
    training_samples: int = 0
    incremental_update: bool = False

    # Model Hyperparameters (for proper reconstruction)
    # These are CRITICAL for loading the model with correct architecture
    num_node_features: int = 768     # Input feature dimension (CodeBERT: 768)
    hidden_channels: int = 256       # Hidden layer dimension
    num_layers: int = 4              # Number of GNN layers
    residual: bool = False           # Use residual connections
    dropout: float = 0.2             # Dropout rate
    use_graphcodebert: bool = False  # Use GraphCodeBERT embeddings
    attention_heads: int = 4         # Number of attention heads (for GAT)

    # Deployment
    deployment_timestamp: Optional[str] = None
    traffic_percentage: float = 0.0  # 0-100%

    # Provenance
    parent_version: Optional[str] = None
    update_method: Optional[str] = None  # "full_retrain", "incremental", "maml"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        d = asdict(self)
        d['model_path'] = str(self.model_path)
        d['stage'] = self.stage.value
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> ModelVersion:
        """Create from dictionary"""
        d['model_path'] = Path(d['model_path'])
        d['stage'] = DeploymentStage(d['stage'])
        return ModelVersion(**d)


@dataclass
class RollbackEvent:
    """A rollback event"""
    timestamp: str
    from_version: str
    to_version: str
    reason: str
    triggered_by: str                # "auto", "manual"
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]


class ModelRegistry:
    """
    Model registry for version management and safe deployment.

    Usage:
        >>> registry = ModelRegistry(Path("out/models"))
        >>>
        >>> # Register new model
        >>> version = registry.register(
        ...     model=updated_model,
        ...     validation_accuracy=0.78,
        ...     old_task_accuracy=0.75,
        ...     update_method="incremental"
        ... )
        >>>
        >>> # Load current production model
        >>> model = registry.get_production_model()
        >>>
        >>> # Rollback if needed
        >>> registry.rollback(reason="accuracy_degradation")
    """

    def __init__(self, registry_root: Path):
        """
        Initialize model registry.

        Args:
            registry_root: Root directory for model storage
        """
        self.registry_root = registry_root
        self.registry_root.mkdir(parents=True, exist_ok=True)

        # Paths
        self.models_dir = self.registry_root / "models"
        self.models_dir.mkdir(exist_ok=True)

        self.metadata_file = self.registry_root / "registry.json"
        self.rollback_log = self.registry_root / "rollbacks.json"

        # Load registry
        self.versions: Dict[str, ModelVersion] = {}
        self.rollbacks: List[RollbackEvent] = []
        self._load_registry()

    def register(
        self,
        model: nn.Module,
        validation_accuracy: float,
        old_task_accuracy: float,
        test_accuracy: Optional[float] = None,
        architecture: str = "GraphSAGE",
        training_samples: int = 0,
        update_method: str = "full_retrain",
        notes: str = "",
        # Model hyperparameters (CRITICAL for loading)
        num_node_features: int = 768,
        hidden_channels: int = 256,
        num_layers: int = 4,
        residual: bool = False,
        dropout: float = 0.2,
        use_graphcodebert: bool = False,
        attention_heads: int = 4
    ) -> ModelVersion:
        """
        Register a new model version.

        Automatically increments version based on update type:
        - MAJOR: Architecture change (manual)
        - MINOR: Incremental learning update
        - PATCH: Bug fix or parameter tuning

        Args:
            model: Model to register
            validation_accuracy: Validation set accuracy
            old_task_accuracy: Accuracy on old tasks (forgetting metric)
            test_accuracy: Test set accuracy (optional)
            architecture: Model architecture
            training_samples: Number of training samples
            update_method: "full_retrain", "incremental", "maml"
            notes: Version notes
            num_node_features: Input feature dimension
            hidden_channels: Hidden layer dimension
            num_layers: Number of GNN layers
            residual: Use residual connections
            dropout: Dropout rate
            use_graphcodebert: Use GraphCodeBERT embeddings
            attention_heads: Number of attention heads (for GAT)

        Returns:
            ModelVersion object
        """
        # Determine next version
        current_prod = self._get_current_production()
        next_version = self._next_version(current_prod, update_method)

        print(f"[ModelRegistry] Registering version {next_version}")

        # Save model weights
        model_filename = f"model_{next_version.replace('.', '_')}.pt"
        model_path = self.models_dir / model_filename
        torch.save(model.state_dict(), model_path)

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())

        # Create version entry
        version = ModelVersion(
            version=next_version,
            model_path=model_path,
            created_at=datetime.now().isoformat(),
            stage=DeploymentStage.DEVELOPMENT,
            validation_accuracy=validation_accuracy,
            old_task_accuracy=old_task_accuracy,
            test_accuracy=test_accuracy,
            architecture=architecture,
            num_parameters=num_params,
            training_samples=training_samples,
            incremental_update=(update_method == "incremental"),
            parent_version=current_prod.version if current_prod else None,
            update_method=update_method,
            notes=notes,
            # Store hyperparameters for proper model reconstruction
            num_node_features=num_node_features,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            residual=residual,
            dropout=dropout,
            use_graphcodebert=use_graphcodebert,
            attention_heads=attention_heads
        )

        # Register
        self.versions[next_version] = version
        self._save_registry()

        print(f"[ModelRegistry] Registered: {next_version} "
              f"(val_acc={validation_accuracy:.4f}, "
              f"old_acc={old_task_accuracy:.4f})")

        return version

    def promote(
        self,
        version: str,
        target_stage: DeploymentStage,
        traffic_percentage: float = 100.0
    ) -> bool:
        """
        Promote model to next deployment stage.

        Workflow:
        - development → staging (validation)
        - staging → canary (10% traffic)
        - canary → production (100% traffic)

        Args:
            version: Version to promote
            target_stage: Target deployment stage
            traffic_percentage: Traffic % for canary deployment

        Returns:
            True if promoted successfully
        """
        if version not in self.versions:
            print(f"[ModelRegistry] Version {version} not found")
            return False

        model_version = self.versions[version]

        # Validation checks
        if target_stage == DeploymentStage.PRODUCTION:
            if not self._validate_for_production(model_version):
                print(f"[ModelRegistry] Version {version} failed production validation")
                return False

        # Demote current production if promoting to production
        if target_stage == DeploymentStage.PRODUCTION:
            current_prod = self._get_current_production()
            if current_prod and current_prod.version != version:
                self.versions[current_prod.version].stage = DeploymentStage.DEPRECATED
                self.versions[current_prod.version].traffic_percentage = 0.0

        # Promote
        model_version.stage = target_stage
        model_version.traffic_percentage = traffic_percentage
        if target_stage == DeploymentStage.PRODUCTION:
            model_version.deployment_timestamp = datetime.now().isoformat()

        self._save_registry()

        print(f"[ModelRegistry] Promoted {version} to {target_stage.value} "
              f"({traffic_percentage}% traffic)")

        return True

    def rollback(
        self,
        reason: str,
        target_version: Optional[str] = None,
        triggered_by: str = "auto"
    ) -> bool:
        """
        Rollback to previous stable version.

        Args:
            reason: Reason for rollback
            target_version: Specific version to rollback to (auto-select if None)
            triggered_by: "auto" or "manual"

        Returns:
            True if rolled back successfully
        """
        current_prod = self._get_current_production()
        if not current_prod:
            print("[ModelRegistry] No production model to rollback from")
            return False

        # Find target version
        if target_version is None:
            # Rollback to parent version
            target_version = current_prod.parent_version
            if not target_version:
                print("[ModelRegistry] No parent version to rollback to")
                return False

        if target_version not in self.versions:
            print(f"[ModelRegistry] Target version {target_version} not found")
            return False

        target = self.versions[target_version]

        print(f"[ModelRegistry] Rolling back from {current_prod.version} "
              f"to {target_version}")
        print(f"[ModelRegistry] Reason: {reason}")

        # Record rollback
        rollback_event = RollbackEvent(
            timestamp=datetime.now().isoformat(),
            from_version=current_prod.version,
            to_version=target_version,
            reason=reason,
            triggered_by=triggered_by,
            metrics_before={
                'validation_accuracy': current_prod.validation_accuracy,
                'old_task_accuracy': current_prod.old_task_accuracy,
            },
            metrics_after={
                'validation_accuracy': target.validation_accuracy,
                'old_task_accuracy': target.old_task_accuracy,
            }
        )
        self.rollbacks.append(rollback_event)

        # Update stages
        self.versions[current_prod.version].stage = DeploymentStage.ROLLED_BACK
        self.versions[current_prod.version].traffic_percentage = 0.0

        self.versions[target_version].stage = DeploymentStage.PRODUCTION
        self.versions[target_version].traffic_percentage = 100.0

        self._save_registry()

        print(f"[ModelRegistry] Rollback complete: now running {target_version}")

        return True

    def get_production_model(self) -> Optional[nn.Module]:
        """
        Load current production model.

        Returns:
            Model or None if no production model
        """
        current_prod = self._get_current_production()
        if not current_prod:
            return None

        return self.load_model(current_prod.version)

    def load_model(
        self,
        version: str,
        model_class: Optional[type] = None
    ) -> Optional[nn.Module]:
        """
        Load model for specific version with proper architecture.

        Constructs the model using stored hyperparameters and loads weights.

        Args:
            version: Version to load
            model_class: Deprecated, ignored (we use stored hyperparameters)

        Returns:
            Loaded model or None if not found
        """
        if version not in self.versions:
            print(f"[ModelRegistry] Version {version} not found")
            return None

        model_version = self.versions[version]

        try:
            from nerion_digital_physicist.agent.brain import build_gnn

            # Build model with stored hyperparameters
            model = build_gnn(
                architecture=model_version.architecture.lower(),
                num_node_features=model_version.num_node_features,
                hidden_channels=model_version.hidden_channels,
                num_classes=2,  # Binary classification
                num_layers=model_version.num_layers,
                residual=model_version.residual,
                dropout=model_version.dropout,
                attention_heads=model_version.attention_heads,
                use_graphcodebert=model_version.use_graphcodebert
            )

            # Load weights
            state_dict = torch.load(model_version.model_path, weights_only=False)
            model.load_state_dict(state_dict)

            print(f"[ModelRegistry] Loaded model {version} "
                  f"({model_version.architecture}, "
                  f"{model_version.num_node_features}→{model_version.hidden_channels}, "
                  f"{model_version.num_layers} layers)")

            return model

        except Exception as e:
            print(f"[ModelRegistry] Failed to load model {version}: {e}")
            return None

    def list_versions(
        self,
        stage: Optional[DeploymentStage] = None
    ) -> List[ModelVersion]:
        """
        List all versions, optionally filtered by stage.

        Args:
            stage: Filter by deployment stage

        Returns:
            List of model versions
        """
        versions = list(self.versions.values())

        if stage:
            versions = [v for v in versions if v.stage == stage]

        # Sort by version (newest first)
        versions.sort(key=lambda v: v.created_at, reverse=True)

        return versions

    def get_version_info(self, version: str) -> Optional[ModelVersion]:
        """Get detailed info for specific version"""
        return self.versions.get(version)

    def compare_versions(
        self,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """
        Compare two model versions.

        Args:
            version1: First version
            version2: Second version

        Returns:
            Comparison dict with metrics differences
        """
        if version1 not in self.versions or version2 not in self.versions:
            return {}

        v1 = self.versions[version1]
        v2 = self.versions[version2]

        return {
            'version1': version1,
            'version2': version2,
            'validation_accuracy_diff': v2.validation_accuracy - v1.validation_accuracy,
            'old_task_accuracy_diff': v2.old_task_accuracy - v1.old_task_accuracy,
            'parameters_diff': v2.num_parameters - v1.num_parameters,
            'training_samples_diff': v2.training_samples - v1.training_samples,
        }

    def _get_current_production(self) -> Optional[ModelVersion]:
        """Get current production model version"""
        for version in self.versions.values():
            if version.stage == DeploymentStage.PRODUCTION:
                return version
        return None

    def _next_version(
        self,
        current: Optional[ModelVersion],
        update_method: str
    ) -> str:
        """Determine next semantic version"""
        if not current:
            return "1.0.0"

        major, minor, patch = map(int, current.version.split('.'))

        if update_method == "incremental" or update_method == "maml":
            # Minor update: incremental learning
            minor += 1
            patch = 0
        else:
            # Patch update: full retrain or bug fix
            patch += 1

        return f"{major}.{minor}.{patch}"

    def _validate_for_production(self, version: ModelVersion) -> bool:
        """
        Validate model is ready for production.

        Checks:
        - Validation accuracy threshold
        - Forgetting threshold (old task accuracy)
        - No significant degradation from current prod

        Returns:
            True if ready for production
        """
        # Minimum thresholds
        MIN_VALIDATION_ACC = 0.60
        MIN_OLD_TASK_ACC = 0.55
        MAX_FORGETTING = 0.10

        if version.validation_accuracy < MIN_VALIDATION_ACC:
            print(f"[ModelRegistry] Validation accuracy too low: "
                  f"{version.validation_accuracy:.4f} < {MIN_VALIDATION_ACC}")
            return False

        if version.old_task_accuracy < MIN_OLD_TASK_ACC:
            print(f"[ModelRegistry] Old task accuracy too low (forgetting): "
                  f"{version.old_task_accuracy:.4f} < {MIN_OLD_TASK_ACC}")
            return False

        # Check against current production
        current_prod = self._get_current_production()
        if current_prod:
            forgetting = current_prod.old_task_accuracy - version.old_task_accuracy
            if forgetting > MAX_FORGETTING:
                print(f"[ModelRegistry] Too much forgetting: "
                      f"{forgetting:.4f} > {MAX_FORGETTING}")
                return False

        return True

    def _load_registry(self):
        """Load registry from disk"""
        if self.metadata_file.exists():
            data = json.loads(self.metadata_file.read_text())
            self.versions = {
                k: ModelVersion.from_dict(v)
                for k, v in data.get('versions', {}).items()
            }

        if self.rollback_log.exists():
            rollback_data = json.loads(self.rollback_log.read_text())
            self.rollbacks = [
                RollbackEvent(**r) for r in rollback_data.get('rollbacks', [])
            ]

    def _save_registry(self):
        """Save registry to disk"""
        data = {
            'versions': {
                k: v.to_dict() for k, v in self.versions.items()
            }
        }
        self.metadata_file.write_text(json.dumps(data, indent=2))

        rollback_data = {
            'rollbacks': [asdict(r) for r in self.rollbacks]
        }
        self.rollback_log.write_text(json.dumps(rollback_data, indent=2))


# Example usage
def example_workflow():
    """Example model registry workflow"""
    from pathlib import Path

    # Initialize registry
    registry = ModelRegistry(Path("out/models/registry"))

    # Simulate model training and registration
    print("\n=== Register Initial Model ===")
    # model = train_initial_model()  # Placeholder
    # version_1 = registry.register(
    #     model=model,
    #     validation_accuracy=0.70,
    #     old_task_accuracy=0.70,
    #     update_method="full_retrain",
    #     notes="Initial production model"
    # )

    # Promote to production
    # registry.promote(version_1.version, DeploymentStage.PRODUCTION)

    print("\n=== Incremental Update ===")
    # updated_model = incremental_update(model)  # Placeholder
    # version_2 = registry.register(
    #     model=updated_model,
    #     validation_accuracy=0.75,
    #     old_task_accuracy=0.68,
    #     update_method="incremental",
    #     notes="Learned from 50 production bugs"
    # )

    # Canary deployment
    # registry.promote(version_2.version, DeploymentStage.CANARY, traffic_percentage=10.0)

    print("\n=== List Versions ===")
    for version in registry.list_versions():
        print(f"{version.version}: {version.stage.value} "
              f"(val_acc={version.validation_accuracy:.4f})")


if __name__ == "__main__":
    example_workflow()
