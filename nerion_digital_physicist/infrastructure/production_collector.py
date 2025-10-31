"""
Production Feedback Collector

Collects bugs detected in production and stores them in ReplayStore for continuous learning.

Integration points:
- Daemon: watch_codebase() and monitor_health() detect bugs
- ReplayStore: Stores experiences with priority sampling
- Model: Uses current GNN to predict and calculate surprise
- Auto-Curriculum: High-surprise bugs trigger lesson generation

Surprise Scoring:
- surprise = |model_confidence - ground_truth|
- High surprise → high priority in replay buffer
- Surprise guides curriculum generation and sampling
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch

# Import existing ReplayStore infrastructure
from nerion_digital_physicist.infrastructure.memory import ReplayStore, Experience


@dataclass
class ProductionBug:
    """A bug detected in production"""
    bug_id: str
    source_code: str
    file_path: str
    language: str
    bug_type: str                     # syntax_error, logic_error, security, performance, etc.
    severity: str                     # low, medium, high, critical

    # Model predictions
    model_prediction: Optional[int] = None
    model_confidence: Optional[float] = None

    # Ground truth
    ground_truth: Optional[int] = None

    # Context
    environment: Dict[str, Any] = None
    stack_trace: Optional[str] = None
    timestamp: str = None

    # Impact
    production_impact: Optional[str] = None  # user_facing, silent, crash, etc.

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.environment is None:
            self.environment = {}


@dataclass
class FeedbackMetrics:
    """Metrics tracked by production feedback collector"""
    total_bugs_collected: int = 0
    high_surprise_bugs: int = 0      # surprise > 0.5
    by_bug_type: Dict[str, int] = None
    by_severity: Dict[str, int] = None
    avg_surprise: float = 0.0
    last_collection_time: Optional[str] = None

    def __post_init__(self):
        if self.by_bug_type is None:
            self.by_bug_type = {}
        if self.by_severity is None:
            self.by_severity = {}


class ProductionFeedbackCollector:
    """
    Collects production bugs and stores in ReplayStore for continuous learning.

    Usage:
        >>> replay_store = ReplayStore(Path("data/replay"))
        >>> collector = ProductionFeedbackCollector(replay_store, model)
        >>>
        >>> # Daemon detects bug
        >>> bug = ProductionBug(
        ...     bug_id="prod_001",
        ...     source_code=buggy_code,
        ...     file_path="app/handler.py",
        ...     language="python",
        ...     bug_type="logic_error",
        ...     severity="high"
        ... )
        >>>
        >>> # Collect with model prediction
        >>> surprise = collector.collect_bug(bug, graph, ground_truth=1)
    """

    def __init__(
        self,
        replay_store: ReplayStore,
        model: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize production feedback collector.

        Args:
            replay_store: Existing ReplayStore instance
            model: Current GNN model for predictions
            device: Device for model inference
        """
        self.replay_store = replay_store
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.model:
            self.model.to(self.device)
            self.model.eval()

        self.metrics = FeedbackMetrics()

    def collect_bug(
        self,
        bug: ProductionBug,
        graph: Any,
        ground_truth: Optional[int] = None
    ) -> float:
        """
        Collect a production bug and store in ReplayStore.

        Args:
            bug: Production bug information
            graph: AST graph representation
            ground_truth: Actual quality label (if available)

        Returns:
            Surprise score (for prioritization)
        """
        # Get model prediction if model available
        if self.model and bug.model_prediction is None:
            with torch.no_grad():
                logits = self.model(graph.to(self.device))
                probs = torch.softmax(logits, dim=0)
                bug.model_prediction = probs.argmax(dim=0).item()
                bug.model_confidence = probs.max().item()

        # Update ground truth if provided
        if ground_truth is not None:
            bug.ground_truth = ground_truth

        # Calculate surprise score
        surprise = self._calculate_surprise(bug)

        # Store in ReplayStore
        self._store_in_replay(bug, surprise)

        # Update metrics
        self._update_metrics(bug, surprise)

        return surprise

    def _calculate_surprise(self, bug: ProductionBug) -> float:
        """
        Calculate surprise score for prioritization.

        Surprise = confidence mismatch between prediction and ground truth
        High surprise → model is very confident but wrong
        Low surprise → model is correct or uncertain

        Args:
            bug: Production bug

        Returns:
            Surprise score [0, 1]
        """
        if bug.model_confidence is None or bug.ground_truth is None:
            # No prediction available, use medium surprise
            return 0.5

        if bug.model_prediction is None:
            return 0.5

        # High surprise if model was confident but wrong
        if bug.model_prediction != bug.ground_truth:
            surprise = bug.model_confidence  # More confident wrong → higher surprise
        else:
            # Model correct but low confidence is also interesting
            surprise = 1.0 - bug.model_confidence

        # Boost surprise for high-severity bugs
        severity_boost = {
            'critical': 1.3,
            'high': 1.2,
            'medium': 1.0,
            'low': 0.8
        }
        surprise *= severity_boost.get(bug.severity, 1.0)

        return min(surprise, 1.0)  # Clamp to [0, 1]

    def _store_in_replay(self, bug: ProductionBug, surprise: float):
        """
        Store bug in ReplayStore.

        EXTENDS existing ReplayStore metadata with production context.
        """
        # Generate experience ID
        experience_id = self._generate_experience_id(bug)

        # Prepare metadata (EXTENDS existing metadata schema)
        metadata = {
            # Existing fields
            'source_code': bug.source_code,
            'file_path': bug.file_path,
            'language': bug.language,
            'category': bug.bug_type,

            # NEW production context fields
            'model_prediction': bug.model_prediction,
            'model_confidence': bug.model_confidence,
            'ground_truth': bug.ground_truth,
            'production_impact': bug.production_impact,
            'bug_type': bug.bug_type,
            'severity': bug.severity,
            'environment': bug.environment,
            'stack_trace': bug.stack_trace,
            'timestamp': bug.timestamp,

            # Provenance tracking
            'provenance': 'production_bug',
            'collection_method': 'daemon_monitor'
        }

        # Store in ReplayStore (REUSES existing infrastructure)
        self.replay_store.append(
            task_id=bug.bug_id,
            template_id="production_bug",
            status="failed" if bug.ground_truth == 1 else "solved",
            surprise=surprise,
            metadata=metadata
        )

    def _generate_experience_id(self, bug: ProductionBug) -> str:
        """Generate unique experience ID from bug content"""
        content = f"{bug.source_code}_{bug.file_path}_{bug.timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _update_metrics(self, bug: ProductionBug, surprise: float):
        """Update collection metrics"""
        self.metrics.total_bugs_collected += 1

        if surprise > 0.5:
            self.metrics.high_surprise_bugs += 1

        # Update by bug type
        if bug.bug_type not in self.metrics.by_bug_type:
            self.metrics.by_bug_type[bug.bug_type] = 0
        self.metrics.by_bug_type[bug.bug_type] += 1

        # Update by severity
        if bug.severity not in self.metrics.by_severity:
            self.metrics.by_severity[bug.severity] = 0
        self.metrics.by_severity[bug.severity] += 1

        # Update average surprise (running average)
        n = self.metrics.total_bugs_collected
        self.metrics.avg_surprise = (
            (self.metrics.avg_surprise * (n - 1) + surprise) / n
        )

        self.metrics.last_collection_time = datetime.now().isoformat()

    def get_high_surprise_bugs(self, k: int = 50) -> List[Experience]:
        """
        Get high-surprise bugs from ReplayStore.

        Args:
            k: Number of bugs to retrieve

        Returns:
            List of high-priority experiences
        """
        # Use ReplayStore's existing priority sampling
        experiences = self.replay_store.sample(k=k, strategy="priority")

        # Filter to production bugs only
        production_bugs = [
            exp for exp in experiences
            if exp.metadata.get('provenance') == 'production_bug'
        ]

        return production_bugs[:k]

    def get_bugs_by_type(self, bug_type: str, k: int = 20) -> List[Experience]:
        """
        Get bugs of specific type.

        Args:
            bug_type: Bug type to retrieve
            k: Number of bugs

        Returns:
            List of bugs matching type
        """
        # Use ReplayStore's load_all and filter
        all_experiences = self.replay_store.load_all()

        matching_bugs = [
            exp for exp in all_experiences
            if exp.metadata.get('bug_type') == bug_type
        ]

        # Sort by surprise (priority)
        matching_bugs.sort(key=lambda x: x.surprise or 0.0, reverse=True)

        return matching_bugs[:k]

    def get_metrics(self) -> FeedbackMetrics:
        """Get current collection metrics"""
        return self.metrics

    def should_trigger_learning_cycle(self) -> bool:
        """
        Check if enough bugs accumulated to trigger learning cycle.

        Triggers when:
        - At least 50 bugs collected
        - Or at least 10 high-surprise bugs
        - Or 24 hours since last collection

        Returns:
            True if should trigger learning
        """
        if self.metrics.total_bugs_collected >= 50:
            return True

        if self.metrics.high_surprise_bugs >= 10:
            return True

        # Check time since last collection
        if self.metrics.last_collection_time:
            last_time = datetime.fromisoformat(self.metrics.last_collection_time)
            hours_since = (datetime.now() - last_time).total_seconds() / 3600
            if hours_since >= 24:
                return True

        return False

    def reset_metrics(self):
        """Reset metrics after learning cycle"""
        self.metrics = FeedbackMetrics()


# Daemon integration helper
class DaemonBugDetector:
    """
    Helper for daemon to detect bugs and trigger feedback collection.

    Integrates with:
    - daemon.watch_codebase() for file change monitoring
    - daemon.monitor_health() for threat detection
    """

    def __init__(
        self,
        collector: ProductionFeedbackCollector,
        codebase_path: Path
    ):
        """
        Initialize daemon bug detector.

        Args:
            collector: Production feedback collector
            codebase_path: Path to monitored codebase
        """
        self.collector = collector
        self.codebase_path = codebase_path

    async def detect_and_collect(self, file_path: Path) -> Optional[float]:
        """
        Detect bug in file and collect feedback.

        This would be called from daemon.watch_codebase() when
        file changes are detected.

        Args:
            file_path: Path to file to analyze

        Returns:
            Surprise score if bug detected, None otherwise
        """
        # TODO: Integrate with actual bug detection
        # For now, this is a placeholder showing the workflow

        # 1. Read file
        source_code = file_path.read_text()

        # 2. Parse to AST graph
        # from nerion_digital_physicist.agent.data import parse_code_to_graph
        # graph = parse_code_to_graph(source_code, language="python")

        # 3. Get model prediction
        # prediction, confidence = self.collector.model.predict(graph)

        # 4. If suspicious, collect
        # if confidence < 0.5 or prediction == 1:  # Low quality
        #     bug = ProductionBug(
        #         bug_id=f"daemon_{time.time()}",
        #         source_code=source_code,
        #         file_path=str(file_path),
        #         language="python",
        #         bug_type="unknown",
        #         severity="medium"
        #     )
        #     surprise = self.collector.collect_bug(bug, graph, ground_truth=None)
        #     return surprise

        return None

    def check_trigger_learning(self) -> bool:
        """Check if should trigger continuous learning cycle"""
        return self.collector.should_trigger_learning_cycle()
