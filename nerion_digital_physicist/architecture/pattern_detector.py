"""
Architectural Pattern Detector

Identifies common architectural patterns in codebases:
- MVC (Model-View-Controller)
- Layered Architecture
- Microservices
- Repository Pattern
- Factory Pattern
- Singleton Pattern
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Set, Optional

from .graph_builder import ArchitectureGraph, Module


class PatternType(Enum):
    """Types of architectural patterns"""
    MVC = "mvc"
    LAYERED = "layered"
    MICROSERVICES = "microservices"
    REPOSITORY = "repository"
    FACTORY = "factory"
    SINGLETON = "singleton"
    OBSERVER = "observer"
    STRATEGY = "strategy"


@dataclass
class ArchitecturalPattern:
    """Detected architectural pattern"""
    pattern_type: PatternType
    confidence: float          # 0.0 to 1.0
    modules: List[str]         # Modules involved in pattern
    evidence: List[str]        # Evidence supporting detection
    violations: List[str]      # Violations of pattern principles


class PatternDetector:
    """
    Detects architectural patterns in code.

    Usage:
        >>> detector = PatternDetector()
        >>> patterns = detector.detect_patterns(graph)
        >>> for pattern in patterns:
        ...     print(f"{pattern.pattern_type}: {pattern.confidence:.2f}")
    """

    def __init__(self):
        self.detected_patterns: List[ArchitecturalPattern] = []

    def detect_patterns(self, graph: ArchitectureGraph) -> List[ArchitecturalPattern]:
        """
        Detect all architectural patterns in the graph.

        Args:
            graph: Architectural graph

        Returns:
            List of detected patterns
        """
        patterns = []

        # Detect structural patterns
        patterns.extend(self._detect_mvc(graph))
        patterns.extend(self._detect_layered(graph))
        patterns.extend(self._detect_repository(graph))

        # Detect design patterns
        patterns.extend(self._detect_factory(graph))
        patterns.extend(self._detect_singleton(graph))
        patterns.extend(self._detect_observer(graph))

        self.detected_patterns = patterns
        return patterns

    def _detect_mvc(self, graph: ArchitectureGraph) -> List[ArchitecturalPattern]:
        """
        Detect MVC (Model-View-Controller) pattern.

        Evidence:
        - Modules named/containing 'model', 'view', 'controller'
        - Models have no UI dependencies
        - Controllers mediate between models and views
        """
        models = []
        views = []
        controllers = []

        for module_name, module in graph.modules.items():
            name_lower = module_name.lower()

            if 'model' in name_lower and 'view' not in name_lower:
                models.append(module_name)
            elif 'view' in name_lower or 'template' in name_lower or 'ui' in name_lower:
                views.append(module_name)
            elif 'controller' in name_lower or 'handler' in name_lower:
                controllers.append(module_name)

        # Need all three components for MVC
        if not (models and views and controllers):
            return []

        # Calculate confidence based on:
        # 1. Presence of all three components
        # 2. Proper separation (models don't import views)
        # 3. Controllers connect models and views

        evidence = []
        violations = []
        confidence = 0.0

        # Check component presence
        if models:
            evidence.append(f"Found {len(models)} model modules")
            confidence += 0.3
        if views:
            evidence.append(f"Found {len(views)} view modules")
            confidence += 0.3
        if controllers:
            evidence.append(f"Found {len(controllers)} controller modules")
            confidence += 0.3

        # Check separation: models shouldn't import views
        models_import_views = False
        for model in models:
            related = graph.find_related_modules(model, "upstream", max_distance=2)
            if any(v in related for v in views):
                models_import_views = True
                violations.append(f"{model} imports view modules (violates MVC)")

        if not models_import_views:
            evidence.append("Models properly separated from views")
            confidence += 0.1

        # Only return if confidence is high enough
        if confidence < 0.5:
            return []

        pattern = ArchitecturalPattern(
            pattern_type=PatternType.MVC,
            confidence=min(confidence, 1.0),
            modules=models + views + controllers,
            evidence=evidence,
            violations=violations
        )

        return [pattern]

    def _detect_layered(self, graph: ArchitectureGraph) -> List[ArchitecturalPattern]:
        """
        Detect layered architecture pattern.

        Evidence:
        - Modules organized in layers (presentation, business, data)
        - Lower layers don't depend on upper layers
        - Clear separation of concerns
        """
        # Identify layers by common naming conventions
        presentation_layer = []
        business_layer = []
        data_layer = []

        for module_name in graph.modules.keys():
            name_lower = module_name.lower()

            if any(x in name_lower for x in ['api', 'view', 'controller', 'handler', 'ui']):
                presentation_layer.append(module_name)
            elif any(x in name_lower for x in ['service', 'business', 'logic', 'core']):
                business_layer.append(module_name)
            elif any(x in name_lower for x in ['data', 'repository', 'dao', 'model', 'db']):
                data_layer.append(module_name)

        # Need at least 2 layers
        non_empty_layers = sum([bool(presentation_layer), bool(business_layer), bool(data_layer)])
        if non_empty_layers < 2:
            return []

        evidence = []
        violations = []
        confidence = 0.0

        # Check layer presence
        if presentation_layer:
            evidence.append(f"Presentation layer: {len(presentation_layer)} modules")
            confidence += 0.25
        if business_layer:
            evidence.append(f"Business layer: {len(business_layer)} modules")
            confidence += 0.25
        if data_layer:
            evidence.append(f"Data layer: {len(data_layer)} modules")
            confidence += 0.25

        # Check proper layering: data doesn't depend on presentation
        data_depends_on_presentation = False
        for data_mod in data_layer:
            related = graph.find_related_modules(data_mod, "upstream", max_distance=3)
            if any(p in related for p in presentation_layer):
                data_depends_on_presentation = True
                violations.append(f"{data_mod} depends on presentation layer (violates layering)")

        if not data_depends_on_presentation:
            evidence.append("Data layer properly separated from presentation")
            confidence += 0.25

        if confidence < 0.4:
            return []

        pattern = ArchitecturalPattern(
            pattern_type=PatternType.LAYERED,
            confidence=min(confidence, 1.0),
            modules=presentation_layer + business_layer + data_layer,
            evidence=evidence,
            violations=violations
        )

        return [pattern]

    def _detect_repository(self, graph: ArchitectureGraph) -> List[ArchitecturalPattern]:
        """
        Detect Repository pattern.

        Evidence:
        - Classes/modules named Repository, Store, DAO
        - Encapsulate data access logic
        - Provide abstract interface to data layer
        """
        repositories = []

        for module_name, module in graph.modules.items():
            name_lower = module_name.lower()

            # Check module name
            if any(x in name_lower for x in ['repository', 'repo', 'store', 'dao']):
                repositories.append(module_name)
                continue

            # Check class names
            for cls in module.classes:
                if any(x in cls.lower() for x in ['repository', 'store', 'dao']):
                    repositories.append(module_name)
                    break

        if not repositories:
            return []

        evidence = []
        confidence = 0.0

        evidence.append(f"Found {len(repositories)} repository modules")
        confidence = min(0.7 + len(repositories) * 0.1, 1.0)

        # Check if repositories have save/find/delete methods
        repo_methods = ['save', 'find', 'get', 'delete', 'update', 'add', 'remove']
        repos_with_crud = 0

        for repo_name in repositories:
            if repo_name in graph.modules:
                module = graph.modules[repo_name]
                func_names = [f.lower() for f in module.functions]
                if any(method in ' '.join(func_names) for method in repo_methods):
                    repos_with_crud += 1

        if repos_with_crud > 0:
            evidence.append(f"{repos_with_crud} repositories with CRUD operations")
            confidence = min(confidence + 0.1, 1.0)

        pattern = ArchitecturalPattern(
            pattern_type=PatternType.REPOSITORY,
            confidence=confidence,
            modules=repositories,
            evidence=evidence,
            violations=[]
        )

        return [pattern]

    def _detect_factory(self, graph: ArchitectureGraph) -> List[ArchitecturalPattern]:
        """
        Detect Factory pattern.

        Evidence:
        - Classes/functions named Factory, Builder, Creator
        - Create instances of other classes
        """
        factories = []

        for module_name, module in graph.modules.items():
            # Check for Factory/Builder in name
            for cls in module.classes:
                if any(x in cls.lower() for x in ['factory', 'builder', 'creator']):
                    factories.append(module_name)
                    break

            # Check for functions like create_*, build_*, make_*
            for func in module.functions:
                func_lower = func.lower()
                if any(func_lower.startswith(x) for x in ['create_', 'build_', 'make_']):
                    if module_name not in factories:
                        factories.append(module_name)
                    break

        if not factories:
            return []

        pattern = ArchitecturalPattern(
            pattern_type=PatternType.FACTORY,
            confidence=min(0.6 + len(factories) * 0.05, 0.95),
            modules=factories,
            evidence=[f"Found {len(factories)} factory modules/classes"],
            violations=[]
        )

        return [pattern]

    def _detect_singleton(self, graph: ArchitectureGraph) -> List[ArchitecturalPattern]:
        """
        Detect Singleton pattern.

        Evidence:
        - Classes with getInstance() or similar methods
        - Private constructors (hard to detect in Python)
        - Global instance variables
        """
        singletons = []

        for module_name, module in graph.modules.items():
            # Check for Singleton in class name
            for cls in module.classes:
                if 'singleton' in cls.lower():
                    singletons.append(module_name)
                    break

            # Check for getInstance pattern
            for func in module.functions:
                if func.lower() in ['get_instance', 'getinstance', 'instance']:
                    if module_name not in singletons:
                        singletons.append(module_name)
                    break

        if not singletons:
            return []

        pattern = ArchitecturalPattern(
            pattern_type=PatternType.SINGLETON,
            confidence=min(0.7 + len(singletons) * 0.05, 0.95),
            modules=singletons,
            evidence=[f"Found {len(singletons)} singleton modules"],
            violations=[]
        )

        return [pattern]

    def _detect_observer(self, graph: ArchitectureGraph) -> List[ArchitecturalPattern]:
        """
        Detect Observer pattern.

        Evidence:
        - Classes with subscribe/notify/emit methods
        - Event/listener/handler patterns
        """
        observers = []

        for module_name, module in graph.modules.items():
            # Check for Observer/Subject/Listener in class names
            for cls in module.classes:
                cls_lower = cls.lower()
                if any(x in cls_lower for x in ['observer', 'subject', 'listener', 'subscriber']):
                    observers.append(module_name)
                    break

            # Check for subscribe/notify/emit methods
            observer_methods = ['subscribe', 'notify', 'emit', 'on', 'trigger', 'dispatch']
            for func in module.functions:
                func_lower = func.lower()
                if any(x in func_lower for x in observer_methods):
                    if module_name not in observers:
                        observers.append(module_name)
                    break

        if not observers:
            return []

        pattern = ArchitecturalPattern(
            pattern_type=PatternType.OBSERVER,
            confidence=min(0.65 + len(observers) * 0.05, 0.95),
            modules=observers,
            evidence=[f"Found {len(observers)} observer/event modules"],
            violations=[]
        )

        return [pattern]

    def generate_report(self) -> str:
        """Generate a human-readable pattern detection report"""
        if not self.detected_patterns:
            return "No architectural patterns detected."

        report = "=== Architectural Pattern Detection Report ===\n\n"

        for pattern in sorted(self.detected_patterns, key=lambda p: p.confidence, reverse=True):
            report += f"{pattern.pattern_type.value.upper()}\n"
            report += f"Confidence: {pattern.confidence:.1%}\n"
            report += f"Modules: {len(pattern.modules)}\n"

            if pattern.evidence:
                report += "Evidence:\n"
                for evidence in pattern.evidence:
                    report += f"  - {evidence}\n"

            if pattern.violations:
                report += "Violations:\n"
                for violation in pattern.violations:
                    report += f"  ⚠️ {violation}\n"

            report += "\n"

        return report


# Example usage
if __name__ == "__main__":
    from pathlib import Path
    from .graph_builder import ArchitecturalGraphBuilder

    # Build graph
    builder = ArchitecturalGraphBuilder()
    graph = builder.build_from_directory(
        Path("/Users/ed/Nerion-V2/nerion_digital_physicist"),
        package_name="nerion_digital_physicist"
    )

    # Detect patterns
    detector = PatternDetector()
    patterns = detector.detect_patterns(graph)

    print(detector.generate_report())
