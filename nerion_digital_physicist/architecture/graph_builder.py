"""
Architectural Graph Builder

Builds repository-wide dependency graphs for system-level understanding.
Enables impact analysis, circular dependency detection, and architectural pattern recognition.
"""
from __future__ import annotations

import ast
import hashlib
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any

import networkx as nx


class DependencyType(Enum):
    """Types of dependencies between modules"""
    IMPORT = "import"              # from module import X
    FUNCTION_CALL = "function_call"  # calls function from module
    INHERITANCE = "inheritance"     # inherits from class
    INSTANTIATION = "instantiation" # creates instance
    TYPE_HINT = "type_hint"        # uses as type annotation


@dataclass
class Module:
    """Represents a Python module in the architecture"""
    path: Path
    name: str                      # Fully qualified name (e.g., app.core.utils)
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    loc: int = 0                  # Lines of code
    complexity: float = 0.0       # Cyclomatic complexity
    last_modified: float = 0.0    # Timestamp
    content_hash: str = ""        # SHA256 of content


@dataclass
class Dependency:
    """Represents a dependency between two modules"""
    source: str                   # Source module name
    target: str                   # Target module name
    dep_type: DependencyType
    line_number: int
    strength: float = 1.0         # Coupling strength (0-1)


class ArchitectureGraph:
    """
    Repository-wide architectural graph.

    Provides:
    - Dependency visualization
    - Circular dependency detection
    - Impact analysis
    - Module search
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.modules: Dict[str, Module] = {}
        self.dependencies: List[Dependency] = []
        self._impact_cache: Dict[str, Set[str]] = {}

    def add_module(self, module: Module):
        """Add a module to the graph"""
        self.modules[module.name] = module
        self.graph.add_node(
            module.name,
            path=str(module.path),
            classes=module.classes,
            functions=module.functions,
            loc=module.loc,
            complexity=module.complexity
        )

    def add_dependency(self, dep: Dependency):
        """Add a dependency edge"""
        self.dependencies.append(dep)

        # Add edge with metadata
        if self.graph.has_edge(dep.source, dep.target):
            # Update existing edge (increase strength)
            edge_data = self.graph[dep.source][dep.target]
            edge_data['strength'] += dep.strength
            edge_data['types'].append(dep.dep_type)
        else:
            # Create new edge
            self.graph.add_edge(
                dep.source,
                dep.target,
                strength=dep.strength,
                types=[dep.dep_type]
            )

        # Invalidate impact cache
        self._impact_cache.clear()

    def find_circular_dependencies(self) -> List[List[str]]:
        """
        Find all circular dependencies (cycles) in the graph.

        Returns:
            List of cycles, where each cycle is a list of module names
        """
        try:
            cycles = list(nx.simple_cycles(self.graph))
            return cycles
        except:
            return []

    def compute_impact(self, module_name: str, max_depth: int = 10) -> Set[str]:
        """
        Compute impact set: all modules affected by changes to this module.

        Uses BFS to find all downstream dependencies.

        Args:
            module_name: Module to analyze
            max_depth: Maximum depth to traverse

        Returns:
            Set of module names that would be affected
        """
        if module_name in self._impact_cache:
            return self._impact_cache[module_name]

        if module_name not in self.graph:
            return set()

        impacted = set()
        queue = deque([(module_name, 0)])
        visited = {module_name}

        while queue:
            current, depth = queue.popleft()

            if depth >= max_depth:
                continue

            # Get all modules that depend on current
            for successor in self.graph.successors(current):
                if successor not in visited:
                    visited.add(successor)
                    impacted.add(successor)
                    queue.append((successor, depth + 1))

        self._impact_cache[module_name] = impacted
        return impacted

    def find_related_modules(
        self,
        module_name: str,
        relation: str = "all",
        max_distance: int = 2
    ) -> Set[str]:
        """
        Find modules related to the given module.

        Args:
            module_name: Module to analyze
            relation: "all", "upstream", "downstream", or "sibling"
            max_distance: Maximum graph distance

        Returns:
            Set of related module names
        """
        if module_name not in self.graph:
            return set()

        related = set()

        if relation in ["all", "upstream"]:
            # Find upstream dependencies (modules this depends on)
            for predecessor in self.graph.predecessors(module_name):
                related.add(predecessor)
                if max_distance > 1:
                    # Recursively find upstream
                    related.update(
                        self.find_related_modules(predecessor, "upstream", max_distance - 1)
                    )

        if relation in ["all", "downstream"]:
            # Find downstream dependents (modules that depend on this)
            for successor in self.graph.successors(module_name):
                related.add(successor)
                if max_distance > 1:
                    # Recursively find downstream
                    related.update(
                        self.find_related_modules(successor, "downstream", max_distance - 1)
                    )

        if relation in ["all", "sibling"]:
            # Find siblings (modules with shared dependencies)
            upstream = set(self.graph.predecessors(module_name))
            for node in self.graph.nodes():
                if node != module_name:
                    node_upstream = set(self.graph.predecessors(node))
                    if upstream & node_upstream:  # Shared dependencies
                        related.add(node)

        return related

    def search_by_functionality(self, query: str) -> List[Tuple[str, float]]:
        """
        Semantic search for modules by functionality.

        Uses module names, class names, and function names for matching.

        Args:
            query: Search query

        Returns:
            List of (module_name, relevance_score) tuples, sorted by score
        """
        results = []
        query_lower = query.lower()
        query_parts = set(query_lower.split())

        for module_name, module in self.modules.items():
            score = 0.0

            # Check module name
            if query_lower in module_name.lower():
                score += 10.0

            # Check word overlap in module name
            module_parts = set(module_name.lower().replace('.', ' ').replace('_', ' ').split())
            overlap = len(query_parts & module_parts)
            score += overlap * 2.0

            # Check classes
            for cls in module.classes:
                if query_lower in cls.lower():
                    score += 5.0

            # Check functions
            for func in module.functions:
                if query_lower in func.lower():
                    score += 3.0

            if score > 0:
                results.append((module_name, score))

        # Sort by relevance
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get architecture statistics"""
        return {
            'num_modules': len(self.modules),
            'num_dependencies': len(self.dependencies),
            'num_edges': self.graph.number_of_edges(),
            'avg_degree': sum(dict(self.graph.degree()).values()) / max(len(self.modules), 1),
            'num_cycles': len(self.find_circular_dependencies()),
            'largest_module_loc': max((m.loc for m in self.modules.values()), default=0),
            'avg_complexity': sum(m.complexity for m in self.modules.values()) / max(len(self.modules), 1)
        }


class ArchitecturalGraphBuilder:
    """
    Builds architectural graphs from Python repositories.

    Usage:
        >>> builder = ArchitecturalGraphBuilder()
        >>> graph = builder.build_from_directory(Path("my_repo"))
        >>> cycles = graph.find_circular_dependencies()
        >>> impact = graph.compute_impact("my_module.utils")
    """

    def __init__(self, exclude_patterns: Optional[List[str]] = None):
        """
        Initialize builder.

        Args:
            exclude_patterns: Glob patterns to exclude (e.g., ['test_*.py', '*_test.py'])
        """
        self.exclude_patterns = exclude_patterns or ['test_*.py', '*_test.py', 'conftest.py']

    def build_from_directory(
        self,
        root_path: Path,
        package_name: Optional[str] = None,
        max_files: int = 10000
    ) -> ArchitectureGraph:
        """
        Build architectural graph from a directory.

        Args:
            root_path: Root directory to scan
            package_name: Base package name (inferred if None)
            max_files: Maximum files to process

        Returns:
            Architectural graph
        """
        start_time = time.time()
        graph = ArchitectureGraph()

        print(f"[ArchGraphBuilder] Scanning {root_path}...")

        # Find all Python files
        python_files = self._find_python_files(root_path, max_files)
        print(f"[ArchGraphBuilder] Found {len(python_files)} Python files")

        # Infer package name if not provided
        if package_name is None:
            package_name = root_path.name

        # Parse each file and extract modules
        file_to_module = {}
        for file_path in python_files:
            module_name = self._get_module_name(file_path, root_path, package_name)
            module = self._parse_module(file_path, module_name)

            if module:
                graph.add_module(module)
                file_to_module[file_path] = module_name

        print(f"[ArchGraphBuilder] Parsed {len(graph.modules)} modules")

        # Extract dependencies
        for file_path, module_name in file_to_module.items():
            dependencies = self._extract_dependencies(file_path, module_name, graph.modules)
            for dep in dependencies:
                graph.add_dependency(dep)

        elapsed = time.time() - start_time
        print(f"[ArchGraphBuilder] Built graph in {elapsed:.2f}s")
        print(f"[ArchGraphBuilder] Stats: {graph.get_statistics()}")

        return graph

    def _find_python_files(self, root_path: Path, max_files: int) -> List[Path]:
        """Find all Python files in directory"""
        files = []

        for pattern in ['**/*.py']:
            for file_path in root_path.glob(pattern):
                if len(files) >= max_files:
                    break

                # Check exclusion patterns
                if any(file_path.match(pat) for pat in self.exclude_patterns):
                    continue

                files.append(file_path)

        return files[:max_files]

    def _get_module_name(self, file_path: Path, root_path: Path, package_name: str) -> str:
        """Convert file path to module name"""
        try:
            relative = file_path.relative_to(root_path)
        except ValueError:
            # File is not relative to root
            return file_path.stem

        # Convert path to module name
        parts = list(relative.parts[:-1]) + [relative.stem]
        if parts[-1] == '__init__':
            parts = parts[:-1]

        # Add package name if it's a submodule
        if parts and parts[0] != package_name:
            parts = [package_name] + parts

        return '.'.join(parts)

    def _parse_module(self, file_path: Path, module_name: str) -> Optional[Module]:
        """Parse a Python file to extract module information"""
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content, filename=str(file_path))

            # Extract classes
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

            # Extract top-level functions
            functions = [
                node.name for node in tree.body
                if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef)
            ]

            # Extract imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            # Calculate LOC
            loc = len([line for line in content.split('\n') if line.strip() and not line.strip().startswith('#')])

            # Calculate content hash
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

            # Get last modified time
            last_modified = file_path.stat().st_mtime

            return Module(
                path=file_path,
                name=module_name,
                classes=classes,
                functions=functions,
                imports=imports,
                loc=loc,
                complexity=self._estimate_complexity(tree),
                last_modified=last_modified,
                content_hash=content_hash
            )

        except Exception as e:
            print(f"[ArchGraphBuilder] Failed to parse {file_path}: {e}")
            return None

    def _estimate_complexity(self, tree: ast.AST) -> float:
        """Estimate cyclomatic complexity from AST"""
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            # Decision points increase complexity
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        return float(complexity)

    def _extract_dependencies(
        self,
        file_path: Path,
        module_name: str,
        all_modules: Dict[str, Module]
    ) -> List[Dependency]:
        """Extract dependencies from a module"""
        dependencies = []

        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content, filename=str(file_path))

            # Track imported names
            module_names = set(all_modules.keys())

            # Extract import dependencies
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        target = alias.name.split('.')[0]
                        # Check if it's an internal module
                        for mod_name in module_names:
                            if mod_name.startswith(target):
                                dependencies.append(Dependency(
                                    source=module_name,
                                    target=mod_name,
                                    dep_type=DependencyType.IMPORT,
                                    line_number=node.lineno,
                                    strength=0.5
                                ))
                                break

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        target = node.module.split('.')[0]
                        for mod_name in module_names:
                            if mod_name.startswith(target) or target in mod_name:
                                dependencies.append(Dependency(
                                    source=module_name,
                                    target=mod_name,
                                    dep_type=DependencyType.IMPORT,
                                    line_number=node.lineno,
                                    strength=1.0
                                ))
                                break

        except Exception as e:
            print(f"[ArchGraphBuilder] Failed to extract dependencies from {file_path}: {e}")

        return dependencies


# Example usage
if __name__ == "__main__":
    # Test on current repository
    builder = ArchitecturalGraphBuilder()
    graph = builder.build_from_directory(
        Path("/Users/ed/Nerion-V2/nerion_digital_physicist"),
        package_name="nerion_digital_physicist"
    )

    print("\n=== Architecture Statistics ===")
    stats = graph.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n=== Circular Dependencies ===")
    cycles = graph.find_circular_dependencies()
    if cycles:
        for i, cycle in enumerate(cycles[:5], 1):
            print(f"{i}. {' -> '.join(cycle)}")
    else:
        print("No circular dependencies found!")

    print("\n=== Search: 'training' ===")
    results = graph.search_by_functionality("training")
    for module, score in results[:5]:
        print(f"  {module} (score: {score:.1f})")
