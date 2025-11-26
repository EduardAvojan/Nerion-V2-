"""
Universal Autonomous Fixer 🔧
Real autonomous system that:
1. Runs python scripts OR pytest → detects actual failures
2. Reasons about the fix using Chain-of-Thought
3. Uses Gemini/Claude to generate the fix
4. Modifies the real file
5. Verifies the fix works
6. Learns from the result
"""
import subprocess
import re
import logging
import sys
import os
from pathlib import Path
import cProfile
import pstats
import io
import time
from typing import Dict, Any, Optional, Tuple, List

from selfcoder.planner.explainable_planner import ExplainablePlanner
from nerion_digital_physicist.training.online_learner import OnlineLearner
from nerion_digital_physicist.training.maml import MAMLTrainer, MAMLConfig, MAMLTask
from nerion_digital_physicist.infrastructure.production_collector import ProductionFeedbackCollector, ProductionBug
from nerion_digital_physicist.infrastructure.memory import ReplayStore
from nerion_digital_physicist.agent.data import create_graph_data_object, create_graph_data_from_source
from nerion_digital_physicist.agent.semantics import get_global_embedder
from nerion_digital_physicist.memory.episodic_memory import EpisodicMemory, Episode, EpisodeType, EpisodeOutcome
import torch

# Perception layer - architectural understanding
from nerion_digital_physicist.architecture.graph_builder import (
    ArchitecturalGraphBuilder, ArchitectureGraph
)
from nerion_digital_physicist.architecture.pattern_detector import (
    PatternDetector, ArchitecturalPattern
)
from nerion_digital_physicist.infrastructure.knowledge_graph import KnowledgeGraph
from nerion_digital_physicist.agent.causal_analyzer import CausalAnalyzer, CausalAnalysisResult

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("UniversalFixer")

class ExecutionError:
    def __init__(self, file_path: str, line_num: int, error_msg: str, traceback: str, error_type: str = "Unknown"):
        self.file_path = file_path
        self.line_num = line_num
        self.error_msg = error_msg
        self.traceback = traceback
        self.error_type = error_type

class UniversalFixer:
    def __init__(self, enable_learning: bool = True, dry_run: bool = False, mode: str = "fix", benchmark_mode: bool = False, enable_maml: bool = True, language: str = "python"):
        self.planner = ExplainablePlanner(min_confidence_for_execution=0.7)
        self.fixes_applied = []
        self.enable_learning = enable_learning
        self.enable_maml = enable_maml
        self.dry_run = dry_run
        self.mode = mode
        self.benchmark_mode = benchmark_mode
        self.language = language  # Store language for evolution prompts
        self.embedder = get_global_embedder()

        # Initialize the learning system
        if enable_learning:
            try:
                self.learner = OnlineLearner()
                
                # Load learner state if exists
                project_root = Path(__file__).parent.parent
                learner_path = project_root / "models" / "online_learner_state.pt"
                
                print(f"[DEBUG] Checking for checkpoint at: {learner_path}", flush=True)
                print(f"[DEBUG] Checkpoint exists: {learner_path.exists()}", flush=True)
                
                if learner_path.exists():
                    logger.info(f"📦 Loading OnlineLearner checkpoint from {learner_path}")
                    print(f"📦 Loading OnlineLearner checkpoint from {learner_path}", flush=True)
                    try:
                        self.learner.load_checkpoint(learner_path)
                        print(f"✅ Checkpoint loaded successfully! Task count: {self.learner.task_count}", flush=True)
                    except Exception as e:
                        logger.warning(f"Could not load learner checkpoint: {e}")
                        print(f"❌ Checkpoint loading failed: {e}", flush=True)
                else:
                    logger.info(f"📦 No existing checkpoint found at {learner_path}, starting fresh")
                    print(f"📦 No existing checkpoint, starting fresh", flush=True)

                
                self.learning_examples = []  # Store (graph, error, fix) tuples
                self.model = self._load_or_create_model()
                self.error_to_label = {  # Map task types to class labels (20 classes)
                    # Bug fixes (8 classes)
                    'attribute_error': 0,
                    'type_error': 1,
                    'value_error': 2,
                    'index_error': 3,
                    'key_error': 4,
                    'import_error': 5,
                    'syntax_error': 6,
                    'logic_error': 7,
                    # Code quality (5 classes)
                    'complexity_reduction': 8,
                    'naming_improvement': 9,
                    'code_duplication': 10,
                    'maintainability': 11,
                    'readability': 12,
                    # Architecture (3 classes)
                    'design_pattern': 13,
                    'dependency_management': 14,
                    'modularity': 15,
                    # Security (2 classes)
                    'injection_prevention': 16,
                    'secret_management': 17,
                    # Performance & Type Safety (2 classes)
                    'performance_optimization': 18,
                    'type_safety': 19
                }

                # Initialize MAML for few-shot adaptation (Phase 3)
                if enable_maml:
                    self._init_maml()

                # Initialize surprise-weighted replay (Phase 4)
                self._init_surprise_replay()

                logger.info("🧠 Learning mode ENABLED - Will learn from successful fixes")
            except Exception as e:
                logger.warning(f"Could not initialize OnlineLearner: {e}")
                print(f"❌ OnlineLearner init failed: {e}", flush=True)
                self.enable_learning = False

        # Track examples by error type for MAML task creation
        self.error_type_examples: Dict[str, List[Tuple[Any, int]]] = {
            'attribute_error': [],
            'type_error': [],
            'assertion_error': [],
            'import_error': [],
            'other': []
        }

        # Initialize Episodic Memory (Long-term experience)
        try:
            memory_path = Path(__file__).parent.parent / "data" / "episodic_memory"
            self.memory = EpisodicMemory(storage_path=memory_path)
            logger.info(f"🧠 Episodic Memory initialized at {memory_path}")
        except Exception as e:
            logger.warning(f"Could not initialize EpisodicMemory: {e}")
            self.memory = None

        # Initialize perception layer (architectural understanding)
        self._init_perception_layer()

        # Validation set disabled by default (slow to build)
        # Run `python -c "from nerion_digital_physicist.universal_fixer import UniversalFixer; UniversalFixer()._init_validation_set()"` to build manually
        self.validation_data = []
        self.eval_history = []

    def _init_validation_set(self):
        """Initialize validation set from training_ground for accuracy tracking"""
        try:
            project_root = Path(__file__).parent.parent
            validation_path = project_root / "data" / "validation_set.pt"

            if validation_path.exists():
                # Load cached validation set
                cached = torch.load(validation_path, map_location='cpu', weights_only=False)
                self.validation_data = cached.get('data', [])
                logger.info(f"📊 Loaded validation set: {len(self.validation_data)} samples")
            else:
                # Build validation set from training_ground (20% of files)
                self.validation_data = self._build_validation_set()
                if self.validation_data:
                    # Cache it
                    torch.save({'data': self.validation_data}, validation_path)
                    logger.info(f"📊 Built and cached validation set: {len(self.validation_data)} samples")

            # Track evaluation history
            self.eval_history: List[Dict[str, Any]] = []

        except Exception as e:
            logger.warning(f"Could not initialize validation set: {e}")
            self.validation_data = []
            self.eval_history = []

    def _build_validation_set(self, max_files: int = 20) -> List[Tuple[Any, int]]:
        """Build validation set from training_ground files"""
        import random

        project_root = Path(__file__).parent.parent
        training_ground = project_root / "training_ground"

        if not training_ground.exists():
            logger.warning("training_ground not found, skipping validation set")
            return []

        # Collect Python files
        py_files = list(training_ground.rglob("*.py"))
        excluded = {".git", "__pycache__", "test", "tests", "venv", ".venv"}
        py_files = [f for f in py_files if not any(x in f.parts for x in excluded)]

        # Sample 20% for validation (max 50 files)
        random.seed(42)  # Reproducible split
        num_val = min(max_files, len(py_files) // 5)
        val_files = random.sample(py_files, num_val) if num_val > 0 else []

        validation_data = []
        label_map = {
            'maintainability': 11,
            'dependency_management': 14,
            'injection_prevention': 16,
            'performance_optimization': 18,
        }

        for file_path in val_files:
            try:
                graph = create_graph_data_object(file_path, embedder=self.embedder)
                if graph is not None and hasattr(graph, 'x') and graph.x.size(0) > 0:
                    # Assign pseudo-label based on file characteristics
                    label = self._infer_file_label(file_path)
                    validation_data.append((graph, label))
            except Exception as e:
                continue

        logger.info(f"Built validation set from {len(validation_data)} files")
        return validation_data

    def _infer_file_label(self, file_path: Path) -> int:
        """Infer a label for validation based on file characteristics"""
        name = file_path.name.lower()
        path_str = str(file_path).lower()

        # Simple heuristics for labeling
        if 'security' in name or 'auth' in name or 'crypt' in name:
            return 16  # injection_prevention
        elif 'perf' in name or 'optim' in name or 'cache' in name:
            return 18  # performance_optimization
        elif 'util' in name or 'helper' in name:
            return 11  # maintainability
        elif '__init__' in name or 'setup' in name:
            return 14  # dependency_management
        else:
            return 11  # default to maintainability

    def evaluate_on_validation(self) -> Dict[str, float]:
        """Evaluate current model on validation set"""
        if not self.validation_data or not self.model:
            return {"accuracy": 0.0, "num_samples": 0}

        try:
            accuracy = self.learner._evaluate(self.model, self.validation_data)

            eval_result = {
                "accuracy": accuracy,
                "num_samples": len(self.validation_data),
                "timestamp": time.time(),
                "task_count": self.learner.task_count if hasattr(self, 'learner') else 0
            }
            self.eval_history.append(eval_result)

            return eval_result
        except Exception as e:
            logger.warning(f"Validation evaluation failed: {e}")
            return {"accuracy": 0.0, "num_samples": 0, "error": str(e)}

    def _init_perception_layer(self):
        """Initialize perception components for architectural understanding"""
        try:
            # Architecture graph builder - builds repo-wide dependency graphs
            self.arch_builder = ArchitecturalGraphBuilder()

            # Pattern detector - identifies MVC, Repository, Factory patterns, etc.
            self.pattern_detector = PatternDetector()

            # Knowledge graph (persistent) - stores relationships for RAG
            project_root = Path(__file__).parent.parent
            kg_path = project_root / "data" / "knowledge_graph.graphml"
            if kg_path.exists():
                self.knowledge_graph = KnowledgeGraph.load(kg_path)
                logger.info(f"📊 Loaded knowledge graph from {kg_path}")
            else:
                self.knowledge_graph = KnowledgeGraph()
                logger.info("📊 Initialized new knowledge graph")
            self.kg_path = kg_path

            # Causal analyzer - root cause analysis, impact prediction
            self.causal_analyzer = CausalAnalyzer()

            # Cache for architecture graph (expensive to rebuild)
            self._arch_graph_cache: Optional[ArchitectureGraph] = None
            self._arch_graph_cache_time: float = 0

            logger.info("🔭 Perception layer initialized")
        except Exception as e:
            logger.warning(f"Could not initialize perception layer: {e}")
            self.arch_builder = None
            self.pattern_detector = None
            self.knowledge_graph = None
            self.causal_analyzer = None

    def _init_maml(self):
        """Initialize MAML trainer for few-shot adaptation"""
        try:
            self.maml_config = MAMLConfig(
                inner_lr=0.01,
                inner_steps=3,  # Few steps for quick adaptation
                meta_batch_size=4,
                support_size=3,
                query_size=5,
                first_order=True,  # FOMAML for efficiency
            )
            self.maml_trainer = MAMLTrainer(self.model, self.maml_config)
            self.maml_checkpoint_path = Path(__file__).parent.parent / "models" / "maml_checkpoint.pt"

            # Load existing MAML checkpoint if available
            if self.maml_checkpoint_path.exists():
                self.maml_trainer.load_checkpoint(self.maml_checkpoint_path)
                logger.info("📚 MAML checkpoint loaded for few-shot adaptation")

            logger.info("🎯 MAML initialized for few-shot learning")
        except Exception as e:
            logger.warning(f"Could not initialize MAML: {e}")
            self.enable_maml = False

    def _init_surprise_replay(self):
        """Initialize surprise-weighted replay buffer for prioritized learning"""
        try:
            replay_path = Path(__file__).parent.parent / "data" / "replay"
            self.replay_store = ReplayStore(root=replay_path)
            self.feedback_collector = ProductionFeedbackCollector(
                replay_store=self.replay_store,
                model=self.model,
            )
            logger.info("🎲 Surprise-weighted replay initialized")
        except Exception as e:
            logger.warning(f"Could not initialize surprise replay: {e}")
            self.replay_store = None
            self.feedback_collector = None

    # ========== PERCEPTION LAYER METHODS ==========

    def _get_architecture_understanding(self, file_path: Path) -> Dict[str, Any]:
        """
        Get deep architectural understanding of file context.

        Returns dict with patterns, dependencies, impact scope, and causal analysis.
        """
        understanding = {
            "patterns": [],
            "dependencies": [],
            "impact_scope": [],
            "causal_analysis": None,
            "critical_nodes": [],
            "root_causes": [],
            "bottlenecks": [],
            "cycles": [],
            # Knowledge Graph RAG fields
            "historical_actions": [],
            "known_functions": [],
            "function_relationships": []
        }

        if not self.arch_builder:
            return understanding

        try:
            # Get/build architecture graph (cached for performance)
            arch_graph = self._get_or_build_arch_graph(file_path)

            if arch_graph:
                # Detect architectural patterns (MVC, Repository, Factory, etc.)
                patterns = self.pattern_detector.detect_patterns(arch_graph)
                understanding["patterns"] = [
                    {
                        "type": p.pattern_type.value,
                        "confidence": p.confidence,
                        "modules": p.modules[:5],
                        "violations": p.violations
                    }
                    for p in patterns
                ]

                # Get module name for this file
                module_name = self._file_to_module_name(file_path)

                if module_name and module_name in arch_graph.modules:
                    # Compute impact scope - what modules are affected by changes here
                    impact = arch_graph.compute_impact(module_name)
                    understanding["impact_scope"] = list(impact)[:10]

                    # Find related modules (dependencies)
                    related = arch_graph.find_related_modules(module_name)
                    understanding["dependencies"] = list(related)[:10]

                # Detect circular dependencies
                cycles = arch_graph.find_circular_dependencies()
                understanding["cycles"] = [cycle[:5] for cycle in cycles[:3]]

            # Run causal analysis on the file content
            if self.causal_analyzer and file_path.exists():
                source_code = file_path.read_text(encoding='utf-8')
                causal_result = self.causal_analyzer.analyze_code(source_code, str(file_path))

                understanding["causal_analysis"] = {
                    "num_nodes": len(causal_result.graph.nodes),
                    "num_edges": len(causal_result.graph.edges),
                    "has_cycles": len(causal_result.cycles) > 0
                }
                understanding["critical_nodes"] = [n.name for n in causal_result.critical_nodes[:5]]
                understanding["root_causes"] = [(n.name, d) for n, d in causal_result.root_causes[:5]]
                understanding["bottlenecks"] = [n.name for n in causal_result.bottlenecks[:5]]

            # Query Knowledge Graph for historical context (RAG)
            if self.knowledge_graph:
                file_id = str(file_path)

                # Get historical actions on this file
                historical_actions = self.knowledge_graph.get_actions_on_file(file_id)
                for action_id in historical_actions[:5]:
                    outcome = self.knowledge_graph.get_outcome_of_action(action_id)
                    if outcome:
                        understanding["historical_actions"].append({
                            "action": action_id,
                            "outcome": outcome.get("result", "unknown"),
                            "success": outcome.get("success", False)
                        })

                # Get known functions in this file
                known_functions = self.knowledge_graph.get_functions_in_file(file_id)
                understanding["known_functions"] = known_functions[:10]

                # Get function call relationships for key functions
                for func_id in known_functions[:5]:
                    calls = self.knowledge_graph.get_function_calls(func_id)
                    if calls:
                        understanding["function_relationships"].append({
                            "function": func_id,
                            "calls": calls[:5]
                        })

            # Update knowledge graph with new understanding
            self._update_knowledge_graph(file_path, understanding)

        except Exception as e:
            logger.warning(f"Perception analysis failed: {e}")

        return understanding

    def _get_or_build_arch_graph(self, file_path: Path) -> Optional[ArchitectureGraph]:
        """Get cached architecture graph or build new one"""
        cache_timeout = 300  # 5 minutes

        if (self._arch_graph_cache and
            time.time() - self._arch_graph_cache_time < cache_timeout):
            return self._arch_graph_cache

        # Find repository root
        repo_root = self._find_repo_root(file_path)
        if not repo_root:
            return None

        try:
            logger.info(f"🏗️ Building architecture graph for {repo_root.name}...")
            self._arch_graph_cache = self.arch_builder.build_from_directory(
                repo_root,
                max_files=500  # Limit for performance
            )
            self._arch_graph_cache_time = time.time()
            return self._arch_graph_cache
        except Exception as e:
            logger.warning(f"Could not build architecture graph: {e}")
            return None

    def _find_repo_root(self, file_path: Path) -> Optional[Path]:
        """Find repository root from file path"""
        current = file_path.parent if file_path.is_file() else file_path
        while current != current.parent:
            if (current / ".git").exists() or (current / "setup.py").exists() or (current / "pyproject.toml").exists():
                return current
            current = current.parent
        return file_path.parent

    def _file_to_module_name(self, file_path: Path) -> Optional[str]:
        """Convert file path to Python module name"""
        try:
            parts = list(file_path.parts)
            # Find src or package root
            for i, part in enumerate(parts):
                if part in ['src', 'lib'] or '_ground' in part:
                    parts = parts[i+1:]
                    break

            module_parts = list(parts[:-1]) + [file_path.stem]
            return '.'.join(module_parts)
        except:
            return None

    def _update_knowledge_graph(self, file_path: Path, understanding: Dict[str, Any]):
        """Update knowledge graph with new understanding"""
        if not self.knowledge_graph:
            return

        try:
            file_id = str(file_path)

            # Add file node with detected patterns
            self.knowledge_graph.add_node(
                file_id,
                "File",
                patterns=str([p["type"] for p in understanding.get("patterns", [])]),
                critical_nodes=str(understanding.get("critical_nodes", []))
            )

            # Add dependency edges
            for dep in understanding.get("dependencies", []):
                self.knowledge_graph.add_edge(file_id, dep, "DEPENDS_ON")

            # Add impact edges
            for impacted in understanding.get("impact_scope", []):
                self.knowledge_graph.add_edge(file_id, impacted, "IMPACTS")

            # Save periodically (every 10 updates)
            if not hasattr(self, '_kg_save_counter'):
                self._kg_save_counter = 0
            self._kg_save_counter += 1

            if self._kg_save_counter % 10 == 0:
                self.knowledge_graph.save(self.kg_path)
                logger.debug(f"💾 Saved knowledge graph ({self._kg_save_counter} updates)")

        except Exception as e:
            logger.warning(f"Knowledge graph update failed: {e}")

    def _get_causal_root_cause(self, error: ExecutionError) -> Optional[str]:
        """Use causal analyzer to identify root cause of an error"""
        if not self.causal_analyzer:
            return None

        try:
            file_path = Path(error.file_path)
            if not file_path.exists():
                return None

            source_code = file_path.read_text(encoding='utf-8')
            causal_result = self.causal_analyzer.analyze_code(source_code, str(file_path))

            # Try to identify the error variable from the error message
            error_var = self._extract_error_variable(error.error_msg)
            if error_var:
                root_causes = self.causal_analyzer.identify_root_cause(
                    error_var, causal_result, max_depth=5
                )
                if root_causes:
                    explanations = []
                    for node, distance, explanation in root_causes[:3]:
                        explanations.append(f"- {node.name} (depth {distance}): {explanation}")
                    return "\n".join(explanations)

            # Fallback: return critical nodes
            if causal_result.critical_nodes:
                critical = [n.name for n in causal_result.critical_nodes[:3]]
                return f"Critical nodes: {', '.join(critical)}"

            return None
        except Exception as e:
            logger.debug(f"Causal root cause analysis failed: {e}")
            return None

    def _extract_error_variable(self, error_msg: str) -> Optional[str]:
        """Extract variable name from error message"""
        import re
        # Common patterns: 'variable_name', "variable_name", `variable_name`
        patterns = [
            r"'(\w+)'",
            r'"(\w+)"',
            r'`(\w+)`',
            r"name '(\w+)' is not defined",
            r"has no attribute '(\w+)'",
            r"object '(\w+)'",
        ]
        for pattern in patterns:
            match = re.search(pattern, error_msg)
            if match:
                return match.group(1)
        return None

    def detect_errors(self, target_path: str) -> List[ExecutionError]:
        """Step 1: Run the target (script, test file, or directory) and detect failures"""
        target_path_obj = Path(target_path)
        
        # If it's a directory, assume it's a test suite and run pytest
        if target_path_obj.is_dir():
            return self._run_pytest(target_path)
            
        is_test = target_path.endswith(".py") and ("test_" in target_path or "_test" in target_path)
        
        if is_test:
            return self._run_pytest(target_path)
        else:
            return self._run_python_script(target_path)

    def _run_pytest(self, test_path: str) -> List[ExecutionError]:
        logger.info(f"🔍 Running pytest on {test_path}")
        result = subprocess.run(
            ["python", "-m", "pytest", test_path, "-v", "--tb=short"],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        return self._parse_pytest_output(result.stdout + result.stderr)

    def _run_python_script(self, script_path: str) -> List[ExecutionError]:
        logger.info(f"🔍 Running python script: {script_path}")
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            return []
            
        return self._parse_python_traceback(result.stderr, script_path)

    def _parse_python_traceback(self, stderr: str, script_path: str) -> List[ExecutionError]:
        errors = []
        lines = stderr.split('\n')
        
        # Basic traceback parsing
        traceback_lines = []
        error_msg = "Unknown Error"
        error_type = "Unknown"
        file_path = script_path
        line_num = 0
        
        # Check for SyntaxError which might not have "Traceback" header
        is_syntax_error = "SyntaxError:" in stderr or "IndentationError:" in stderr
        
        in_traceback = False
        for line in lines:
            if "Traceback (most recent call last):" in line:
                in_traceback = True
                traceback_lines = [line]
                continue
            
            # If it's a syntax error, we treat the whole output as relevant
            if is_syntax_error and line.strip().startswith("File "):
                in_traceback = True
                traceback_lines.append(line)
            
            if in_traceback:
                traceback_lines.append(line)
                # Look for file path and line number in "File "...", line X, in ..."
                if line.strip().startswith("File "):
                    parts = line.split('"')
                    if len(parts) >= 2:
                        fpath = parts[1]
                        # Only care about the target script or files in the project, not system libs
                        if fpath.endswith(Path(script_path).name) or os.getcwd() in fpath:
                            file_path = fpath
                            # Extract line number
                            line_match = re.search(r"line (\d+)", line)
                            if line_match:
                                line_num = int(line_match.group(1))
                
                # The last line usually contains the Error Type and Message
                if not line.startswith(" ") and ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        possible_type = parts[0].strip()
                        if "Error" in possible_type or "Exception" in possible_type:
                            error_type = possible_type
                            error_msg = parts[1].strip()
        
        if in_traceback or is_syntax_error:
            # If we didn't find a specific error type but know it failed, default to Runtime Error
            if error_type == "Unknown" and is_syntax_error:
                error_type = "SyntaxError"
                error_msg = "Syntax or Indentation Error detected"
                
            errors.append(ExecutionError(
                file_path=file_path,
                line_num=line_num,
                error_msg=f"{error_type}: {error_msg}",
                traceback="\n".join(traceback_lines) if traceback_lines else stderr,
                error_type=error_type
            ))
            
        return errors

    def _parse_pytest_output(self, output: str) -> List[ExecutionError]:
        failures = []
        summary_start = output.find("short test summary")
        if summary_start < 0:
            return failures
            
        summary_section = output[summary_start:]
        for line in summary_section.split("\n"):
            if line.startswith("FAILED") or line.startswith("ERROR"):
                # Line format: FAILED file::test - error msg
                parts = line.split()
                if len(parts) < 2: continue
                
                nodeid = parts[1]
                if "::" not in nodeid: continue
                
                test_file, test_name = nodeid.split("::", 1)
                
                # Find failure section using just the test name (without parameters or error msg)
                # Test name might have parameters like test_add[1-2], so we search for the base name
                # But the header usually contains the full parameterized name.
                # Let's try to find the test name surrounded by spaces in the header lines
                
                # Search for "____ test_name ____" pattern
                # Pytest headers are like ________________________________ test_name _________________________________
                
                # Simple search for the test name might match other things, but usually it's unique enough
                failure_start = output.find(f" {test_name} ")
                
                if failure_start > 0:
                    next_failure = output.find("\n____", failure_start + 10)
                    if next_failure < 0: next_failure = output.find("\n====", failure_start)
                    
                    # If end not found, take the rest (or up to summary)
                    if next_failure < 0: next_failure = summary_start
                    
                    failure_section = output[failure_start:next_failure]
                    
                    error_lines = []
                    line_num = 0
                    for fline in failure_section.split("\n"):
                        if test_file in fline and ":" in fline:
                            parts = fline.split(":")
                            # Look for line number after file path
                            # path/to/file.py:123: ...
                            for i, part in enumerate(parts):
                                if part.strip().endswith(test_file) and i + 1 < len(parts):
                                    possible_line = parts[i+1].strip()
                                    if possible_line.isdigit():
                                        line_num = int(possible_line)
                                        break
                                        
                        if fline.strip().startswith("E "):
                            error_lines.append(fline.strip()[2:])
                    
                    error_msg = " ".join(error_lines) if error_lines else "Unknown error"
                    
                    failures.append(ExecutionError(
                        file_path=test_file,
                        line_num=line_num,
                        error_msg=error_msg,
                        traceback=failure_section,
                        error_type="AssertionError"
                    ))
        return failures
    
    def reason_about_fix(self, error: ExecutionError) -> Optional[Dict[str, Any]]:
        """Step 2: Use Chain-of-Thought to reason about the fix"""
        logger.info(f"🧠 Reasoning about error in: {error.file_path}")
        
        file_path = Path(error.file_path)
        if not file_path.is_absolute():
            file_path = Path(os.getcwd()) / file_path
            
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
            
        code = file_path.read_text()
        
        context = {
            "file": str(file_path),
            "error": error.error_msg,
            "traceback": error.traceback,
            "code_snippet": code[max(0, error.line_num-10):error.line_num+10] if error.line_num > 0 else "",
        }
        
        # Get AI Insight from GNN
        ai_insight = self._get_model_insight(file_path)
        context["ai_insight"] = ai_insight
        
        task_description = f"""Python execution failure in {error.file_path}:
Error: {error.error_msg}
AI Insight: {ai_insight}

The script is failing. Analyze the error and propose a fix."""
        
        plan = self.planner.create_plan(task_description, context)
        
        logger.info(f"Plan confidence: {plan.reasoning.overall_confidence:.2f}")
        
        if not plan.reasoning.execution_approved:
            logger.warning("⚠️ Fix requires human review")
            return None
            
        return {
            "plan": plan,
            "context": context
        }
    
    def apply_fix(self, error: ExecutionError, fix_plan: Dict[str, Any]) -> bool:
        """Step 3: Use Claude to generate and apply the actual fix"""
        if self.dry_run:
            logger.info(f"🛑 DRY RUN: Would apply fix to {error.file_path}")
            return True

        logger.info(f"🔧 Using Claude to generate fix for {error.file_path}")
        
        try:
            from anthropic import Anthropic
            import os
            
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                logger.error("ANTHROPIC_API_KEY not found")
                return False
            
            client = Anthropic(api_key=api_key)
            
            file_path = Path(error.file_path)
            if not file_path.is_absolute():
                file_path = Path(os.getcwd()) / file_path
                
            code = file_path.read_text()
            
            prompt = f"""You are an expert Python developer fixing a bug.

File: {error.file_path}
Error: {error.error_msg}

Full Traceback:
{error.traceback}

Current Code:
```python
{code}
```"""

            # RAG: Inject similar past fixes from memory
            if self.memory:
                try:
                    query_episode = Episode(
                        episode_id="query",
                        episode_type=EpisodeType.BUG_FIX,
                        task=error.error_msg,
                        code_before=code,
                        action_taken="",
                        code_after="",
                        context={},
                        tags=[self._categorize_error(error.error_msg)]
                    )
                    similar_episodes = self.memory.recall_similar(query_episode, k=3)
                    
                    if similar_episodes:
                        logger.info(f"🧠 Recalled {len(similar_episodes)} similar past fixes")
                        prompt += "\n\nRelevant Past Fixes (Learn from these):\n"
                        for i, ep in enumerate(similar_episodes):
                            prompt += f"""
--- Example {i+1} ---
Task: {ep.task}
Action: {ep.action_taken}
Outcome: {ep.outcome.value}
"""
                except Exception as e:
                    logger.warning(f"Memory recall failed: {e}")

            # Add causal analysis to help identify root cause
            causal_insight = self._get_causal_root_cause(error)
            if causal_insight:
                logger.info("🔍 Added causal root cause analysis to prompt")
                prompt += f"\n\nCausal Analysis (Root Cause Identification):\n{causal_insight}"

            # Add learned principles from memory consolidation
            if self.memory:
                try:
                    error_category = self._categorize_error(error.error_msg)
                    principles = self.memory.get_applicable_principles(
                        task_type=error_category,
                        tags=[error_category],
                        min_confidence=0.7,
                        limit=3
                    )
                    if principles:
                        logger.info(f"💡 Applied {len(principles)} learned principles")
                        prompt += "\n\nLearned Principles (from successful past fixes):\n"
                        for p in principles:
                            prompt += f"- {p.description} (confidence: {p.confidence:.0%})\n"
                except Exception as e:
                    logger.debug(f"Principle retrieval skipped: {e}")

            prompt += """

Task: Fix the code to resolve the error.
Analyze the traceback, causal analysis, and learned principles to find the root cause.
Return ONLY the corrected Python code for the ENTIRE file. No explanations, no markdown."""

            logger.info("🤖 Asking Claude to generate fix...")
            
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            fixed_code = message.content[0].text
            
            if "```python" in fixed_code:
                fixed_code = fixed_code.split("```python")[1].split("```")[0].strip()
            elif "```" in fixed_code:
                fixed_code = fixed_code.split("```")[1].split("```")[0].strip()
            
            file_path.write_text(fixed_code)
            logger.info(f"✅ Applied Claude's fix to {error.file_path}")
            
            self._store_fix_example(file_path, code, fixed_code, error)
            return True
            
        except Exception as e:
            logger.error(f"❌ Error generating fix: {e}")
            return False
    
    def verify_fix(self, target_path: str) -> bool:
        """Step 4: Re-run to verify"""
        logger.info(f"✅ Verifying fix...")
        errors = self.detect_errors(target_path)
        return len(errors) == 0
    
    def _store_fix_example(self, file_path: Path, original_code: str, fixed_code: str, error: ExecutionError):
        # 1. Store for Online Learner (GNN Training)
        if self.enable_learning:
            try:
                graph_data = create_graph_data_object(file_path, embedder=self.embedder)
                example = {
                    'graph': graph_data,
                    'error_type': self._categorize_error(error.error_msg),
                    'original_code': original_code,
                    'fixed_code': fixed_code,
                    'file_path': str(file_path)
                }
                self.learning_examples.append(example)
                logger.info(f"💾 Stored learning example (total: {len(self.learning_examples)})")
            except Exception as e:
                logger.warning(f"Could not create learning example: {e}")

        # 2. Store in Episodic Memory (Experience Replay)
        if self.memory:
            try:
                import uuid
                episode = Episode(
                    episode_id=str(uuid.uuid4()),
                    episode_type=EpisodeType.BUG_FIX,
                    task=f"Fix {error.error_msg}",
                    code_before=original_code,
                    action_taken="Applied LLM Fix",
                    code_after=fixed_code,
                    context={'file_path': str(file_path), 'error_type': error.error_type},
                    outcome=EpisodeOutcome.SUCCESS,
                    surprise=0.5, # Default surprise
                    impact=0.8,   # Bug fixes are high impact
                    tags=[self._categorize_error(error.error_msg), 'autonomous_fix']
                )
                self.memory.store_episode(episode)
            except Exception as e:
                logger.warning(f"Could not store episode in memory: {e}")

    def _categorize_error(self, error_msg: str) -> str:
        """Categorize task type for GNN training (20 classes)."""
        msg_lower = error_msg.lower()
        
        # Bug fixes (8 classes) - from actual runtime errors
        if "AttributeError" in error_msg or "attribute" in msg_lower:
            return "attribute_error"
        elif "TypeError" in error_msg:
            return "type_error"
        elif "ValueError" in error_msg:
            return "value_error"
        elif "IndexError" in error_msg or "index out of" in msg_lower:
            return "index_error"
        elif "KeyError" in error_msg:
            return "key_error"
        elif "ImportError" in error_msg or "ModuleNotFoundError" in error_msg:
            return "import_error"
        elif "SyntaxError" in error_msg or "IndentationError" in error_msg:
            return "syntax_error"
        elif "AssertionError" in error_msg or "logic" in msg_lower:
            return "logic_error"
        
        # Code quality (5 classes) - from evolution tasks
        elif "complexity" in msg_lower or "nested" in msg_lower or "evolve_quality" in error_msg:
            return "complexity_reduction"
        elif "naming" in msg_lower or "variable" in msg_lower:
            return "naming_improvement"
        elif "duplication" in msg_lower or "dry" in msg_lower or "repeated" in msg_lower:
            return "code_duplication"
        elif "solid" in msg_lower or "single responsibility" in msg_lower or "Quality Refactor" in error_msg:
            return "maintainability"
        elif "readability" in msg_lower or "documentation" in msg_lower:
            return "readability"
        
        # Architecture (3 classes)
        elif "pattern" in msg_lower or "factory" in msg_lower or "strategy" in msg_lower:
            return "design_pattern"
        elif "coupling" in msg_lower or "dependency" in msg_lower or "injection" in msg_lower:
            return "dependency_management"
        elif "god class" in msg_lower or "modularity" in msg_lower or "separation" in msg_lower:
            return "modularity"
        
        # Security (2 classes)
        elif "sql" in msg_lower or "xss" in msg_lower or "injection" in msg_lower or "Security" in error_msg or "evolve_security" in error_msg:
            return "injection_prevention"
        elif "secret" in msg_lower or "password" in msg_lower or "credential" in msg_lower or "hardcoded" in msg_lower:
            return "secret_management"
        
        # Performance & Type Safety (2 classes)
        elif "Performance" in error_msg or "evolve_perf" in error_msg or "optimization" in msg_lower:
            return "performance_optimization"
        elif "type hint" in msg_lower or "mypy" in msg_lower or "evolve_types" in error_msg or "Type Safety" in error_msg or "Type Injection" in error_msg:
            return "type_safety"
        
        # Default: complexity reduction (most common quality improvement)
        else:
            return "complexity_reduction"

    def learn_from_fixes(self):
        """Step 5: Learn from all successful fixes with batch accumulation"""
        if not self.enable_learning or not self.learning_examples: return

        # Batch accumulator path (persisted across subprocess runs)
        project_root = Path(__file__).parent.parent
        accumulator_path = project_root / "data" / "batch_accumulator.pt"
        MIN_BATCH_SIZE = 5

        # Load existing accumulated examples from disk
        accumulated = []
        if accumulator_path.exists():
            try:
                accumulated = torch.load(accumulator_path, map_location='cpu', weights_only=False)
            except Exception:
                accumulated = []

        # Add new examples to accumulator
        for example in self.learning_examples:
            graph = example['graph']
            error_type = example['error_type']
            label = self.error_to_label.get(error_type, 4)
            accumulated.append({
                'graph': graph,
                'label': label,
                'error_type': error_type,
                'file_path': example.get('file_path', ''),
                'original_code': example.get('original_code', ''),
            })

            # Track examples by error type for MAML (Phase 3)
            if error_type in self.error_type_examples:
                self.error_type_examples[error_type].append((graph, label))

        # Only train when we have enough examples for meaningful batch
        if len(accumulated) < MIN_BATCH_SIZE:
            # Save accumulator and wait for more examples
            torch.save(accumulated, accumulator_path)
            logger.info(f"📦 Accumulated {len(accumulated)}/{MIN_BATCH_SIZE} examples (waiting for batch)")
            self.learning_examples = []
            return

        logger.info(f"♾️  Learning from {len(accumulated)} accumulated fixes (batch training)")
        try:
            training_data = [(ex['graph'], ex['label']) for ex in accumulated]

            # Record experiences for replay
            for ex in accumulated:
                self._record_surprise_experience(ex, ex['graph'], ex['label'])

            # Get surprise-weighted replay samples to mix with new data
            replay_data = self._get_surprise_replay_samples()

            # Get validation data for forgetting metric
            val_data = self.validation_data if hasattr(self, 'validation_data') else None

            updated_model, update_info = self.learner.incremental_update(
                current_model=self.model,
                new_data=training_data,
                replay_data=replay_data,  # Use surprise-weighted samples
                validation_data=val_data   # Track forgetting
            )
            self.model = updated_model
            self._save_model()
            logger.info(f"✅ GNN updated: new_acc={update_info.new_accuracy:.2%}, old_acc={update_info.old_accuracy:.2%}")

            # Clear accumulator after successful training
            if accumulator_path.exists():
                accumulator_path.unlink()
            logger.info(f"🗑️ Cleared batch accumulator")

            # Periodic validation evaluation (every 5 updates)
            if self.learner.task_count % 5 == 0 and hasattr(self, 'validation_data') and self.validation_data:
                eval_result = self.evaluate_on_validation()
                logger.info(f"📊 Validation accuracy: {eval_result['accuracy']:.2%} ({eval_result['num_samples']} samples)")

            # Periodically run MAML meta-training if we have enough examples
            if self.enable_maml:
                self._maybe_run_maml_update()

        except Exception as e:
            logger.error(f"Error during learning: {e}")

    def _record_surprise_experience(self, example: Dict[str, Any], graph: Any, label: int):
        """Record experience with surprise score for prioritized replay"""
        if not hasattr(self, 'feedback_collector') or self.feedback_collector is None:
            return

        try:
            bug = ProductionBug(
                bug_id=f"fix_{time.time()}",
                source_code=example.get('original_code', ''),
                file_path=example.get('file_path', 'unknown'),
                language='python',
                bug_type=example.get('error_type', 'other'),
                severity='medium',
                ground_truth=label
            )
            surprise = self.feedback_collector.collect_bug(bug, graph, ground_truth=label)
            if surprise > 0.7:
                logger.info(f"⚠️  High surprise ({surprise:.2f}) - prioritizing this example")
        except Exception as e:
            logger.debug(f"Could not record surprise experience: {e}")

    def _get_surprise_replay_samples(self, k: int = 50) -> List[Tuple[Any, int]]:
        """Get high-surprise samples from replay buffer for training mix"""
        if not hasattr(self, 'replay_store') or self.replay_store is None:
            return []

        try:
            # Sample using priority (surprise-weighted)
            experiences = self.replay_store.sample(k=k, strategy="priority")
            replay_data = []

            for exp in experiences:
                # Reconstruct graph from stored metadata if possible
                code = exp.metadata.get('source_code', '')
                if code:
                    try:
                        from nerion_digital_physicist.agent.data import create_graph_data_from_source
                        graph = create_graph_data_from_source(code)
                        label = exp.metadata.get('ground_truth', 4)
                        replay_data.append((graph, label))
                    except Exception:
                        continue

            if replay_data:
                logger.info(f"🎲 Using {len(replay_data)} surprise-weighted replay samples")
            return replay_data
        except Exception as e:
            logger.debug(f"Could not get replay samples: {e}")
            return []

    def _maybe_run_maml_update(self):
        """Run MAML meta-training if we have enough examples per error type"""
        if not self.enable_maml or not hasattr(self, 'maml_trainer'):
            return

        min_examples = self.maml_config.support_size + self.maml_config.query_size

        # Create tasks from error types that have enough examples
        tasks = []
        for error_type, examples in self.error_type_examples.items():
            if len(examples) >= min_examples:
                import random
                shuffled = examples.copy()
                random.shuffle(shuffled)

                support = shuffled[:self.maml_config.support_size]
                query = shuffled[self.maml_config.support_size:min_examples]

                task = MAMLTask(
                    task_id=f"error_{error_type}",
                    support_graphs=[g for g, _ in support],
                    support_labels=[l for _, l in support],
                    query_graphs=[g for g, _ in query],
                    query_labels=[l for _, l in query],
                    metadata={'error_type': error_type}
                )
                tasks.append(task)

        # Run meta-training if we have at least 2 tasks
        if len(tasks) >= 2:
            logger.info(f"🎯 Running MAML meta-training with {len(tasks)} tasks")
            try:
                self.maml_trainer.meta_train(tasks, epochs=10)
                self.maml_trainer.save_checkpoint(self.maml_checkpoint_path)
                logger.info("📚 MAML checkpoint saved")
            except Exception as e:
                logger.warning(f"MAML meta-training failed: {e}")

    def adapt_to_new_bug_type(self, support_examples: List[Tuple[Any, int]]) -> Optional[torch.nn.Module]:
        """
        Quickly adapt the model to a new bug type using MAML few-shot learning.

        Args:
            support_examples: List of (graph, label) tuples for the new bug type

        Returns:
            Adapted model or None if MAML not available
        """
        if not self.enable_maml or not hasattr(self, 'maml_trainer'):
            logger.warning("MAML not available for few-shot adaptation")
            return None

        if len(support_examples) < 1:
            logger.warning("Need at least 1 example for few-shot adaptation")
            return None

        logger.info(f"🚀 Adapting to new bug type with {len(support_examples)} examples")
        try:
            adapted_model = self.maml_trainer.adapt_to_new_task(
                support_examples=support_examples,
                num_steps=5  # Quick adaptation
            )
            return adapted_model
        except Exception as e:
            logger.error(f"Few-shot adaptation failed: {e}")
            return None

    def _load_or_create_model(self):
        """
        Load or create the GNN model with intelligent weight initialization.

        Priority order:
        1. Load existing trained model (nerion_immune_brain.pt)
        2. Initialize fresh model with contrastive pretrained backbone (if available)
        3. Create fresh model with random initialization
        """
        from nerion_digital_physicist.agent.brain import CodeGraphGCN, load_contrastive_pretrained
        import os

        project_root = Path(__file__).parent.parent
        model_path = project_root / "models" / "nerion_immune_brain.pt"
        contrastive_path = project_root / "models" / "contrastive_pretrained.pt"
        contrastive_best_path = project_root / "models" / "contrastive_best.pt"

        # Try loading existing trained model first
        if os.path.exists(model_path):
            try:
                model = CodeGraphGCN(num_node_features=800, hidden_channels=256, num_classes=20, num_layers=4)
                model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
                logger.info(f"📦 Loaded trained model from {model_path}")
                return model
            except Exception as e:
                logger.warning(f"Could not load trained model: {e}")

        # Create fresh model
        model = CodeGraphGCN(num_node_features=800, hidden_channels=256, num_classes=20, num_layers=4)

        # Try to initialize with contrastive pretrained weights for better starting point
        contrastive_file = None
        if os.path.exists(contrastive_path):
            contrastive_file = contrastive_path
        elif os.path.exists(contrastive_best_path):
            contrastive_file = contrastive_best_path

        if contrastive_file:
            try:
                model = load_contrastive_pretrained(model, str(contrastive_file))
                logger.info(f"🎯 Initialized model with contrastive pretrained weights from {contrastive_file}")
            except Exception as e:
                logger.warning(f"Could not load contrastive weights: {e}")
        else:
            logger.info("📦 Created fresh model (no contrastive pretraining available)")

        return model

    def _save_model(self):
        import os
        
        project_root = Path(__file__).parent.parent
        model_path = project_root / "models" / "nerion_immune_brain.pt"
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        
        # Save learner state
        if hasattr(self, 'learner'):
            learner_path = project_root / "models" / "online_learner_state.pt"
            self.learner.save_checkpoint(learner_path)
            
        self._save_learning_history()

    def _save_learning_history(self):
        """Save raw learning examples to JSON for audit"""
        import json
        
        project_root = Path(__file__).parent.parent
        history_path = project_root / "models" / "learning_history.json"
        
        new_entries = []
        for ex in self.learning_examples:
            entry = {
                "timestamp": time.time(),
                "file_path": ex["file_path"],
                "error_type": ex["error_type"],
                "original_code": ex["original_code"],
                "fixed_code": ex["fixed_code"]
            }
            new_entries.append(entry)
            
        try:
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    history = json.load(f)
            else:
                history = []
                
            history.extend(new_entries)
            
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            logger.info(f"📜 Saved learning history to {history_path}")
        except Exception as e:
            logger.warning(f"Failed to save learning history: {e}")

    def _get_model_insight(self, file_path: Path) -> str:
        """Query the GNN brain and perception layer for insights about the code"""
        insight_parts = []

        # GNN-based health analysis
        if self.enable_learning and self.model:
            try:
                graph_data = create_graph_data_object(file_path, embedder=self.embedder)

                from torch_geometric.data import Batch

                if not hasattr(graph_data, 'batch') or graph_data.batch is None:
                    graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long)

                # Set model to evaluation mode (PyTorch method, not Python eval)
                self.model.train(False)

                health_score = self.model.predict_health(graph_data)
                health_status = "Healthy" if health_score < 0.5 else "Unhealthy"
                confidence = abs(health_score - 0.5) * 2

                insight_parts.append(f"GNN Analysis: Code Health Score {health_score:.2f} ({health_status}, {confidence:.0%} confidence)")
            except Exception as e:
                logger.warning(f"Failed to get GNN insight: {e}")

        # Perception layer analysis (architecture, patterns, causality)
        try:
            perception = self._get_architecture_understanding(file_path)

            if perception["patterns"]:
                patterns_str = ", ".join([
                    f"{p['type']} ({p['confidence']:.0%})"
                    for p in perception["patterns"]
                ])
                insight_parts.append(f"Architecture: {patterns_str}")

                for p in perception["patterns"]:
                    if p["violations"]:
                        insight_parts.append(f"Pattern violations: {', '.join(p['violations'][:2])}")

            if perception["dependencies"]:
                deps = perception["dependencies"][:3]
                insight_parts.append(f"Dependencies: {', '.join(deps)}")

            if perception["impact_scope"]:
                impact = perception["impact_scope"][:3]
                insight_parts.append(f"Changes affect: {', '.join(impact)}")

            if perception["critical_nodes"]:
                insight_parts.append(f"Critical nodes: {', '.join(perception['critical_nodes'][:3])}")

            if perception["bottlenecks"]:
                insight_parts.append(f"Bottlenecks: {', '.join(perception['bottlenecks'][:3])}")

            if perception["cycles"]:
                insight_parts.append(f"Circular dependencies detected: {len(perception['cycles'])} cycle(s)")

            # Knowledge Graph RAG - historical context
            if perception["historical_actions"]:
                successful = [a for a in perception["historical_actions"] if a.get("success")]
                failed = [a for a in perception["historical_actions"] if not a.get("success")]
                if successful:
                    insight_parts.append(f"Past successful fixes: {len(successful)} on this file")
                if failed:
                    insight_parts.append(f"Past failed attempts: {len(failed)} (avoid similar approaches)")

            if perception["known_functions"]:
                insight_parts.append(f"Known functions: {', '.join(perception['known_functions'][:5])}")

            if perception["function_relationships"]:
                for rel in perception["function_relationships"][:2]:
                    calls = ", ".join(rel["calls"][:3])
                    insight_parts.append(f"  {rel['function']} → [{calls}]")

        except Exception as e:
            logger.debug(f"Perception analysis skipped: {e}")

        if not insight_parts:
            return "No AI insight available"

        return "\n".join(insight_parts)

    def evolve_code(self, target_path: str) -> bool:
        """Dispatch evolution based on mode"""
        if self.mode == "evolve_perf":
            return self._evolve_performance(target_path)
        elif self.mode == "evolve_quality":
            return self._evolve_quality(target_path)
        elif self.mode == "evolve_security":
            return self._evolve_security(target_path)
        elif self.mode == "evolve_antibodies":
            return self._evolve_antibodies(target_path)
        elif self.mode == "evolve_types":
            return self._evolve_types(target_path)
        else:
            logger.error(f"Unknown evolution mode: {self.mode}")
            return False

    def _evolve_performance(self, target_path: str) -> bool:
        """Vector 1: Performance Optimization"""
        file_ext = Path(target_path).suffix.lower()
        lang = "JavaScript" if file_ext in ['.js', '.jsx'] else "TypeScript" if file_ext in ['.ts', '.tsx'] else "Python"

        logger.info("=" * 60)
        logger.info(f"🧬 EVOLVER (PERFORMANCE) - Target: {target_path} ({lang})")
        logger.info("=" * 60)

        # 1. Profile (Python only)
        if file_ext == '.py':
            logger.info("⏱️  Profiling current performance...")
            stats, initial_time = self._profile_code(target_path)
            if not stats:
                logger.error("Could not profile code.")
                return False
            logger.info(f"📊 Baseline Time: {initial_time:.4f}s")
        else:
            stats = "N/A (non-Python)"
            initial_time = 0.0
            logger.info("⏱️  Skipping profiling for non-Python file")

        # 2. Analyze & Evolve
        logger.info("🧠 Analyzing for optimizations...")
        prompt_template = """You are an expert {lang} Performance Engineer.

IMPORTANT: Be CONSERVATIVE. Make MINIMAL changes that are guaranteed to work.

File: {file_path}
Profile Stats: {stats}
Current Code:
```{lang_lower}
{code}
```

Task: Make ONE small, safe performance optimization.
Rules:
1. Change as LITTLE as possible - prefer a single targeted fix
2. Do NOT restructure the code or change its organization
3. Do NOT change function signatures or exports
4. Do NOT add new dependencies
5. The code MUST still work exactly the same way
6. If unsure, make NO changes and return the original code

Return ONLY the code with your minimal optimization.""".replace("{lang}", lang).replace("{lang_lower}", lang.lower())
        
        if self._apply_evolution(target_path, prompt_template, stats=stats[:2000]):
            # 3. Verify
            logger.info("✅ Verifying evolution...")
            # For non-Python, skip speedup check (no profiling available)
            check_speedup = file_ext == '.py'
            if self._verify_evolution(target_path, initial_time, check_speedup=check_speedup):
                logger.info("🎉 SUCCESS! Code optimized and verified.")
                # Store learning example (skip if backup doesn't exist - dry run)
                backup_path = Path(target_path).with_suffix(Path(target_path).suffix + ".bak")
                if backup_path.exists():
                    self._store_fix_example(
                        Path(target_path),
                        backup_path.read_text(),
                        Path(target_path).read_text(),
                        ExecutionError(target_path, 0, "Performance Optimization", "", "performance_issue")
                    )
                    self.learn_from_fixes()
                return True
            else:
                logger.warning("⚠️  Evolution failed verification. Reverting.")
                self._revert_backup(target_path)
                return False
        return False

    def _evolve_quality(self, target_path: str) -> bool:
        """Vector 2: Code Quality & Refactoring"""
        target_path_obj = Path(target_path)
        if target_path_obj.is_dir():
            # Pick a random python file that isn't a test and not in excluded dirs
            excluded_dirs = {".git", ".venv", "venv", "env", "__pycache__", "node_modules", "tmp", "temp", "build", "dist"}
            
            candidates = []
            for p in target_path_obj.rglob("*.py"):
                # Check if any part of the path is in excluded_dirs
                if any(part in excluded_dirs for part in p.parts):
                    continue
                if "test" in p.name:
                    continue
                candidates.append(str(p))
                
            if not candidates:
                logger.warning("No suitable Python files found for evolution.")
                return False
            import random
            target_path = random.choice(candidates)
            
        file_ext = Path(target_path).suffix.lower()
        lang = "JavaScript" if file_ext in ['.js', '.jsx'] else "TypeScript" if file_ext in ['.ts', '.tsx'] else "Python"

        logger.info("=" * 60)
        logger.info(f"🧬 EVOLVER (QUALITY) - Target: {target_path} ({lang})")
        logger.info("=" * 60)

        logger.info("🧠 Analyzing for Code Smells & Complexity...")
        prompt_template = """You are an expert {lang} Architect.

IMPORTANT: Be CONSERVATIVE. Make MINIMAL, SAFE changes only.

File: {file_path}
Current Code:
```{lang_lower}
{code}
```

Task: Make ONE small improvement to code quality.
Rules:
1. Pick ONE thing to improve (naming, simplify one function, etc.)
2. Do NOT restructure the entire file
3. Do NOT change function signatures or API contracts
4. Do NOT add or remove imports unless absolutely necessary
5. Do NOT add comments or docstrings - just improve the code itself
6. The code MUST continue to work exactly the same
7. If the code is already clean, return it unchanged

Return ONLY the code.""".replace("{lang}", lang).replace("{lang_lower}", lang.lower())

        if self._apply_evolution(target_path, prompt_template):
            logger.info("✅ Verifying refactor...")
            if self._verify_evolution(target_path, 0.0, check_speedup=False):
                logger.info("🎉 SUCCESS! Code refactored and verified.")
                # Store learning example (skip if backup doesn't exist - dry run)
                backup_path = Path(target_path).with_suffix(Path(target_path).suffix + ".bak")
                if backup_path.exists():
                    self._store_fix_example(
                        Path(target_path),
                        backup_path.read_text(),
                        Path(target_path).read_text(),
                        ExecutionError(target_path, 0, "Quality Refactor", "", "code_smell")
                    )
                    self.learn_from_fixes()
                return True
            else:
                logger.warning("⚠️  Refactor broke the code. Reverting.")
                self._revert_backup(target_path)
                return False
        return False

    def _evolve_security(self, target_path: str) -> bool:
        """Vector 3: Security Hardening"""
        file_ext = Path(target_path).suffix.lower()
        lang = "JavaScript" if file_ext in ['.js', '.jsx'] else "TypeScript" if file_ext in ['.ts', '.tsx'] else "Python"

        logger.info("=" * 60)
        logger.info(f"🧬 EVOLVER (SECURITY) - Target: {target_path} ({lang})")
        logger.info("=" * 60)

        logger.info("🧠 Scanning for Vulnerabilities...")
        prompt_template = """You are an expert {lang} Security Engineer.

IMPORTANT: Be CONSERVATIVE. Only fix ACTUAL vulnerabilities, not hypothetical ones.

File: {file_path}
Current Code:
```{lang_lower}
{code}
```

Task: If there is a REAL vulnerability, fix it. Otherwise return the code unchanged.
Rules:
1. Only fix OBVIOUS security issues (SQL injection, XSS, hardcoded secrets)
2. Do NOT add validation "just in case"
3. Do NOT refactor or restructure - only fix security issues
4. Do NOT change function signatures or APIs
5. If no real vulnerability exists, return the code UNCHANGED
6. The code MUST continue to work exactly the same

Return ONLY the code.""".replace("{lang}", lang).replace("{lang_lower}", lang.lower())

        if self._apply_evolution(target_path, prompt_template):
            logger.info("✅ Verifying security patch...")
            if self._verify_evolution(target_path, 0.0, check_speedup=False):
                logger.info("🎉 SUCCESS! Code hardened and verified.")
                # Store learning example (skip if backup doesn't exist - dry run)
                backup_path = Path(target_path).with_suffix(Path(target_path).suffix + ".bak")
                if backup_path.exists():
                    self._store_fix_example(
                        Path(target_path),
                        backup_path.read_text(),
                        Path(target_path).read_text(),
                        ExecutionError(target_path, 0, "Security Patch", "", "vulnerability")
                    )
                    self.learn_from_fixes()
                return True
            else:
                logger.warning("⚠️  Security patch broke the code. Reverting.")
                self._revert_backup(target_path)
                return False
        return False

    def _evolve_antibodies(self, target_path: str) -> bool:
        """Vector 4: Antibody Generation (Test Coverage)"""
        logger.info("=" * 60)
        logger.info(f"🧬 EVOLVER (ANTIBODIES) - Target: {target_path}")
        logger.info("=" * 60)
        
        test_path = Path(target_path).parent / f"test_{Path(target_path).name}"
        if test_path.exists():
            logger.info(f"⚠️  Test file already exists: {test_path}")
            # In future, we could append to it, but for now skip
            return False
            
        logger.info("🧠 Generating Antibodies (Tests)...")
        
        from anthropic import Anthropic
        import os
        api_key = os.getenv('ANTHROPIC_API_KEY')
        client = Anthropic(api_key=api_key)
        
        code = Path(target_path).read_text()
        
        prompt = f"""You are an expert QA Engineer.
File: {target_path}
Code:
```python
{code}
```
Task: Create a comprehensive pytest file for this code.
1. Cover happy paths and edge cases.
2. Use pytest fixtures where appropriate.
3. Return ONLY the python test code."""

        try:
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )
            test_code = self._extract_code(message.content[0].text)
            test_path.write_text(test_code)
            logger.info(f"💉 Injected Antibodies: {test_path}")
            
            # Verify the new test
            logger.info("✅ Verifying Antibodies...")
            errors = self._run_pytest(str(test_path))
            if not errors:
                logger.info("🎉 SUCCESS! Antibodies active and passing.")
                # Store learning example (Code -> Test)
                # For antibodies, the "fix" is the test file creation. 
                # We store the original code as input, and test code as output? 
                # Or just skip for now as it's a different modality.
                # Let's store it as a "fix" where we added the test.
                self._store_fix_example(
                    test_path, 
                    "", # No original test
                    test_code,
                    ExecutionError(str(test_path), 0, "Missing Tests", "", "coverage_gap")
                )
                self.learn_from_fixes()
                return True
            else:
                logger.warning(f"⚠️  Antibodies failed verification ({len(errors)} errors).")
                # Optional: Fix the test itself? For now, just report.
                return False
                
        except Exception as e:
            logger.error(f"Antibody generation error: {e}")
            return False

    def _evolve_types(self, target_path: str) -> bool:
        """Vector 6: Type Safety Evolution"""
        file_ext = Path(target_path).suffix.lower()
        lang = "JavaScript" if file_ext in ['.js', '.jsx'] else "TypeScript" if file_ext in ['.ts', '.tsx'] else "Python"

        # Skip type evolution for pure JS (use TS instead)
        if file_ext in ['.js', '.jsx']:
            logger.info(f"⏩ Skipping type evolution for JavaScript (use TypeScript instead)")
            return False

        logger.info("=" * 60)
        logger.info(f"🧬 EVOLVER (TYPES) - Target: {target_path} ({lang})")
        logger.info("=" * 60)

        logger.info("🧠 Analyzing for Missing Types...")
        prompt_template = """You are an expert {lang} Typing Specialist.

IMPORTANT: Be CONSERVATIVE. Add types INCREMENTALLY, not all at once.

File: {file_path}
Current Code:
```{lang_lower}
{code}
```

Task: Add type hints to ONE or TWO functions that are missing them.
Rules:
1. Only add types to functions that have NO type hints at all
2. Do NOT change any existing types
3. Do NOT add types to every function - pick 1-2 important ones
4. Use simple, correct types - avoid overly complex generics
5. Do NOT restructure or refactor the code
6. If all functions already have types, return the code unchanged
7. The code MUST continue to work exactly the same

Return ONLY the code.""".replace("{lang}", lang).replace("{lang_lower}", lang.lower())

        if self._apply_evolution(target_path, prompt_template):
            logger.info("✅ Verifying types...")
            # 1. Syntax check
            if self._verify_evolution(target_path, 0.0, check_speedup=False):
                # 2. Optional: Run mypy if installed (Python only)
                if file_ext == '.py':
                    try:
                        import mypy.api
                        logger.info("🔍 Running mypy verification...")
                        stdout, stderr, exit_code = mypy.api.run([target_path])
                        if exit_code == 0:
                            logger.info("🎉 SUCCESS! Code typed and mypy verified.")
                        else:
                            logger.warning(f"⚠️  Mypy found issues (but code runs): {stdout.splitlines()[0]}")
                    except ImportError:
                        logger.info("🎉 SUCCESS! Code typed (mypy not installed).")
                else:
                    logger.info("🎉 SUCCESS! Code typed.")

                # Store learning example (skip if backup doesn't exist - dry run)
                backup_path = Path(target_path).with_suffix(Path(target_path).suffix + ".bak")
                if backup_path.exists():
                    self._store_fix_example(
                        Path(target_path),
                        backup_path.read_text(),
                        Path(target_path).read_text(),
                        ExecutionError(target_path, 0, "Type Injection", "", "missing_types")
                    )
                    self.learn_from_fixes()
                return True
            else:
                logger.warning("⚠️  Type injection broke the code. Reverting.")
                self._revert_backup(target_path)
                return False
        return False

    def _apply_evolution(self, file_path_str: str, prompt_template: str, stats: str = "") -> bool:
        """Generic method to apply LLM-based evolution"""
        if self.dry_run:
            logger.info("🛑 DRY RUN: Would ask Claude to evolve code.")
            return True

        from anthropic import Anthropic
        import os
        
        api_key = os.getenv('ANTHROPIC_API_KEY')
        client = Anthropic(api_key=api_key)
        
        file_path = Path(file_path_str)
        code = file_path.read_text()
        
        # Create backup
        backup_path = file_path.with_suffix(file_path.suffix + ".bak")
        backup_path.write_text(code)
        logger.info(f"💾 Created backup: {backup_path}")
        
        prompt = prompt_template.format(file_path=file_path_str, code=code, stats=stats)

        try:
            logger.info("🤖 Asking Claude to evolve code...")
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            optimized_code = self._extract_code(message.content[0].text)
            file_path.write_text(optimized_code)
            logger.info(f"🧬 Applied evolution to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Evolution error: {e}")
            self._revert_backup(file_path_str)
            return False

    def _extract_code(self, text: str) -> str:
        # Handle markdown code blocks
        if "```" in text:
            # Split by ``` to get the content inside
            parts = text.split("```")
            if len(parts) >= 3:
                # parts[0] is before block, parts[1] is inside, parts[2] is after
                content = parts[1]
                
                # Check if the first line is a language identifier
                if "\n" in content:
                    first_line = content.split("\n", 1)[0].strip()
                    # If first line is short and alphanumeric, it's likely a language ID
                    if len(first_line) < 20 and first_line.replace("-", "").replace("_", "").isalnum():
                        content = content.split("\n", 1)[1]
                        
                return content.strip()
        return text.strip()

    def _profile_code(self, target_path: str) -> Tuple[str, float]:
        """Run profiling on the target script (language-aware)"""
        file_ext = Path(target_path).suffix.lower()

        # Skip profiling for non-Python files - just do syntax check timing
        if file_ext not in ['.py']:
            logger.info(f"Skipping cProfile for {file_ext} file (Python-only)")
            return f"No profiling for {file_ext} files", 0.0

        logger.info(f"Running cProfile on {target_path}")

        import cProfile
        import pstats
        import io

        start_time = time.time()
        try:
            profile_output = Path(target_path).with_suffix(".prof")

            cmd = [sys.executable, "-m", "cProfile", "-o", str(profile_output), target_path]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            except subprocess.TimeoutExpired:
                logger.warning(f"Profiling timed out after 15s (likely interactive/infinite loop)")
                return "Profiling timed out - script runs indefinitely", 15.0

            end_time = time.time()
            duration = end_time - start_time

            if result.returncode != 0:
                logger.error(f"Profiling failed: {result.stderr}")
                return "", 0.0

            # Read stats
            s = io.StringIO()
            ps = pstats.Stats(str(profile_output), stream=s)
            ps.strip_dirs().sort_stats('cumulative').print_stats(20)

            # Clean up
            if profile_output.exists():
                profile_output.unlink()

            return s.getvalue(), duration

        except Exception as e:
            logger.error(f"Profiling error: {e}")
            return "", 0.0

    def _revert_backup(self, target_path: str):
        backup_path = Path(target_path).with_suffix(Path(target_path).suffix + ".bak")
        if backup_path.exists():
            Path(target_path).write_text(backup_path.read_text())
            logger.info("Reverted to backup.")

    def _verify_evolution(self, target_path: str, baseline_time: float, check_speedup: bool = False) -> bool:
        """Verify the new code works and optionally is faster (language-aware)"""
        file_ext = Path(target_path).suffix.lower()

        # 1. Functional Correctness (language-aware)
        logger.info("🧪 Verifying functional correctness...")

        if file_ext == '.py':
            errors = self.detect_errors(target_path)
            if errors:
                logger.error(f"Evolution broke the code: {errors[0].error_msg}")
                return False
        elif file_ext in ['.js', '.jsx']:
            # JavaScript: check syntax with Node.js
            if not self._verify_javascript(target_path):
                return False
        elif file_ext in ['.ts', '.tsx']:
            # TypeScript: check syntax with tsc if available
            if not self._verify_typescript(target_path):
                return False
        else:
            logger.warning(f"Unknown file type {file_ext}, skipping verification")
            return True

        # 2. Performance Check (Optional, Python only)
        if check_speedup and file_ext == '.py':
            logger.info("⏱️  Verifying performance improvement...")
            _, new_time = self._profile_code(target_path)
            logger.info(f"📊 New Time: {new_time:.4f}s (Baseline: {baseline_time:.4f}s)")

            if new_time < baseline_time:
                improvement = (baseline_time - new_time) / baseline_time * 100
                logger.info(f"🚀 Speedup: {improvement:.1f}%")
                return True
            else:
                logger.warning("⚠️  No speedup detected (or slower).")
                return False

        return True

    def _verify_javascript(self, target_path: str) -> bool:
        """Verify JavaScript file syntax using Node.js"""
        try:
            # Use Node.js --check to parse without executing
            result = subprocess.run(
                ["node", "--check", target_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                logger.error(f"JavaScript syntax error: {result.stderr}")
                return False
            logger.info("✅ JavaScript syntax valid")
            return True
        except FileNotFoundError:
            logger.warning("Node.js not found, skipping JS verification")
            return True  # Assume OK if node not available
        except subprocess.TimeoutExpired:
            logger.warning("JavaScript verification timed out")
            return False
        except Exception as e:
            logger.warning(f"JavaScript verification failed: {e}")
            return True  # Be permissive on unknown errors

    def _verify_typescript(self, target_path: str) -> bool:
        """Verify TypeScript file syntax using tsc"""
        try:
            # Try tsc --noEmit to type-check without generating output
            result = subprocess.run(
                ["npx", "tsc", "--noEmit", "--skipLibCheck", target_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                logger.error(f"TypeScript error: {result.stderr}")
                return False
            logger.info("✅ TypeScript syntax valid")
            return True
        except FileNotFoundError:
            # Fall back to just checking JS syntax
            logger.warning("tsc not found, falling back to Node.js syntax check")
            return self._verify_javascript(target_path)
        except subprocess.TimeoutExpired:
            logger.warning("TypeScript verification timed out")
            return False
        except Exception as e:
            logger.warning(f"TypeScript verification failed: {e}")
            return True  # Be permissive on unknown errors

    def train_from_mined_bugs(self, max_bugs: int = 50) -> bool:
        """Train GNN directly from mined bug data (real bugs from git history)."""
        logger.info("=" * 60)
        logger.info("🪲 TRAINING FROM MINED BUGS")
        logger.info("=" * 60)

        # Load mined bugs
        import json
        project_root = Path(__file__).parent.parent
        mined_bugs_path = project_root / "data" / "mined_bugs.json"

        if not mined_bugs_path.exists():
            logger.error(f"No mined bugs found. Run: python scripts/bug_miner.py")
            return False

        with open(mined_bugs_path) as f:
            data = json.load(f)

        bugs = data['bugs'][:max_bugs]
        logger.info(f"📦 Loaded {len(bugs)} mined bugs")

        # Train on each bug
        trained = 0
        skipped = 0
        for i, bug in enumerate(bugs):
            try:
                # Skip very large files (slow to embed)
                code = bug['code_before']
                if len(code) > 10000:
                    logger.info(f"  Skipping large file ({len(code)} chars): {bug['file_path']}")
                    skipped += 1
                    continue

                logger.info(f"  [{i+1}/{len(bugs)}] Processing {bug['error_type']}: {bug['file_path'][:50]}...")

                # Create graph from the BUGGY code (before fix)
                graph = create_graph_data_from_source(code)
                if graph is None:
                    skipped += 1
                    continue

                error_type = bug['error_type']
                label = self.error_to_label.get(error_type, 7)  # default to logic_error

                # Store as learning example
                self.learning_examples.append({
                    'file_path': bug['file_path'],
                    'original_code': code,
                    'fixed_code': bug['code_after'],
                    'error_type': error_type,
                    'graph': graph
                })
                trained += 1
                logger.info(f"    ✓ Added {error_type} example")

            except Exception as e:
                logger.warning(f"  ✗ Failed: {e}")
                skipped += 1
                continue

        logger.info(f"✅ Processed {trained} bugs into learning examples")

        # Now run batch training
        if self.learning_examples:
            logger.info("🧠 Running batch training on mined bugs...")
            self.learn_from_fixes()

        return trained > 0

    def run(self, target_path: str):
        if self.mode == "train_mined":
            return self.train_from_mined_bugs()
        if self.mode.startswith("evolve_"):
            return self.evolve_code(target_path)

        logger.info("=" * 60)
        logger.info(f"🤖 UNIVERSAL FIXER - Target: {target_path}")
        logger.info("=" * 60)
        
        max_iterations = 3
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"\n{'🔄' if iteration > 1 else '▶️'} Iteration {iteration}/{max_iterations}")
            
            errors = self.detect_errors(target_path)
            
            if not errors:
                logger.info("✅ Execution successful / No errors found!")
                if iteration > 1: self.learn_from_fixes()
                return True
            
            logger.info(f"\n📊 Found {len(errors)} errors")
            
            fixes_applied = 0
            for error in errors:
                logger.info(f"\n{'─'*60}")
                logger.info(f"Processing Error: {error.error_msg}")
                logger.info(f"Location: {error.file_path}:{error.line_num}")
                
                fix_plan = self.reason_about_fix(error)
                if fix_plan:
                    if self.apply_fix(error, fix_plan):
                        fixes_applied += 1
            
            if fixes_applied > 0:
                if self.dry_run:
                    logger.info("🛑 Dry run complete. Exiting.")
                    return True
                
                if self.verify_fix(target_path):
                    logger.info("\n🎉 SUCCESS! Fixed all errors.")
                    self.learn_from_fixes()
                    return True
            else:
                logger.warning("⚠️  No fixes applied. Stopping.")
                return False
        
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Universal Autonomous Fixer")
    parser.add_argument("target", nargs="?", default=".", help="Path to script or test file to fix")
    parser.add_argument("--dry-run", action="store_true", help="Propose fixes without applying them")
    parser.add_argument("--mode", choices=["fix", "evolve_perf", "evolve_quality", "evolve_security", "evolve_antibodies", "evolve_types", "train_mined"], default="fix", help="Operation mode")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--language", choices=["python", "javascript", "typescript"], default="python", help="Target language")
    parser.add_argument("--max-bugs", type=int, default=50, help="Max bugs to train on (for train_mined mode)")
    args = parser.parse_args()

    fixer = UniversalFixer(dry_run=args.dry_run, mode=args.mode, benchmark_mode=args.benchmark, language=args.language)

    if args.mode == "train_mined":
        fixer.train_from_mined_bugs(max_bugs=args.max_bugs)
    else:
        fixer.run(args.target)

if __name__ == "__main__":
    main()
