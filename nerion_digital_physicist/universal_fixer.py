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
from nerion_digital_physicist.agent.data import create_graph_data_object
from nerion_digital_physicist.agent.semantics import get_global_embedder
from nerion_digital_physicist.memory.episodic_memory import EpisodicMemory, Episode, EpisodeType, EpisodeOutcome
import torch

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
        """Initialize surprise-weighted replay buffer (Phase 4)"""
        try:
            replay_path = Path(__file__).parent.parent / "data" / "replay"
            self.replay_store = ReplayStore(storage_path=replay_path)
            self.feedback_collector = ProductionFeedbackCollector(
                replay_store=self.replay_store,
                model=self.model,
            )
            logger.info("🎲 Surprise-weighted replay initialized")
        except Exception as e:
            logger.warning(f"Could not initialize surprise replay: {e}")
            self.feedback_collector = None
        
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

            prompt += """
Task: Fix the code to resolve the error.
Analyze the traceback to find the root cause.
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
        elif "type hint" in msg_lower or "mypy" in msg_lower or "evolve_types" in error_msg or "Type Safety" in error_msg:
            return "type_safety"
        
        # Default: complexity reduction (most common quality improvement)
        else:
            return "complexity_reduction"

    def learn_from_fixes(self):
        """Step 5: Learn from all successful fixes"""
        if not self.enable_learning or not self.learning_examples: return

        logger.info(f"♾️  Learning from {len(self.learning_examples)} successful fixes")
        try:
            training_data = []
            for example in self.learning_examples:
                graph = example['graph']
                error_type = example['error_type']
                label = self.error_to_label.get(error_type, 4)
                training_data.append((graph, label))

                # Track examples by error type for MAML (Phase 3)
                if error_type in self.error_type_examples:
                    self.error_type_examples[error_type].append((graph, label))

                # Record in surprise-weighted replay buffer (Phase 4)
                self._record_surprise_experience(example, graph, label)

            # Get surprise-weighted replay samples to mix with new data
            replay_data = self._get_surprise_replay_samples()

            updated_model, update_info = self.learner.incremental_update(
                current_model=self.model,
                new_data=training_data,
                replay_data=replay_data  # Use surprise-weighted samples
            )
            self.model = updated_model
            self._save_model()
            logger.info(f"✅ GNN updated: accuracy={update_info.new_accuracy:.2%}")

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
        from nerion_digital_physicist.agent.brain import CodeGraphGCN
        import os
        
        # Dynamic path relative to project root
        project_root = Path(__file__).parent.parent
        model_path = project_root / "models" / "nerion_immune_brain.pt"
        
        if os.path.exists(model_path):
            try:
                model = CodeGraphGCN(num_node_features=800, hidden_channels=256, num_classes=20, num_layers=4)
                model.load_state_dict(torch.load(model_path))
                return model
            except: pass
        return CodeGraphGCN(num_node_features=800, hidden_channels=256, num_classes=20, num_layers=4)

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
        """Query the GNN brain for insights about the code"""
        if not self.enable_learning or not self.model:
            return "No AI insight available (Learning disabled)"
            
        try:
            graph_data = create_graph_data_object(file_path, embedder=self.embedder)
            
            # Prepare batch for model
            from torch_geometric.data import Batch
            
            # Ensure graph_data has batch attribute
            if not hasattr(graph_data, 'batch') or graph_data.batch is None:
                graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long)
                
            self.model.eval()
            
            # Get health score using the new predict_health method
            health_score = self.model.predict_health(graph_data)
            
            # Interpret score
            health_status = "Healthy" if health_score < 0.5 else "Unhealthy"
            confidence = abs(health_score - 0.5) * 2  # 0.5 -> 0%, 0.0/1.0 -> 100%
            
            return f"GNN Analysis: Code Health Score {health_score:.2f} ({health_status}, Confidence: {confidence:.0%})"
        except Exception as e:
            logger.warning(f"Failed to get model insight: {e}")
            return "AI Insight unavailable due to error"

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
        logger.info("=" * 60)
        logger.info(f"🧬 EVOLVER (PERFORMANCE) - Target: {target_path}")
        logger.info("=" * 60)
        
        # 1. Profile
        logger.info("⏱️  Profiling current performance...")
        stats, initial_time = self._profile_code(target_path)
        if not stats:
            logger.error("Could not profile code.")
            return False
            
        logger.info(f"📊 Baseline Time: {initial_time:.4f}s")
        
        # 2. Analyze & Evolve
        logger.info("🧠 Analyzing for optimizations...")
        prompt_template = """You are an expert Python Performance Engineer.
File: {file_path}
Profile Stats: {stats}
Current Code:
```python
{code}
```
Task: Optimize for performance (O(n) or O(1)). Maintain EXACTLY the same functionality.
Return ONLY the optimized Python code."""
        
        if self._apply_evolution(target_path, prompt_template, stats=stats[:2000]):
            # 3. Verify
            logger.info("✅ Verifying evolution...")
            if self._verify_evolution(target_path, initial_time, check_speedup=True):
                logger.info("🎉 SUCCESS! Code optimized and verified.")
                # Store learning example
                self._store_fix_example(
                    Path(target_path), 
                    Path(target_path).with_suffix(Path(target_path).suffix + ".bak").read_text(),
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
            
        logger.info("=" * 60)
        logger.info(f"🧬 EVOLVER (QUALITY) - Target: {target_path}")
        logger.info("=" * 60)
        
        logger.info("🧠 Analyzing for Code Smells & Complexity...")
        prompt_template = """You are an expert Python Architect.
File: {file_path}
Current Code:
```python
{code}
```
Task: Refactor this code to improve Readability and Maintainability.
1. Apply SOLID principles.
2. Reduce Cognitive Complexity.
3. Fix naming conventions.
4. Add type hints if missing.
5. Maintain EXACTLY the same functionality.
Return ONLY the refactored Python code."""

        if self._apply_evolution(target_path, prompt_template):
            logger.info("✅ Verifying refactor...")
            if self._verify_evolution(target_path, 0.0, check_speedup=False):
                logger.info("🎉 SUCCESS! Code refactored and verified.")
                # Store learning example
                self._store_fix_example(
                    Path(target_path), 
                    Path(target_path).with_suffix(Path(target_path).suffix + ".bak").read_text(),
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
        logger.info("=" * 60)
        logger.info(f"🧬 EVOLVER (SECURITY) - Target: {target_path}")
        logger.info("=" * 60)
        
        logger.info("🧠 Scanning for Vulnerabilities...")
        prompt_template = """You are an expert Python Security Engineer.
File: {file_path}
Current Code:
```python
{code}
```
Task: Harden this code against vulnerabilities.
1. Fix SQL Injection / XSS risks.
2. Remove hardcoded secrets (use os.getenv).
3. Add input validation.
4. Maintain EXACTLY the same functionality.
Return ONLY the hardened Python code."""

        if self._apply_evolution(target_path, prompt_template):
            logger.info("✅ Verifying security patch...")
            if self._verify_evolution(target_path, 0.0, check_speedup=False):
                logger.info("🎉 SUCCESS! Code hardened and verified.")
                # Store learning example
                self._store_fix_example(
                    Path(target_path), 
                    Path(target_path).with_suffix(Path(target_path).suffix + ".bak").read_text(),
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
        logger.info("=" * 60)
        logger.info(f"🧬 EVOLVER (TYPES) - Target: {target_path}")
        logger.info("=" * 60)
        
        logger.info("🧠 Analyzing for Missing Types...")
        prompt_template = """You are an expert Python Typing Specialist.
File: {file_path}
Current Code:
```python
{code}
```
Task: Add Python type hints to this code to achieve 100% type safety.
1. Add types to all function arguments and return values.
2. Use `typing` module (List, Dict, Optional, etc.) or modern syntax.
3. Maintain EXACTLY the same functionality.
Return ONLY the typed Python code."""

        if self._apply_evolution(target_path, prompt_template):
            logger.info("✅ Verifying types...")
            # 1. Syntax check
            if self._verify_evolution(target_path, 0.0, check_speedup=False):
                # 2. Optional: Run mypy if installed
                try:
                    import mypy.api
                    logger.info("🔍 Running mypy verification...")
                    stdout, stderr, exit_code = mypy.api.run([target_path])
                    if exit_code == 0:
                        logger.info("🎉 SUCCESS! Code typed and mypy verified.")
                    else:
                        logger.warning(f"⚠️  Mypy found issues (but code runs): {stdout.splitlines()[0]}")
                        # We still accept it if it runs, but log the warning
                except ImportError:
                    logger.info("🎉 SUCCESS! Code typed (mypy not installed).")
                
                # Store learning example
                self._store_fix_example(
                    Path(target_path), 
                    Path(target_path).with_suffix(Path(target_path).suffix + ".bak").read_text(),
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
        """Run cProfile on the target script"""
        logger.info(f"Running cProfile on {target_path}")
        
        # Create a runner script that imports and runs the target
        # This is safer than exec() and allows cProfile to work properly
        import cProfile
        import pstats
        import io
        
        start_time = time.time()
        try:
            # Run the script in a subprocess to ensure clean state
            # But for profiling we need to capture internal function calls.
            # Subprocess is harder to profile unless we inject code.
            # Let's try running it in-process but with care.
            
            # Actually, for the fixer, we want to profile the *execution* of the script.
            # The best way is to run `python -m cProfile -o output.pstats script.py`
            # and then parse the output file.
            
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
        """Verify the new code works and optionally is faster"""
        # 1. Functional Correctness
        logger.info("🧪 Verifying functional correctness...")
        errors = self.detect_errors(target_path)
        if errors:
            logger.error(f"Evolution broke the code: {errors[0].error_msg}")
            return False
            
        # 2. Performance Check (Optional)
        if check_speedup:
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

    def run(self, target_path: str):
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
    parser.add_argument("target", help="Path to script or test file to fix")
    parser.add_argument("--dry-run", action="store_true", help="Propose fixes without applying them")
    parser.add_argument("--mode", choices=["fix", "evolve_perf", "evolve_quality", "evolve_security", "evolve_antibodies", "evolve_types"], default="fix", help="Operation mode")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--language", choices=["python", "javascript", "typescript"], default="python", help="Target language")
    args = parser.parse_args()
    
    fixer = UniversalFixer(dry_run=args.dry_run, mode=args.mode, benchmark_mode=args.benchmark, language=args.language)
    fixer.run(args.target)

if __name__ == "__main__":
    main()
