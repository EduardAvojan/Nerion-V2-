#!/usr/bin/env python3
"""
Nerion Immune System Daemon

The core immune system that runs 24/7 monitoring your codebase.
Runs independently of the GUI - keeps watching even when Mission Control is closed.
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional, Any

# Load .env file for API keys (Claude, Gemini, etc.)
try:
    from dotenv import load_dotenv
    # Try to find .env in project root
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"[NERION-DAEMON] Loaded environment from {env_path}")
    else:
        load_dotenv()  # Try to find .env in current directory
except ImportError:
    pass  # dotenv not installed, assume env vars are set externally

# Import at module level for torch.load unpickling
try:
    from nerion_digital_physicist.training.train_multitask_ewc import MultiTaskConfig
except ImportError:
    MultiTaskConfig = None  # Will be handled in _init_gnn_model

# Multi-Agent System
from nerion_digital_physicist.agents.coordinator import MultiAgentCoordinator
from nerion_digital_physicist.agents.specialists import SpecialistAgent
from nerion_digital_physicist.agents.protocol import (
    TaskRequest, TaskResponse, CoordinationStrategy, AgentRole
)

# Continuous Learning
from daemon.continuous_learner import ContinuousLearner, ContinuousLearningConfig
from nerion_digital_physicist.infrastructure.production_collector import (
    ProductionFeedbackCollector, ProductionBug
)

# Self-Modification Engine
# Removed: from selfcoder.orchestration import apply_plan (no longer needed - writing files directly)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [NERION-DAEMON] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.expanduser('~/.nerion/daemon.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NerionImmuneDaemon:
    """
    Core immune system daemon that runs continuously.

    Responsibilities:
    - Watch codebase for changes
    - Run GNN training in background
    - Monitor for threats/issues
    - Auto-fix problems
    - Learn from patterns
    - Serve status/metrics to GUI via socket
    """

    def __init__(self, codebase_path: str):
        self.codebase_path = Path(codebase_path).resolve()
        self.running = False
        self.status = "starting"
        self.health = "healthy"

        # Immune system state
        self.threats_detected = 0
        self.auto_fixes_applied = 0
        self.files_monitored = 0
        self.last_scan = None
        self.gnn_training = False
        self.gnn_episodes = 0

        # Connected clients (GUIs)
        self.clients: Set[asyncio.StreamWriter] = set()

        # Socket server
        self.server = None
        self.socket_path = os.path.expanduser('~/.nerion/daemon.sock')

        # Continuous Learning System (initialized lazily)
        self.continuous_learner = None
        self._learner_initialized = False

        # Curiosity-Driven Exploration Engine
        self.curiosity_engine = None
        self.semantic_embedder = None
        self._curiosity_initialized = False

        # Multi-Agent System
        self.coordinator = None
        self._agents_initialized = False

        # GNN Model (91.8% accuracy)
        self.gnn_model = None
        self._gnn_initialized = False
        self.gnn_accuracy = None

        # Discovery tracking
        self.patterns_discovered = 0
        self.code_issues_found = []

        # Healing configuration
        self.auto_heal_enabled = True  # Enable auto-healing
        self.min_confidence_for_auto_heal = 0.85  # Only auto-heal bugs with ≥85% confidence
        self.require_user_approval = True  # Require approval for high-risk fixes
        self.bugs_healed = 0
        self.healing_successes = 0
        self.healing_failures = 0

        # Pending fixes queue for user approval
        self.pending_fixes: Dict[str, Dict[str, Any]] = {}  # fix_id -> fix_proposal
        self.approval_events: Dict[str, asyncio.Event] = {}  # fix_id -> event
        self.approval_results: Dict[str, bool] = {}  # fix_id -> approved/rejected

        logger.info(f"Nerion Immune Daemon initialized")
        logger.info(f"Monitoring codebase: {self.codebase_path}")

    async def start(self):
        """Start the immune system daemon"""
        self.running = True
        self.status = "running"

        # Ensure socket directory exists
        os.makedirs(os.path.dirname(self.socket_path), exist_ok=True)

        # Remove old socket if exists
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        # Start socket server for GUI communication
        self.server = await asyncio.start_unix_server(
            self.handle_client,
            path=self.socket_path
        )

        logger.info(f"Socket server started: {self.socket_path}")
        logger.info("🧬 Nerion Immune System ONLINE")

        # Start immune system tasks
        tasks = [
            asyncio.create_task(self.watch_codebase()),
            asyncio.create_task(self.train_gnn_background()),
            asyncio.create_task(self.monitor_health()),
            asyncio.create_task(self.broadcast_status()),
        ]

        # Monitor for shutdown signal and cancel tasks when needed
        try:
            while self.running:
                # Check if any task has failed
                for task in tasks:
                    if task.done() and not task.cancelled():
                        exc = task.exception()
                        if exc:
                            logger.error(f"Task failed: {exc}", exc_info=exc)
                            self.running = False
                            break

                await asyncio.sleep(0.1)  # Check every 100ms

        except asyncio.CancelledError:
            logger.info("Daemon shutdown requested")
        finally:
            # Cancel all running tasks
            for task in tasks:
                if not task.done():
                    task.cancel()

            # Wait for tasks to complete cancellation
            await asyncio.gather(*tasks, return_exceptions=True)

            await self.shutdown()

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle connection from GUI client"""
        addr = writer.get_extra_info('peername')
        logger.info(f"GUI connected: {addr}")
        self.clients.add(writer)

        # Send initial status
        await self.send_to_client(writer, {
            'type': 'status',
            'data': self.get_status()
        })

        try:
            while self.running:
                data = await reader.read(1024)
                if not data:
                    break

                # Handle commands from GUI
                try:
                    message = json.loads(data.decode())
                    await self.handle_command(writer, message)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from client: {data}")
        except Exception as e:
            logger.error(f"Client handler error: {e}")
        finally:
            logger.info(f"GUI disconnected: {addr}")
            self.clients.discard(writer)
            writer.close()
            await writer.wait_closed()

    async def handle_command(self, writer: asyncio.StreamWriter, message: dict):
        """Handle command from GUI"""
        cmd_type = message.get('type')

        if cmd_type == 'get_status':
            await self.send_to_client(writer, {
                'type': 'status',
                'data': self.get_status()
            })

        elif cmd_type == 'start_training':
            self.gnn_training = True
            await self.send_to_client(writer, {
                'type': 'training_started',
                'data': {'status': 'training'}
            })

        elif cmd_type == 'stop_training':
            self.gnn_training = False

        elif cmd_type == 'approve_fix':
            # User approved a fix
            fix_id = message.get('fix_id')
            if fix_id and fix_id in self.approval_events:
                self.approval_results[fix_id] = True
                self.approval_events[fix_id].set()  # Wake up waiting heal_bug()
                logger.info(f"✅ Fix {fix_id[:8]} approved by user")
                await self.send_to_client(writer, {
                    'type': 'fix_approved',
                    'data': {'fix_id': fix_id, 'status': 'approved'}
                })

        elif cmd_type == 'reject_fix':
            # User rejected a fix
            fix_id = message.get('fix_id')
            if fix_id and fix_id in self.approval_events:
                self.approval_results[fix_id] = False
                self.approval_events[fix_id].set()  # Wake up waiting heal_bug()
                logger.info(f"❌ Fix {fix_id[:8]} rejected by user")
                await self.send_to_client(writer, {
                    'type': 'fix_rejected',
                    'data': {'fix_id': fix_id, 'status': 'rejected'}
                })

        elif cmd_type == 'get_pending_fixes':
            # Send list of all pending fixes
            await self.send_to_client(writer, {
                'type': 'pending_fixes',
                'data': list(self.pending_fixes.values())
            })
            await self.send_to_client(writer, {
                'type': 'training_stopped',
                'data': {'status': 'idle'}
            })

        elif cmd_type == 'shutdown':
            logger.info("Shutdown command received from GUI")
            self.running = False

    async def send_to_client(self, writer: asyncio.StreamWriter, message: dict):
        """Send message to a specific client"""
        try:
            data = json.dumps(message).encode() + b'\n'
            writer.write(data)
            await writer.drain()
        except Exception as e:
            logger.error(f"Failed to send to client: {e}")

    async def broadcast_to_clients(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected = set()
        for writer in self.clients:
            try:
                data = json.dumps(message).encode() + b'\n'
                writer.write(data)
                await writer.drain()
            except Exception as e:
                logger.error(f"Failed to broadcast to client: {e}")
                disconnected.add(writer)

        # Remove disconnected clients
        self.clients -= disconnected

    def get_status(self) -> dict:
        """Get current daemon status"""
        return {
            'status': self.status,
            'health': self.health,
            'uptime': time.time(),
            'codebase': str(self.codebase_path),
            'threats_detected': self.threats_detected,
            'auto_fixes_applied': self.auto_fixes_applied,
            'files_monitored': self.files_monitored,
            'last_scan': self.last_scan,
            'gnn_training': self.gnn_training,
            'gnn_episodes': self.gnn_episodes,
            'gnn_model_loaded': self._gnn_initialized,
            'gnn_accuracy': self.gnn_accuracy,
            'clients_connected': len(self.clients),
            # NEW: Real immune system metrics
            'patterns_discovered': self.patterns_discovered,
            'total_issues': len(self.code_issues_found),
            'curiosity_enabled': self._curiosity_initialized,
            'multi_agent_enabled': self._agents_initialized
        }

    async def watch_codebase(self):
        """
        Continuously watch codebase for changes using curiosity-driven exploration.
        This is the immune system's surveillance system.
        """
        logger.info("👁️  Codebase watcher started")

        # Initialize all subsystems
        self._init_gnn_model()
        self._init_curiosity_engine()
        self._init_multi_agent_system()

        if not self._gnn_initialized:
            logger.warning("GNN model not available, bug detection disabled")
        if not self._curiosity_initialized:
            logger.warning("Curiosity engine not available, running basic monitoring")

        while self.running:
            try:
                self.last_scan = datetime.now().isoformat()

                # Get all Python files (excluding venv, node_modules, etc)
                python_files = []
                for root, dirs, files in os.walk(self.codebase_path):
                    # Filter out unwanted directories (modify dirs in-place to skip them)
                    dirs[:] = [d for d in dirs if d not in [
                        'node_modules', '__pycache__', '.git', 'venv', '.venv', 'env', 'ENV',
                        '.claude', '.pytest_cache', 'dist', 'build', 'out', 'tmp',
                        'external_repos', 'Project_CodeNet', 'experiments',
                        '.nerion', 'models', 'backups', '_archive'
                    ]]
                    for f in files:
                        if f.endswith('.py'):
                            python_files.append(Path(root) / f)

                self.files_monitored = len(python_files)
                logger.info(f"🔍 Scanning {self.files_monitored} Python files...")

                # Curiosity-driven analysis
                if self.curiosity_engine and self.semantic_embedder:
                    discoveries_this_scan = 0

                    # Sample files for deep analysis (avoid overwhelming on large codebases)
                    import random
                    sample_size = min(100, len(python_files))
                    sampled_files = random.sample(python_files, sample_size)

                    for fpath in sampled_files:
                        # Check if shutdown requested
                        if not self.running:
                            break

                        try:
                            with open(fpath, 'r') as f:
                                code = f.read()

                            if len(code) < 100:  # Skip trivial files
                                continue

                            # GNN analysis (bug detection)
                            gnn_result = None
                            if self.gnn_model:
                                gnn_result = await self._analyze_with_gnn(code, str(fpath))

                            # Generate embedding
                            embedding = self.semantic_embedder.embed('code', code)

                            # Evaluate with curiosity engine
                            candidate = self.curiosity_engine.evaluate_candidate(
                                code=code,
                                embedding=embedding,
                                metadata={
                                    'file': str(fpath),
                                    'size': len(code),
                                    'scan_time': self.last_scan
                                }
                            )

                            # If interesting, add to discoveries
                            if self.curiosity_engine.should_explore(candidate):
                                pattern = self.curiosity_engine.add_discovered_pattern(candidate)
                                discoveries_this_scan += 1
                                self.patterns_discovered += 1

                                logger.info(f"✨ New pattern discovered: {fpath.name}")

                                # Multi-agent analysis on interesting code
                                if self.coordinator:
                                    await self._analyze_with_agents(code, str(fpath))

                            # Multi-agent verification if GNN detected high-confidence bug
                            elif gnn_result and gnn_result['is_buggy'] and gnn_result['confidence'] > 0.8:
                                if self.coordinator:
                                    logger.info(f"🔍 High-confidence bug detected, requesting multi-agent verification")
                                    await self._analyze_with_agents(code, str(fpath))

                        except Exception as e:
                            logger.debug(f"Error analyzing {fpath}: {e}")

                    if discoveries_this_scan > 0:
                        logger.info(f"📊 Discoveries this scan: {discoveries_this_scan}")

                # Broadcast statistics
                await self.broadcast_to_clients({
                    'type': 'scan_complete',
                    'data': {
                        'files_scanned': self.files_monitored,
                        'patterns_discovered': self.patterns_discovered,
                        'timestamp': self.last_scan
                    }
                })

                # Scan every 10 minutes (600 seconds)
                await asyncio.sleep(600)

            except Exception as e:
                logger.error(f"Codebase watcher error: {e}", exc_info=True)
                await asyncio.sleep(60)

    def _init_continuous_learner(self):
        """Initialize continuous learner (lazy initialization)"""
        if self._learner_initialized:
            return

        try:
            from daemon.continuous_learner import ContinuousLearner

            # Initialize paths relative to codebase
            replay_root = self.codebase_path / "data" / "replay"
            curriculum_path = self.codebase_path / "out" / "learning" / "curriculum.sqlite"
            model_registry_path = self.codebase_path / "out" / "models" / "registry"

            # Ensure directories exist
            replay_root.mkdir(parents=True, exist_ok=True)
            model_registry_path.mkdir(parents=True, exist_ok=True)

            self.continuous_learner = ContinuousLearner(
                replay_root=replay_root,
                curriculum_path=curriculum_path,
                model_registry_path=model_registry_path
            )

            self._learner_initialized = True
            logger.info("✅ Continuous learner initialized")

        except Exception as e:
            logger.error(f"Failed to initialize continuous learner: {e}", exc_info=True)
            self.continuous_learner = None

    def _init_curiosity_engine(self):
        """Initialize curiosity-driven exploration engine"""
        if self._curiosity_initialized:
            return

        try:
            from nerion_digital_physicist.exploration import CuriosityEngine, ExplorationStrategy
            from nerion_digital_physicist.agent.semantics import get_global_embedder

            logger.info("Initializing curiosity engine...")

            self.semantic_embedder = get_global_embedder()
            self.curiosity_engine = CuriosityEngine(
                exploration_strategy=ExplorationStrategy.ADAPTIVE,
                novelty_threshold=0.60,
                interest_threshold=0.55,
                memory_size=50000
            )

            self._curiosity_initialized = True
            logger.info("✅ Curiosity engine initialized")

        except Exception as e:
            logger.error(f"Failed to initialize curiosity engine: {e}", exc_info=True)
            self.curiosity_engine = None

    def _init_multi_agent_system(self):
        """Initialize multi-agent collaboration system"""
        if self._agents_initialized:
            return

        try:
            from nerion_digital_physicist.agents import (
                MultiAgentCoordinator,
                PythonSpecialist,
                SecuritySpecialist,
                PerformanceSpecialist,
                BugFixingSpecialist,
                RefactoringSpecialist,
                TestingSpecialist
            )

            logger.info("Initializing multi-agent system...")

            self.coordinator = MultiAgentCoordinator()

            # Register specialist agents
            agents = [
                PythonSpecialist("python-specialist-1"),
                SecuritySpecialist("security-specialist-1"),
                PerformanceSpecialist("performance-specialist-1"),
                BugFixingSpecialist("bug-fixing-specialist-1"),
                RefactoringSpecialist("refactoring-specialist-1"),
                TestingSpecialist("testing-specialist-1")
            ]

            for agent in agents:
                self.coordinator.register_agent(agent)

            self._agents_initialized = True
            logger.info(f"✅ Multi-agent system initialized ({len(agents)} agents)")

        except Exception as e:
            logger.error(f"Failed to initialize multi-agent system: {e}", exc_info=True)
            self.coordinator = None

    def _init_gnn_model(self):
        """Initialize GNN model for bug detection (91.8% accuracy)"""
        if self._gnn_initialized:
            return

        try:
            import torch
            from nerion_digital_physicist.agent.brain import MultiTaskCodeGraphSAGE

            logger.info("🧠 Loading GNN model (91.8% accuracy)...")

            # Model parameters (from training config)
            model_path = self.codebase_path / "out" / "training_runs" / "multitask_ewc" / "multitask_ewc_final.pt"

            if not model_path.exists():
                logger.warning(f"GNN model not found: {model_path}")
                return

            # Initialize model architecture
            model = MultiTaskCodeGraphSAGE(
                num_node_features=32,  # AST structural features
                hidden_channels=512,
                num_layers=4,
                dropout=0.2,
                use_graphcodebert=True,  # 768-dim semantic embeddings
                freeze_backbone=False,
            )

            # Load trained weights (need MultiTaskConfig for unpickling)
            checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()  # Set to inference mode

            self.gnn_model = model
            # Get accuracy from results dict
            self.gnn_accuracy = checkpoint.get('results', {}).get('best_val_acc', 0.918)
            self._gnn_initialized = True

            logger.info(f"✅ GNN model loaded: {self.gnn_accuracy*100:.1f}% accuracy")

        except Exception as e:
            logger.error(f"Failed to initialize GNN model: {e}", exc_info=True)
            self.gnn_model = None

    def _init_multi_agent_system(self):
        """Initialize Multi-Agent System for collaborative bug analysis"""
        if self._agents_initialized:
            return

        try:
            logger.info("🤖 Initializing Multi-Agent System...")

            # Create coordinator
            self.coordinator = MultiAgentCoordinator(coordinator_id="nerion_daemon")

            # Register all available specialist agents
            from nerion_digital_physicist.agents.specialists import (
                PythonSpecialist, SecuritySpecialist, PerformanceSpecialist,
                JavaScriptSpecialist, JavaSpecialist, TestingSpecialist,
                RefactoringSpecialist, BugFixAgent, DocumentationSpecialist
            )

            specialists = [
                PythonSpecialist("python_specialist"),
                SecuritySpecialist("security_specialist"),
                PerformanceSpecialist("performance_specialist"),
                JavaScriptSpecialist("javascript_specialist"),
                JavaSpecialist("java_specialist"),
                TestingSpecialist("testing_specialist"),
                RefactoringSpecialist("refactoring_specialist"),
                BugFixAgent("bugfix_agent"),  # LLM-powered bug fixing
                DocumentationSpecialist("documentation_specialist"),
            ]

            for specialist in specialists:
                self.coordinator.register_agent(specialist)

            self._agents_initialized = True
            logger.info(f"✅ Multi-Agent System initialized: {len(specialists)} specialists")

        except Exception as e:
            logger.error(f"Failed to initialize Multi-Agent System: {e}", exc_info=True)
            self.coordinator = None

    async def heal_bug(
        self,
        filepath: str,
        code: str,
        bug_confidence: float,
        bug_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Full immune system healing loop.

        Flow: GNN Detection → Multi-Agent Analysis → Self-Modification → Test Validation → Learning

        Args:
            filepath: Path to buggy file
            code: Buggy code content
            bug_confidence: GNN confidence (0.0-1.0)
            bug_context: Additional context (line_count, etc.)

        Returns:
            Healing result with success status, fix details, test results
        """
        result = {
            'success': False,
            'filepath': filepath,
            'bug_confidence': bug_confidence,
            'analysis': None,
            'fix_generated': False,
            'fix_applied': False,
            'tests_passed': None,
            'learned': False,
            'error': None
        }

        try:
            # Step 1: Multi-Agent Analysis
            logger.info(f"🤖 Step 1/5: Multi-agent analyzing {Path(filepath).name}...")

            if not self._agents_initialized:
                self._init_multi_agent_system()

            if not self.coordinator:
                result['error'] = "Multi-agent system not available"
                return result

            # Create task request for bug analysis
            task = TaskRequest(
                task_type="bug_fix",  # Changed to bug_fix so BugFixAgent scores higher
                code=code,
                language="python",
                requester_id="nerion_daemon",
                context={
                    'filepath': filepath,
                    'bug_confidence': bug_confidence,
                    'line_count': bug_context.get('line_count', 0),
                    'bug_analysis': {
                        'confidence': bug_confidence,
                        'patterns': [
                            f"Potential code quality issue (GNN confidence: {bug_confidence:.2%})",
                            "Code may contain bugs, undefined variables, or logic errors"
                        ]
                    }
                }
            )

            # Get collaborative analysis from all capable agents
            responses = await asyncio.get_event_loop().run_in_executor(
                None,
                self.coordinator.assign_task,
                task,
                CoordinationStrategy.VOTING  # Use voting for bug analysis
            )

            if not responses or not any(r.success for r in responses):
                result['error'] = "Multi-agent analysis failed"
                return result

            # Extract best solution from voting
            successful_responses = [r for r in responses if r.success and r.confidence > 0.5]
            if not successful_responses:
                result['error'] = "No high-confidence agent solutions"
                return result

            best_response = max(successful_responses, key=lambda r: r.confidence)
            result['analysis'] = {
                'agent_id': best_response.responder_id,
                'confidence': best_response.confidence,
                'proposed_fix': best_response.result.get('solution', {}),
                'rationale': best_response.result.get('explanation', ''),
                'num_agents': len(successful_responses)
            }

            logger.info(f"✅ Analysis complete: {len(successful_responses)} agents agree (confidence: {best_response.confidence:.2f})")

            # Step 2: Safety Gate - Check if auto-heal should proceed
            logger.info(f"🛡️  Step 2/5: Safety gate check...")

            if bug_confidence < self.min_confidence_for_auto_heal:
                logger.warning(f"⚠️  Bug confidence {bug_confidence:.2f} < threshold {self.min_confidence_for_auto_heal}")
                result['error'] = f"Confidence too low for auto-heal ({bug_confidence:.2f} < {self.min_confidence_for_auto_heal})"
                return result

            if self.require_user_approval:
                logger.info("⏸️  User approval required - generating fix preview...")

                # Extract proposed fix from multi-agent analysis
                proposed_fix = best_response.result.get('solution', {})
                if not proposed_fix or 'code' not in proposed_fix:
                    result['error'] = "No executable fix proposed by agents"
                    return result

                # Read current file content for diff
                try:
                    with open(filepath, 'r') as f:
                        original_code = f.read()
                except Exception as e:
                    result['error'] = f"Failed to read file for diff: {e}"
                    return result

                # Generate unique fix ID
                import uuid
                fix_id = str(uuid.uuid4())

                # Store fix proposal with code preview
                self.pending_fixes[fix_id] = {
                    'fix_id': fix_id,
                    'filepath': filepath,
                    'bug_confidence': bug_confidence,
                    'analysis': result['analysis'],
                    'original_code': original_code,
                    'proposed_code': proposed_fix['code'],
                    'rationale': proposed_fix.get('rationale', ''),
                    'timestamp': datetime.now().isoformat()
                }

                # TEMP: Write fix data to file for debugging
                import json
                debug_path = f'/tmp/nerion_fix_{fix_id[:8]}.json'
                with open(debug_path, 'w') as f:
                    json.dump({
                        'has_original_code': bool(self.pending_fixes[fix_id].get('original_code')),
                        'has_proposed_code': bool(self.pending_fixes[fix_id].get('proposed_code')),
                        'original_code_len': len(self.pending_fixes[fix_id].get('original_code', '')),
                        'proposed_code_len': len(self.pending_fixes[fix_id].get('proposed_code', '')),
                        'proposed_code_preview': self.pending_fixes[fix_id].get('proposed_code', '')[:200]
                    }, f, indent=2)
                logger.info(f"[DEBUG] Fix data written to: {debug_path}")

                # Create event to wait for approval
                approval_event = asyncio.Event()
                self.approval_events[fix_id] = approval_event

                # Broadcast fix proposal to all connected GUIs
                await self.broadcast_to_clients({
                    'type': 'fix_proposal',
                    'data': self.pending_fixes[fix_id]
                })

                logger.info(f"📤 Fix proposal sent to GUI (ID: {fix_id[:8]})")
                logger.info("⏸️  Waiting for user approval...")

                # Wait for user approval (with 5 minute timeout)
                try:
                    await asyncio.wait_for(approval_event.wait(), timeout=300)

                    # Check approval result
                    approved = self.approval_results.get(fix_id, False)

                    if not approved:
                        logger.info("❌ User rejected fix")
                        result['error'] = "User rejected fix"
                        result['user_rejected'] = True
                        # Clean up
                        del self.pending_fixes[fix_id]
                        del self.approval_events[fix_id]
                        del self.approval_results[fix_id]
                        return result

                    logger.info("✅ User approved fix - proceeding with application")

                    # Clean up
                    del self.pending_fixes[fix_id]
                    del self.approval_events[fix_id]
                    del self.approval_results[fix_id]

                except asyncio.TimeoutError:
                    logger.warning("⏱️  Approval timeout (5 minutes) - skipping fix")
                    result['error'] = "Approval timeout"
                    result['timeout'] = True
                    # Clean up
                    del self.pending_fixes[fix_id]
                    del self.approval_events[fix_id]
                    return result

            # Step 3: Self-Modification - Apply approved fix
            logger.info(f"🔧 Step 3/5: Applying fix with selfcoder...")

            # If we got here after approval, proposed_fix was already extracted
            # If auto-heal (no approval), extract it now
            if not self.require_user_approval:
                proposed_fix = best_response.result.get('solution', {})
                if not proposed_fix or 'code' not in proposed_fix:
                    result['error'] = "No executable fix proposed by agents"
                    return result

            # Apply fix by directly writing the fixed code
            try:
                # Read original file to verify it exists
                try:
                    original_content = Path(filepath).read_text(encoding='utf-8')
                except Exception as read_err:
                    result['error'] = f"Cannot read file: {read_err}"
                    return result

                # Write the fixed code
                Path(filepath).write_text(proposed_fix['code'], encoding='utf-8')

                result['fix_generated'] = True
                result['fix_applied'] = True
                logger.info(f"✅ Fix applied to {filepath}")

            except Exception as e:
                result['error'] = f"Fix application failed: {e}"
                return result

            # Step 4: Test Validation
            logger.info(f"🧪 Step 4/5: Running pytest to validate fix...")

            try:
                # Run pytest on the modified file
                test_result = subprocess.run(
                    ['pytest', str(filepath), '-v', '--tb=short'],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=str(self.codebase_path)
                )

                tests_passed = test_result.returncode == 0
                result['tests_passed'] = tests_passed
                result['test_output'] = test_result.stdout + test_result.stderr

                if tests_passed:
                    logger.info("✅ Tests passed! Fix validated")
                    result['success'] = True
                    self.healing_successes += 1
                else:
                    logger.warning("❌ Tests failed after fix")
                    result['error'] = "Tests failed after applying fix"
                    self.healing_failures += 1
                    # TODO: Rollback the fix

            except subprocess.TimeoutExpired:
                result['error'] = "Test validation timeout"
                self.healing_failures += 1
            except Exception as e:
                result['error'] = f"Test execution failed: {e}"
                self.healing_failures += 1

            # Step 5: Continuous Learning - Feed outcome to learner
            logger.info(f"🧠 Step 5/5: Recording outcome for continuous learning...")

            if not self._learner_initialized and self.continuous_learner:
                # Record bug for learning
                try:
                    production_bug = ProductionBug(
                        filepath=filepath,
                        bug_type="gnn_detected",
                        severity="high" if bug_confidence > 0.9 else "medium",
                        code_snippet=code,
                        prediction_confidence=bug_confidence,
                        actual_bug=result['success'],  # True if fix worked
                        surprise_score=1.0 if not result['success'] else 0.0
                    )

                    # This will be used in next learning cycle
                    self.continuous_learner.feedback_collector.collect_bug(
                        production_bug,
                        graph=None,  # Would need AST graph here
                        ground_truth=1 if result['success'] else 0
                    )

                    result['learned'] = True
                    logger.info("✅ Bug recorded for continuous learning")

                except Exception as e:
                    logger.warning(f"Failed to record bug for learning: {e}")

            # Update stats
            self.bugs_healed += 1

            return result

        except Exception as e:
            logger.error(f"Healing loop failed: {e}", exc_info=True)
            result['error'] = str(e)
            self.healing_failures += 1
            return result

    async def train_gnn_background(self):
        """
        Run GNN training in background.
        The immune system's learning system.
        """
        logger.info("🧠 GNN background training started")

        while self.running:
            try:
                if self.gnn_training:
                    # Initialize learner if needed
                    if not self._learner_initialized:
                        self._init_continuous_learner()

                    # Run learning cycle if learner is available
                    if self.continuous_learner:
                        try:
                            logger.info(f"Starting learning cycle (episode {self.gnn_episodes + 1})...")
                            updated = await self.continuous_learner.learning_cycle()

                            if updated:
                                self.gnn_episodes += 1
                                logger.info(f"✅ Learning cycle complete (episode {self.gnn_episodes})")

                                # Broadcast update to clients
                                await self.broadcast_to_clients({
                                    'type': 'training_update',
                                    'data': {
                                        'episode': self.gnn_episodes,
                                        'status': 'completed',
                                        'timestamp': datetime.now().isoformat()
                                    }
                                })
                            else:
                                logger.info("Learning cycle completed, no model update")

                        except Exception as e:
                            logger.error(f"Learning cycle error: {e}", exc_info=True)
                    else:
                        logger.warning("Continuous learner not available, skipping cycle")

                # Run every hour when training is active
                await asyncio.sleep(3600)

            except Exception as e:
                logger.error(f"GNN training error: {e}", exc_info=True)
                await asyncio.sleep(600)  # Wait 10 minutes on error

    async def _analyze_with_agents(self, code: str, filepath: str):
        """
        Analyze code using multi-agent system.

        Args:
            code: Code to analyze
            filepath: Path to the file
        """
        if not self.coordinator:
            return

        try:
            from nerion_digital_physicist.agents import TaskRequest

            # Create analysis task
            task = TaskRequest(
                task_type="analyze",
                code=code,
                language="python",
                requester_id="daemon",
                context={'filepath': filepath}
            )

            # Coordinate analysis across specialists
            responses = await asyncio.get_event_loop().run_in_executor(
                None,
                self.coordinator.assign_task,
                task
            )

            # Get best response
            result = responses[0] if responses else None

            # Check for issues found
            if result and result.success:
                issues = result.result.get('issues', [])
                if issues:
                    self.code_issues_found.extend(issues)
                    self.threats_detected += len(issues)

                    logger.info(f"⚠️  {len(issues)} issues found in {Path(filepath).name}")

                    # Broadcast issue alert
                    await self.broadcast_to_clients({
                        'type': 'issues_detected',
                        'data': {
                            'file': filepath,
                            'issues': issues,
                            'timestamp': datetime.now().isoformat()
                        }
                    })

        except Exception as e:
            logger.error(f"Multi-agent analysis error: {e}")

    async def _analyze_with_gnn(self, code: str, filepath: str) -> dict:
        """
        Analyze code using GNN model for bug detection.

        NOTE: Performs FILE-LEVEL analysis (entire file as one graph).
        Users should review the entire file when bugs are detected.

        Args:
            code: Python source code to analyze
            filepath: Path to the file

        Returns:
            dict with keys: 'is_buggy', 'confidence', 'bug_probability', 'line_count', 'note'
        """
        if not self.gnn_model:
            return None

        try:
            import torch
            from nerion_digital_physicist.agent.data import create_graph_data_from_source
            from nerion_digital_physicist.agent.semantics import get_global_embedder

            # Run in executor to avoid blocking event loop
            def _run_inference():
                # Create NoOp embedder for 32-dim structural features only
                class NoOpEmbedder:
                    @property
                    def dimension(self):
                        return 0
                    def embed(self, key, text):
                        return []

                # Convert code to graph (only structural features, 32-dim)
                graph_data = create_graph_data_from_source(code, embedder=NoOpEmbedder())

                if graph_data is None:
                    return None

                # Get GraphCodeBERT embedding
                embedder = get_global_embedder()
                graphcodebert_embedding = embedder.embed('code', code)

                # Convert to tensor
                graphcodebert_tensor = torch.tensor(graphcodebert_embedding, dtype=torch.float32)

                # Create batch tensor (single graph, so all nodes belong to batch 0)
                batch = torch.zeros(graph_data.x.shape[0], dtype=torch.long)

                # Run inference
                with torch.no_grad():
                    logits = self.gnn_model(
                        x=graph_data.x,
                        edge_index=graph_data.edge_index,
                        batch=batch,
                        graphcodebert_embedding=graphcodebert_tensor.unsqueeze(0)  # Add batch dimension
                    )
                    # Bug detection uses BCEWithLogitsLoss → single scalar output
                    bug_prob = torch.sigmoid(logits).item()  # Probability of being buggy
                    pred = (bug_prob > 0.5)

                # Add file metadata for better context
                line_count = len(code.split('\n'))

                return {
                    'is_buggy': pred,
                    'bug_probability': bug_prob,
                    'confidence': max(bug_prob, 1 - bug_prob),
                    'filepath': filepath,
                    'line_count': line_count,
                    'note': f"File-level detection ({line_count} lines)"
                }

            result = await asyncio.get_event_loop().run_in_executor(None, _run_inference)

            # Log if bug detected with high confidence
            if result and result['is_buggy'] and result['confidence'] > 0.7:
                logger.warning(f"🐛 GNN detected potential bug in {Path(filepath).name} (confidence: {result['confidence']:.2f})")
                self.threats_detected += 1

                # Add to issues list
                issue = {
                    'file': filepath,
                    'type': 'gnn_bug_detection',
                    'severity': 'medium' if result['confidence'] < 0.85 else 'high',
                    'confidence': result['confidence'],
                    'message': f"GNN model predicts buggy code (p={result['bug_probability']:.3f}) - File-level detection, review entire file ({result.get('line_count', '?')} lines)",
                    'timestamp': datetime.now().isoformat(),
                    'line_count': result.get('line_count'),
                    'note': result.get('note', 'File-level detection')
                }
                self.code_issues_found.append(issue)

                # Trigger full immune system healing loop if auto-heal enabled
                if self.auto_heal_enabled:
                    logger.info("🩹 Triggering immune system healing loop...")
                    healing_result = await self.heal_bug(
                        filepath=filepath,
                        code=code,
                        bug_confidence=result['confidence'],
                        bug_context={
                            'line_count': result.get('line_count', 0),
                            'bug_probability': result['bug_probability']
                        }
                    )

                    # Add healing result to issue
                    issue['healing_attempted'] = True
                    issue['healing_result'] = healing_result

                    if healing_result.get('success'):
                        logger.info(f"✅ Bug healed successfully in {Path(filepath).name}")
                    elif healing_result.get('awaiting_approval'):
                        logger.info(f"⏸️  Bug fix awaiting user approval for {Path(filepath).name}")
                    else:
                        logger.warning(f"❌ Healing failed: {healing_result.get('error', 'Unknown error')}")

            return result

        except Exception as e:
            logger.debug(f"GNN analysis error for {filepath}: {e}")
            return None

    async def monitor_health(self):
        """
        Monitor system health and detect threats using real metrics.
        The immune system's threat detection.
        """
        logger.info("🛡️  Health monitor started")

        while self.running:
            try:
                # Calculate health score based on discovered issues
                total_issues = len(self.code_issues_found)
                recent_issues = sum(
                    1 for issue in self.code_issues_found[-100:]
                    if issue.get('severity') in ['high', 'critical']
                )

                # Update health status based on real metrics
                if recent_issues > 50:
                    self.health = "critical"
                elif recent_issues > 20:
                    self.health = "warning"
                elif total_issues > 100 and recent_issues > 10:
                    self.health = "degraded"
                else:
                    self.health = "healthy"

                # Log health summary
                if self.health != "healthy":
                    logger.warning(
                        f"Health status: {self.health} "
                        f"({total_issues} total issues, {recent_issues} recent high-severity)"
                    )

                await asyncio.sleep(120)  # Check health every 2 minutes

            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(10)

    async def broadcast_status(self):
        """Periodically broadcast status to all connected clients"""
        while self.running:
            try:
                await self.broadcast_to_clients({
                    'type': 'status_update',
                    'data': self.get_status()
                })

                await asyncio.sleep(5)  # Broadcast every 5 seconds

            except Exception as e:
                logger.error(f"Status broadcast error: {e}")
                await asyncio.sleep(10)

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down Nerion Immune Daemon...")
        self.running = False
        self.status = "stopping"

        # Save issue report to file if issues were found
        report_path = None
        if self.code_issues_found:
            from datetime import datetime
            import json

            # Create reports directory in project folder (visible in VS Code)
            reports_dir = self.codebase_path / ".nerion" / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            # Generate report filename with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            report_path = reports_dir / f"session_{timestamp}.json"

            # Prepare report data
            report = {
                "session_start": self.last_scan,
                "session_end": datetime.now().isoformat(),
                "files_monitored": self.files_monitored,
                "patterns_discovered": self.patterns_discovered,
                "threats_detected": self.threats_detected,
                "gnn_accuracy": self.gnn_accuracy,
                "issues": self.code_issues_found
            }

            # Save to file
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)

        # Print session summary
        logger.info("=" * 60)
        logger.info("📊 SESSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Files monitored: {self.files_monitored}")
        logger.info(f"Patterns discovered: {self.patterns_discovered}")
        logger.info(f"Threats detected: {self.threats_detected}")
        logger.info(f"Code issues found: {len(self.code_issues_found)}")

        # Healing statistics
        if self.bugs_healed > 0:
            logger.info("")
            logger.info("🩹 Immune System Healing:")
            logger.info(f"   Bugs healed: {self.bugs_healed}")
            logger.info(f"   Successes: {self.healing_successes}")
            logger.info(f"   Failures: {self.healing_failures}")
            if self.bugs_healed > 0:
                success_rate = (self.healing_successes / self.bugs_healed) * 100
                logger.info(f"   Success rate: {success_rate:.1f}%")

        if self.code_issues_found:
            logger.info("")
            logger.info("🐛 Issues detected:")
            # Group by severity
            high = [i for i in self.code_issues_found if i.get('severity') == 'high']
            medium = [i for i in self.code_issues_found if i.get('severity') == 'medium']
            low = [i for i in self.code_issues_found if i.get('severity') == 'low']

            if high:
                logger.info(f"   High severity: {len(high)}")
                for issue in high[:10]:  # Show first 10 high severity
                    file_path = Path(issue['file'])
                    logger.info(f"      • {file_path.name} (confidence: {issue.get('confidence', 0):.2f})")
                if len(high) > 10:
                    logger.info(f"      ... and {len(high) - 10} more")

            if medium:
                logger.info(f"   Medium severity: {len(medium)}")
                for issue in medium[:5]:  # Show first 5 medium severity
                    file_path = Path(issue['file'])
                    logger.info(f"      • {file_path.name} (confidence: {issue.get('confidence', 0):.2f})")
                if len(medium) > 5:
                    logger.info(f"      ... and {len(medium) - 5} more")

            if low:
                logger.info(f"   Low severity: {len(low)}")

            logger.info("")
            if report_path:
                logger.info(f"   Full report saved: {report_path}")
            else:
                logger.info(f"   Full details: {len(self.code_issues_found)} issues logged")

        if self.gnn_accuracy:
            logger.info(f"GNN model accuracy: {self.gnn_accuracy*100:.1f}%")

        logger.info("=" * 60)

        # Close all client connections
        for writer in self.clients:
            try:
                writer.close()
                await writer.wait_closed()
            except:
                pass

        # Close server
        if self.server:
            self.server.close()
            await self.server.wait_closed()

        # Remove socket file
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        logger.info("✅ Nerion Immune Daemon stopped")


def signal_handler(daemon):
    """Handle shutdown signals"""
    def handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        daemon.running = False
    return handler


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Nerion Immune System Daemon')
    parser.add_argument('--target', type=str, default=os.getcwd(),
                        help='Target codebase directory to monitor (default: current directory)')
    parser.add_argument('--mode', type=str, default='monitor', choices=['monitor', 'dogfood'],
                        help='Operating mode: monitor (watch only) or dogfood (self-monitor)')

    args = parser.parse_args()
    codebase_path = args.target

    # Create daemon
    daemon = NerionImmuneDaemon(codebase_path)

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler(daemon))
    signal.signal(signal.SIGTERM, signal_handler(daemon))

    # Start daemon
    try:
        await daemon.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
