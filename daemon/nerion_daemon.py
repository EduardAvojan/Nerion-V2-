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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

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

        # Discovery tracking
        self.patterns_discovered = 0
        self.code_issues_found = []

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

        # Run until shutdown
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Daemon shutdown requested")
        finally:
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

        # Initialize curiosity engine and multi-agent system
        self._init_curiosity_engine()
        self._init_multi_agent_system()

        if not self._curiosity_initialized:
            logger.warning("Curiosity engine not available, running basic monitoring")

        while self.running:
            try:
                self.last_scan = datetime.now().isoformat()

                # Get all Python files (excluding venv, node_modules, etc)
                python_files = []
                for root, dirs, files in os.walk(self.codebase_path):
                    # Filter out unwanted directories
                    dirs[:] = [d for d in dirs if d not in [
                        'node_modules', '__pycache__', '.git', 'venv',
                        '.claude', '.pytest_cache', 'dist', 'build'
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
                        try:
                            with open(fpath, 'r') as f:
                                code = f.read()

                            if len(code) < 100:  # Skip trivial files
                                continue

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
                metadata={'filepath': filepath}
            )

            # Coordinate analysis across specialists
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.coordinator.coordinate_task,
                task
            )

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
    # Get codebase path from args or use current directory
    if len(sys.argv) > 1:
        codebase_path = sys.argv[1]
    else:
        codebase_path = os.getcwd()

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
