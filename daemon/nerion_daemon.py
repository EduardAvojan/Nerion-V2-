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
            'clients_connected': len(self.clients)
        }

    async def watch_codebase(self):
        """
        Continuously watch codebase for changes.
        This is the immune system's surveillance system.
        """
        logger.info("👁️  Codebase watcher started")

        while self.running:
            try:
                # Scan codebase
                self.last_scan = datetime.now().isoformat()

                # Count files being monitored
                python_files = list(self.codebase_path.rglob('*.py'))
                self.files_monitored = len(python_files)

                # TODO: Actual file watching with watchdog library
                # For now, periodic scans

                logger.debug(f"Scan complete: {self.files_monitored} files monitored")

                await asyncio.sleep(60)  # Scan every minute

            except Exception as e:
                logger.error(f"Codebase watcher error: {e}")
                await asyncio.sleep(10)

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

    async def monitor_health(self):
        """
        Monitor system health and detect threats.
        The immune system's threat detection.
        """
        logger.info("🛡️  Health monitor started")

        while self.running:
            try:
                # Check for issues
                # TODO: Actual health checks
                # - Code quality metrics
                # - Test coverage
                # - Security vulnerabilities
                # - Performance issues

                # Update health status
                if self.threats_detected > 10:
                    self.health = "warning"
                elif self.threats_detected > 50:
                    self.health = "critical"
                else:
                    self.health = "healthy"

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
