#!/usr/bin/env python3
"""
Nerion Mission Control - Terminal Server
=========================================

FastAPI backend that provides:
1. WebSocket terminal with real PTY (bash shell with nerion commands)
2. WebSocket events stream for dashboard updates
3. REST API for system status

This serves as the backend for the web-based mission control cockpit.
"""
import asyncio
import json
import os
import pty
import select
import struct
import termios
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


app = FastAPI(title="Nerion Mission Control API", version="1.0.0")

# Enable CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PTYManager:
    """Manages a pseudo-terminal (PTY) for running a real bash shell."""

    def __init__(self):
        self.master_fd: Optional[int] = None
        self.pid: Optional[int] = None
        self.running = False

    def spawn_shell(self) -> tuple[int, int]:
        """Spawn a bash shell in a PTY.

        Returns:
            Tuple of (pid, master_fd)
        """
        # Fork a PTY
        pid, master_fd = pty.fork()

        if pid == 0:  # Child process
            # Set up environment
            env = os.environ.copy()

            # Ensure nerion is in PATH
            nerion_path = Path(__file__).parent.parent.parent  # /Users/ed/Nerion-V2
            if str(nerion_path) not in env.get('PYTHONPATH', ''):
                env['PYTHONPATH'] = f"{nerion_path}:{env.get('PYTHONPATH', '')}"

            # Set terminal type
            env['TERM'] = 'xterm-256color'

            # Execute bash
            os.execlpe('/bin/bash', 'bash', env)
        else:  # Parent process
            self.pid = pid
            self.master_fd = master_fd
            self.running = True

            # Set terminal size (default 80x24)
            self._set_terminal_size(80, 24)

            return pid, master_fd

    def _set_terminal_size(self, cols: int, rows: int):
        """Set the terminal window size."""
        if self.master_fd is not None:
            winsize = struct.pack("HHHH", rows, cols, 0, 0)
            try:
                termios.TIOCSWINSZ = getattr(termios, 'TIOCSWINSZ', 0x5414)
                import fcntl
                fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, winsize)
            except Exception as e:
                print(f"Failed to set terminal size: {e}")

    def resize(self, cols: int, rows: int):
        """Resize the terminal."""
        self._set_terminal_size(cols, rows)

    def write(self, data: bytes):
        """Write data to the PTY (user input)."""
        if self.master_fd is not None and self.running:
            try:
                os.write(self.master_fd, data)
            except OSError:
                self.running = False

    def read(self, timeout: float = 0.1) -> bytes:
        """Read data from the PTY (shell output).

        Args:
            timeout: Read timeout in seconds

        Returns:
            Output bytes from shell
        """
        if self.master_fd is None or not self.running:
            return b''

        try:
            # Use select to avoid blocking
            ready, _, _ = select.select([self.master_fd], [], [], timeout)
            if ready:
                return os.read(self.master_fd, 1024 * 10)  # Read up to 10KB
        except OSError:
            self.running = False

        return b''

    def close(self):
        """Close the PTY and terminate the shell."""
        self.running = False
        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except OSError:
                pass
        if self.pid is not None:
            try:
                os.kill(self.pid, 9)  # SIGKILL
                os.waitpid(self.pid, 0)
            except (OSError, ChildProcessError):
                pass


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "Nerion Mission Control API"}


@app.get("/api/health")
async def health():
    """Get system health status."""
    # TODO: Query actual Nerion health
    return {
        "status": "healthy",
        "build_health": 98,
        "uptime_seconds": 3888000,
        "components": {
            "voice": "online",
            "network": "online",
            "learning": "active",
            "llm": "claude"
        }
    }


@app.websocket("/api/terminal")
async def terminal_websocket(websocket: WebSocket):
    """WebSocket endpoint for terminal I/O.

    This provides a real bash shell with bidirectional I/O:
    - Client sends keystrokes → PTY
    - PTY output → Client display
    """
    await websocket.accept()

    # Spawn shell
    pty_manager = PTYManager()
    try:
        pty_manager.spawn_shell()
        print(f"[Terminal] Spawned shell with PID {pty_manager.pid}")

        # Create tasks for bidirectional communication
        async def read_from_pty():
            """Read from PTY and send to WebSocket."""
            while pty_manager.running:
                data = await asyncio.get_event_loop().run_in_executor(
                    None, pty_manager.read, 0.05
                )
                if data:
                    await websocket.send_bytes(data)
                await asyncio.sleep(0.01)  # Small delay to prevent busy loop

        async def read_from_websocket():
            """Read from WebSocket and write to PTY."""
            try:
                while True:
                    message = await websocket.receive()

                    if 'bytes' in message:
                        # Terminal input (keystrokes)
                        data = message['bytes']
                        await asyncio.get_event_loop().run_in_executor(
                            None, pty_manager.write, data
                        )
                    elif 'text' in message:
                        # Handle resize events (JSON)
                        try:
                            event = json.loads(message['text'])
                            if event.get('type') == 'resize':
                                cols = event.get('cols', 80)
                                rows = event.get('rows', 24)
                                pty_manager.resize(cols, rows)
                        except json.JSONDecodeError:
                            pass
            except WebSocketDisconnect:
                print("[Terminal] WebSocket disconnected")

        # Run both tasks concurrently
        await asyncio.gather(
            read_from_pty(),
            read_from_websocket(),
        )

    except Exception as e:
        print(f"[Terminal] Error: {e}")
    finally:
        pty_manager.close()
        print("[Terminal] PTY closed")


@app.websocket("/api/events")
async def events_websocket(websocket: WebSocket):
    """WebSocket endpoint for system events.

    This streams structured events for dashboard panel updates:
    - health_update
    - signal_update
    - autonomous_action
    - memory_update
    - artifact_created
    - upgrade_ready
    """
    await websocket.accept()

    try:
        # Send initial state
        await websocket.send_json({
            "type": "health_update",
            "data": {
                "build_health": 98,
                "active_threats": 2,
                "auto_fixes_24h": 23
            }
        })

        await websocket.send_json({
            "type": "signal_update",
            "data": {
                "voice": "online",
                "network": "online",
                "learning": "active",
                "llm": "claude"
            }
        })

        # Keep connection alive and send periodic updates
        while True:
            await asyncio.sleep(5)

            # TODO: Send real events from Nerion
            # For now, just keep alive
            await websocket.send_json({
                "type": "heartbeat",
                "data": {"timestamp": asyncio.get_event_loop().time()}
            })

    except WebSocketDisconnect:
        print("[Events] WebSocket disconnected")


@app.get("/api/memory")
async def get_memory():
    """Get memory entries."""
    # TODO: Query actual Nerion memory
    return {
        "count": 234,
        "pinned": [
            {"fact": "Code style: snake_case", "score": 3.5},
            {"fact": "Database: PostgreSQL", "score": 3.2},
            {"fact": "Testing: pytest", "score": 3.0}
        ],
        "recent": [
            {"fact": "Uses Docker", "score": 2.1},
            {"fact": "Prefers async/await", "score": 1.8}
        ]
    }


@app.get("/api/artifacts")
async def get_artifacts():
    """Get list of artifacts."""
    # TODO: Query actual artifacts from /out/artifacts/
    return {
        "artifacts": [
            {"name": "security_audit.md", "type": "security", "size": 4523},
            {"name": "refactor_plan.json", "type": "plan", "size": 12043},
            {"name": "bug_analysis.json", "type": "analysis", "size": 8721}
        ]
    }


@app.get("/api/learning/timeline")
async def get_learning_timeline():
    """Get learning timeline events."""
    # TODO: Query actual learning data from /out/learning/
    return {
        "events": [
            {"time": "14:23", "event": "Learned: Prefer pytest over unittest"},
            {"time": "12:45", "event": "Tool adjustment: Use ruff for linting"},
            {"time": "09:12", "event": "Style preference: Concise responses"}
        ]
    }


@app.get("/api/upgrades/pending")
async def get_pending_upgrades():
    """Get pending upgrade proposals."""
    # TODO: Query actual upgrade lane
    return {
        "pending": [
            {
                "id": "upgrade_001",
                "title": "Add type hints to utils module",
                "impact": "medium",
                "risk": "low",
                "description": "Enhance code quality with comprehensive type hints"
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn

    print("""
    ╔════════════════════════════════════════════════════╗
    ║  🧬 NERION MISSION CONTROL - TERMINAL SERVER      ║
    ╠════════════════════════════════════════════════════╣
    ║  WebSocket Terminal:  ws://localhost:8000/api/terminal  ║
    ║  WebSocket Events:    ws://localhost:8000/api/events    ║
    ║  REST API:            http://localhost:8000/api/        ║
    ╚════════════════════════════════════════════════════╝
    """)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
