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
import re
import select
import struct
import termios
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Set, Dict, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from output_parser import OutputParser
from json_logging import (
    log_request,
    log_auth_failure,
    log_rate_limit,
    log_injection_attempt,
    log_terminal_event,
    logger,
)
from injection_detector import detect_injection
import socket

# Daemon Socket Path
DAEMON_SOCKET_PATH = Path.home() / ".nerion" / "daemon.sock"

# Connected WebSocket clients for event broadcasting
event_clients: Set[WebSocket] = set()

async def send_daemon_command(command: dict) -> dict:
    """Send a command to the Nerion Daemon via Unix socket."""
    if not DAEMON_SOCKET_PATH.exists():
        return {"status": "error", "message": "Daemon not running (socket not found)"}
        
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(DAEMON_SOCKET_PATH))
        
        # Send command
        message = json.dumps(command).encode('utf-8')
        client.sendall(message)
        
        # Wait for response (optional, daemon might just ack)
        # For now we just fire and forget or wait for a small ack if implemented
        # But the daemon protocol we saw earlier writes back to the writer
        
        # Let's try to read a response
        client.settimeout(2.0)
        response = client.recv(4096)
        client.close()
        
        if response:
            return json.loads(response.decode('utf-8'))
        return {"status": "sent"}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}


app = FastAPI(title="Nerion Mission Control API", version="1.0.0")

# Global state for event broadcasting
event_clients: Set[WebSocket] = set()
output_parser = OutputParser()

# API Key authentication (set NERION_API_KEY in .env for production)
NERION_API_KEY = os.getenv("NERION_API_KEY", "nerion-dev-key-local-only")


# ============================================================================
# RATE LIMITING (Token Bucket Algorithm)
# ============================================================================
class RateLimiter:
    """Token bucket rate limiter for per-IP request throttling."""

    def __init__(self, requests_per_minute: int = 60):
        """Initialize rate limiter.

        Args:
            requests_per_minute: Allow N requests per minute per IP
        """
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, Tuple[int, float]] = defaultdict(lambda: (0, time.time()))

    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed for this IP.

        Args:
            client_ip: Client IP address

        Returns:
            True if request is allowed, False if rate limit exceeded
        """
        now = time.time()
        requests_count, last_reset = self.requests[client_ip]

        # Reset bucket every 60 seconds
        if now - last_reset > 60:
            self.requests[client_ip] = (0, now)
            return True

        # Check if within limit
        if requests_count < self.requests_per_minute:
            self.requests[client_ip] = (requests_count + 1, last_reset)
            return True

        return False


# Initialize rate limiters for different endpoints
rest_api_limiter = RateLimiter(requests_per_minute=120)  # REST API: 120/min
websocket_limiter = RateLimiter(requests_per_minute=10)  # WebSocket: 10/min (strict)


def get_client_ip(request) -> str:
    """Extract client IP from request, handling proxies."""
    # Check X-Forwarded-For header first (for proxied requests)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    # Fall back to direct connection IP
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def injection_detection_middleware(request: Request, call_next):
    """Detect and block injection attacks."""
    client_ip = get_client_ip(request)
    endpoint = request.url.path

    # Skip health check endpoints
    if endpoint in ["/health", "/docs", "/openapi.json"]:
        return await call_next(request)

    # Check query parameters
    query_string = request.url.query
    if query_string and detect_injection(query_string, client_ip, endpoint):
        return JSONResponse(
            status_code=400,
            content={"error": "Suspicious input detected. Request blocked."},
        )

    # For POST/PUT requests, check body
    if request.method in ["POST", "PUT"]:
        try:
            body = await request.body()
            body_str = body.decode("utf-8") if body else ""

            if body_str and detect_injection(body_str, client_ip, endpoint):
                return JSONResponse(
                    status_code=400,
                    content={"error": "Suspicious input detected. Request blocked."},
                )

            # Create a new request with the same body for downstream processing
            async def receive():
                return {"type": "http.request", "body": body}

            request._receive = receive
        except Exception as e:
            logger.warning(f"Error checking request body: {e}")

    # Continue to next middleware
    response = await call_next(request)
    return response


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Centralized logging middleware with request timing."""
    start_time = time.time()
    client_ip = get_client_ip(request)

    # Get request details
    method = request.method
    endpoint = request.url.path
    user_agent = request.headers.get("user-agent")

    try:
        response = await call_next(request)
        response_time_ms = (time.time() - start_time) * 1000

        # Log the request
        log_request(
            client_ip=client_ip,
            method=method,
            endpoint=endpoint,
            status_code=response.status_code,
            response_time_ms=response_time_ms,
            user_agent=user_agent,
            auth_status="authenticated" if "Authorization" in request.headers else "unauthenticated",
        )

        # Add logging headers
        response.headers["X-Response-Time"] = f"{response_time_ms:.2f}"
        return response
    except Exception as e:
        response_time_ms = (time.time() - start_time) * 1000
        logger.error(f"Request failed: {method} {endpoint}", exc_info=e)
        raise


@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    """Rate limit REST API endpoints."""
    client_ip = get_client_ip(request)

    if not rest_api_limiter.is_allowed(client_ip):
        log_rate_limit(
            client_ip=client_ip,
            endpoint=request.url.path,
            limit=120,
            period="minute",
        )
        return JSONResponse(
            status_code=429,
            content={"error": "Too many requests. Rate limit: 120/minute"},
            headers={"Retry-After": "60"},
        )

    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(rest_api_limiter.requests_per_minute)
    response.headers["X-RateLimit-Remaining"] = str(
        rest_api_limiter.requests_per_minute
        - rest_api_limiter.requests.get(client_ip, (0, 0))[0]
    )
    return response

def verify_api_key(websocket: WebSocket) -> bool:
    """Verify API key from WebSocket headers or query parameters.

    Args:
        websocket: WebSocket connection

    Returns:
        True if valid API key provided, False otherwise
    """
    # Check Authorization header (Bearer token)
    auth_header = websocket.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.replace("Bearer ", "")
        return token == NERION_API_KEY

    # Check query parameter (for WebSocket connections)
    token = websocket.query_params.get("token")
    if token:
        return token == NERION_API_KEY

    return False

# Security headers middleware - Add before CORS to ensure they're applied
@app.middleware("http")
async def add_security_headers(request, call_next):
    """Add security headers to all HTTP responses."""
    response = await call_next(request)

    # Prevent MIME type sniffing (ensure browser respects Content-Type)
    response.headers["X-Content-Type-Options"] = "nosniff"

    # Prevent clickjacking (deny framing in any context)
    response.headers["X-Frame-Options"] = "DENY"

    # XSS protection (enable browser's built-in XSS filter)
    response.headers["X-XSS-Protection"] = "1; mode=block"

    # Content Security Policy - strict policy for local development
    # Allows only same-origin resources and inline scripts (necessary for WebSocket)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self'; "
        "connect-src 'self' ws://localhost:* wss://localhost:*; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )

    # HSTS - enforce HTTPS (commented out for localhost development, enable in production)
    # response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    # Referrer policy - don't leak referrer information
    response.headers["Referrer-Policy"] = "no-referrer"

    # Permissions policy - restrict browser features
    response.headers["Permissions-Policy"] = (
        "geolocation=(), "
        "microphone=(), "
        "camera=(), "
        "payment=(), "
        "usb=(), "
        "magnetometer=(), "
        "gyroscope=(), "
        "accelerometer=()"
    )

    return response


# Enable CORS for web frontend
# CORS restricted to localhost only (server binds to 127.0.0.1:8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Electron app origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text for parsing.

    Args:
        text: Text with ANSI codes

    Returns:
        Clean text without ANSI codes
    """
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


async def broadcast_event(event: dict):
    """Broadcast event to all connected event clients.

    Args:
        event: Event dictionary to broadcast
    """
    dead_clients = set()

    for client in event_clients:
        try:
            await client.send_json(event)
        except Exception:
            dead_clients.add(client)

    # Remove dead connections
    event_clients.difference_update(dead_clients)


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

    BULLETPROOF FEATURES:
    - Automatic keepalive pings
    - Robust error handling with recovery
    - Buffer overflow protection
    - Graceful degradation
    - API key authentication
    """
    # Authenticate before accepting connection
    if not verify_api_key(websocket):
        client_ip = get_client_ip(websocket) if hasattr(websocket, 'client') and websocket.client else "unknown"
        log_auth_failure(
            client_ip=client_ip,
            endpoint="/api/terminal",
            reason="Invalid or missing API key",
            user_agent=websocket.headers.get("user-agent"),
        )
        await websocket.close(code=4001, reason="Unauthorized: Invalid or missing API key")
        return

    # Rate limit WebSocket connections (strict: 10 connections per minute per IP)
    client_ip = get_client_ip(websocket) if hasattr(websocket, 'client') and websocket.client else "unknown"
    if not websocket_limiter.is_allowed(client_ip):
        log_rate_limit(
            client_ip=client_ip,
            endpoint="/api/terminal",
            limit=10,
            period="minute",
        )
        await websocket.close(code=4029, reason="Rate limit exceeded: 10 connections/minute")
        return

    await websocket.accept()
    log_terminal_event(
        event_type="client_connected",
        client_ip=client_ip,
    )

    # Spawn shell
    pty_manager = PTYManager()
    pty_task = None
    ws_task = None
    keepalive_task = None

    # Connection state
    connection_alive = True
    last_activity = asyncio.get_event_loop().time()

    try:
        pty_manager.spawn_shell()
        log_terminal_event(
            event_type="shell_spawned",
            pid=pty_manager.pid,
            client_ip=client_ip,
        )

        # Keepalive task to prevent WebSocket timeout
        async def keepalive():
            """Send periodic pings to keep connection alive."""
            try:
                while connection_alive:
                    await asyncio.sleep(30)  # Ping every 30 seconds
                    try:
                        # Send a ping frame
                        await websocket.send_json({
                            "type": "ping",
                            "timestamp": asyncio.get_event_loop().time()
                        })
                        print("[Terminal] Sent keepalive ping")
                    except Exception as e:
                        print(f"[Terminal] Keepalive failed: {e}")
                        break
            except asyncio.CancelledError:
                print("[Terminal] Keepalive task cancelled")
                raise

        # Create tasks for bidirectional communication
        async def read_from_pty():
            """Read from PTY and send to WebSocket with robust error handling."""
            consecutive_errors = 0
            max_consecutive_errors = 10

            try:
                while pty_manager.running and connection_alive:
                    try:
                        # Read with timeout
                        data = await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(
                                None, pty_manager.read, 0.1
                            ),
                            timeout=2.0  # 2 second timeout
                        )

                        if data:
                            # Reset error counter on successful read
                            consecutive_errors = 0

                            # Send raw bytes to terminal display
                            try:
                                await websocket.send_bytes(data)
                                nonlocal last_activity
                                last_activity = asyncio.get_event_loop().time()
                            except Exception as send_error:
                                print(f"[Terminal] Failed to send data to WebSocket: {send_error}")
                                break

                            # Parse output for dashboard events (non-critical)
                            try:
                                text = data.decode('utf-8', errors='ignore')
                                clean_text = strip_ansi(text)
                                events = output_parser.parse_buffer(clean_text)

                                # Broadcast events to all connected event clients
                                for event in events:
                                    await broadcast_event(event)
                            except Exception as parse_error:
                                # Don't let parsing errors kill the terminal
                                print(f"[Terminal] Error parsing output (non-critical): {parse_error}")

                        await asyncio.sleep(0.02)  # Small delay to prevent busy loop

                    except asyncio.TimeoutError:
                        # Timeout is normal - just continue
                        consecutive_errors = 0
                        continue
                    except Exception as read_error:
                        consecutive_errors += 1
                        print(f"[Terminal] PTY read error ({consecutive_errors}/{max_consecutive_errors}): {read_error}")

                        if consecutive_errors >= max_consecutive_errors:
                            print("[Terminal] Too many consecutive errors, stopping PTY read")
                            break

                        # Brief pause before retry
                        await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                print("[Terminal] PTY read task cancelled")
                raise
            except Exception as e:
                print(f"[Terminal] Fatal PTY read error: {e}")
                import traceback
                traceback.print_exc()

        async def read_from_websocket():
            """Read from WebSocket and write to PTY with robust error handling."""
            consecutive_errors = 0
            max_consecutive_errors = 10

            try:
                while connection_alive:
                    try:
                        # Receive with timeout to detect dead connections
                        message = await asyncio.wait_for(
                            websocket.receive(),
                            timeout=120.0  # 2 minute timeout
                        )

                        # Reset error counter on successful receive
                        consecutive_errors = 0
                        nonlocal last_activity
                        last_activity = asyncio.get_event_loop().time()

                        # Check for disconnect
                        if message.get('type') == 'websocket.disconnect':
                            print("[Terminal] WebSocket disconnect message received")
                            break

                        if 'bytes' in message:
                            # Terminal input (keystrokes)
                            data = message['bytes']
                            try:
                                await asyncio.get_event_loop().run_in_executor(
                                    None, pty_manager.write, data
                                )
                            except Exception as write_error:
                                print(f"[Terminal] Failed to write to PTY: {write_error}")
                                # Don't break - PTY might recover

                        elif 'text' in message:
                            # Check for injection attacks in text messages
                            if detect_injection(message['text'], client_ip, "/api/terminal"):
                                print(f"[Terminal] Injection attempt detected from {client_ip}")
                                await websocket.send_json({
                                    "type": "error",
                                    "message": "Suspicious input detected. Request blocked."
                                })
                                continue

                            # Handle resize events and pong responses (JSON)
                            try:
                                event = json.loads(message['text'])
                                if event.get('type') == 'resize':
                                    cols = event.get('cols', 80)
                                    rows = event.get('rows', 24)
                                    pty_manager.resize(cols, rows)
                                    print(f"[Terminal] Resized to {cols}x{rows}")
                                elif event.get('type') == 'pong':
                                    # Client responded to ping
                                    pass
                            except json.JSONDecodeError as json_error:
                                # Ignore invalid JSON
                                print(f"[Terminal] Invalid JSON (ignored): {json_error}")

                    except asyncio.TimeoutError:
                        print("[Terminal] WebSocket receive timeout - connection may be dead")
                        break
                    except WebSocketDisconnect:
                        print("[Terminal] WebSocket disconnected normally")
                        break
                    except Exception as recv_error:
                        consecutive_errors += 1
                        print(f"[Terminal] WebSocket receive error ({consecutive_errors}/{max_consecutive_errors}): {recv_error}")

                        if consecutive_errors >= max_consecutive_errors:
                            print("[Terminal] Too many consecutive errors, stopping WebSocket read")
                            break

                        await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                print("[Terminal] WebSocket read task cancelled")
                raise
            except Exception as e:
                print(f"[Terminal] Fatal WebSocket read error: {e}")
                import traceback
                traceback.print_exc()

        # Create tasks
        pty_task = asyncio.create_task(read_from_pty())
        ws_task = asyncio.create_task(read_from_websocket())
        keepalive_task = asyncio.create_task(keepalive())

        # Wait for any task to complete
        done, pending = await asyncio.wait(
            [pty_task, ws_task, keepalive_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        print(f"[Terminal] Task completed: {[t.get_name() for t in done]}")

        # Signal shutdown
        connection_alive = False

        # Cancel remaining tasks gracefully
        for task in pending:
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception as cancel_error:
                print(f"[Terminal] Error cancelling task: {cancel_error}")

    except Exception as e:
        print(f"[Terminal] Top-level error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure connection is marked as dead
        connection_alive = False

        # Cleanup tasks
        for task in [pty_task, ws_task, keepalive_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=1.0)
                except Exception:
                    pass

        # Close PTY
        pty_manager.close()
        print(f"[Terminal] Connection closed. Session duration: {asyncio.get_event_loop().time() - last_activity:.1f}s since last activity")


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

    # Register this client for event broadcasting
    event_clients.add(websocket)
    print(f"[Events] Client connected. Total clients: {len(event_clients)}")

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

            # Send heartbeat to keep connection alive
            await websocket.send_json({
                "type": "heartbeat",
                "data": {"timestamp": asyncio.get_event_loop().time()}
            })

    except WebSocketDisconnect:
        print("[Events] WebSocket disconnected")
    finally:
        # Unregister client on disconnect
        event_clients.discard(websocket)
        print(f"[Events] Client disconnected. Total clients: {len(event_clients)}")


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

@app.post("/api/control/start_autonomy")
async def start_autonomy():
    """Start the autonomous testing loop."""
    logger.info("Requesting start of autonomous testing")
    result = await send_daemon_command({"type": "start_autonomous_testing"})
    return result

@app.post("/api/control/stop_autonomy")
async def stop_autonomy():
    """Stop the autonomous testing loop."""
    logger.info("Requesting stop of autonomous testing")
    result = await send_daemon_command({"type": "stop_autonomous_testing"})
    return result

@app.get("/api/training/status")
async def get_training_status():
    """Get real training status from daemon."""
    # We can ask the daemon for status
    result = await send_daemon_command({"type": "get_status"})
    if result.get("status") == "error":
        # Fallback to mock if daemon unreachable
        return {"status": "offline", "message": result.get("message")}
    return result


async def log_debug(msg):
    with open("app/api/debug_events.log", "a") as f:
        f.write(f"{time.ctime()}: {msg}\n")

async def listen_to_daemon_events():
    """Background task to listen for events from the daemon and broadcast to clients."""
    await log_debug("Starting daemon listener task...")
    while True:
        try:
            await log_debug(f"Attempting to connect to {DAEMON_SOCKET_PATH}...")
            reader, writer = await asyncio.open_unix_connection(DAEMON_SOCKET_PATH)
            await log_debug("Connected to daemon event stream!")
            
            while True:
                try:
                    line = await reader.readline()
                    if not line:
                        await log_debug("Daemon sent EOF (connection closed by daemon)")
                        break
                    
                    # await log_debug(f"Received raw: {line[:50]}...") 
                    
                    message = json.loads(line.decode())
                    # Broadcast to all connected WebSocket clients
                    if event_clients:
                        await log_debug(f"Broadcasting to {len(event_clients)} clients")
                        # Create tasks for all sends to avoid blocking
                        await asyncio.gather(
                            *[client.send_json(message) for client in event_clients],
                            return_exceptions=True
                        )
                except json.JSONDecodeError as e:
                    await log_debug(f"JSON decode error: {e}")
                    continue
                except Exception as e:
                    await log_debug(f"Error processing message: {e}")
                    import traceback
                    await log_debug(traceback.format_exc())
                    
            await log_debug("Daemon connection loop ended, closing writer...")
            writer.close()
            await writer.wait_closed()
            
        except (FileNotFoundError, ConnectionRefusedError):
            await log_debug("Daemon socket not found or refused. Retrying in 5s...")
            await asyncio.sleep(5)
        except Exception as e:
            await log_debug(f"Daemon listener CRASH: {e}")
            import traceback
            await log_debug(traceback.format_exc())
            await asyncio.sleep(5)

# Global reference to prevent GC
daemon_listener_task = None

@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup."""
    global daemon_listener_task
    # Important: Keep a strong reference to the task to prevent garbage collection
    daemon_listener_task = asyncio.create_task(listen_to_daemon_events())

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global daemon_listener_task
    if daemon_listener_task:
        daemon_listener_task.cancel()
        try:
            await daemon_listener_task
        except asyncio.CancelledError:
            pass

if __name__ == "__main__":
    import uvicorn
    print("""
    ╔════════════════════════════════════════════════════╗
    ║  🧬 NERION MISSION CONTROL - TERMINAL SERVER      ║
    ╠════════════════════════════════════════════════════╣
    ║  WebSocket Terminal:  ws://localhost:8000/api/terminal  ║
    ║  WebSocket Events:    ws://localhost:8000/api/events    ║
    ║  REST API:            http://localhost:8000/api/        ║
    ║  Protocol:            HTTP (Local Development)          ║
    ╚════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )
