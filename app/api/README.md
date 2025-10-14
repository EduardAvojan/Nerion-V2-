# Nerion Mission Control API

Backend server for the Nerion Mission Control web-based cockpit.

## Features

- **Real Terminal**: WebSocket-based PTY with full bash shell
- **Event Streaming**: Real-time system events for dashboard updates
- **REST API**: System health, memory, artifacts, learning data

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/ed/Nerion-V2/app/api
pip install -r requirements.txt
```

### 2. Run the Server

```bash
python terminal_server.py
```

The server will start on `http://localhost:8000`

### 3. Test the Terminal

Open `test_terminal.html` in your browser:

```bash
open test_terminal.html
```

You should see a fully functional bash terminal where you can:
- Run any bash commands (`ls`, `cd`, `git`, etc.)
- Execute `nerion` CLI commands (if properly configured)
- Use terminal features (history, tab completion, etc.)

## API Endpoints

### REST Endpoints

- `GET /` - Health check
- `GET /api/health` - System health status
- `GET /api/memory` - Memory entries
- `GET /api/artifacts` - List of artifacts
- `GET /api/learning/timeline` - Learning events
- `GET /api/upgrades/pending` - Pending upgrades

### WebSocket Endpoints

#### `/api/terminal`
Real-time terminal I/O with PTY.

**Client → Server:**
- Binary: Keystrokes (UTF-8 encoded)
- JSON: `{"type": "resize", "cols": 80, "rows": 24}`

**Server → Client:**
- Binary: Terminal output (raw bytes with ANSI codes)

#### `/api/events`
Real-time system events for dashboard updates.

**Server → Client (JSON):**
```json
{
  "type": "health_update",
  "data": {"build_health": 98, "active_threats": 2}
}
```

Event types:
- `health_update` - System health metrics
- `signal_update` - Component status
- `autonomous_action` - Auto-fix events
- `memory_update` - Memory changes
- `artifact_created` - New artifacts
- `upgrade_ready` - Upgrade proposals
- `heartbeat` - Keep-alive ping

## Architecture

```
Web Browser (xterm.js)
         ↓ WebSocket
FastAPI Backend (terminal_server.py)
         ↓
PTY (pseudo-terminal)
         ↓
Bash Shell with Nerion CLI
```

## Development

### Add New REST Endpoint

```python
@app.get("/api/your-endpoint")
async def your_endpoint():
    return {"data": "value"}
```

### Send Event via WebSocket

In the `/api/events` handler:

```python
await websocket.send_json({
    "type": "your_event",
    "data": {"key": "value"}
})
```

## Next Steps

1. ✅ Terminal server with PTY
2. ✅ Test client with xterm.js
3. ⏳ Parse terminal output for events
4. ⏳ Build React frontend
5. ⏳ Add dashboard panels
6. ⏳ Integrate chat mode

## Troubleshooting

### "Connection refused" error
- Make sure the server is running: `python terminal_server.py`
- Check the WebSocket URL matches (default: `ws://localhost:8000/api/terminal`)

### Terminal not responding
- Check browser console for errors
- Verify WebSocket connection status (should show "Connected ✓")

### Nerion commands not found
- Ensure Nerion is in your PATH
- Check the `PYTHONPATH` environment variable is set correctly

## References

- [Mission Control Design Doc](../../docs/MISSION_CONTROL_DESIGN.md)
- [xterm.js Documentation](https://xtermjs.org/)
- [FastAPI WebSocket Guide](https://fastapi.tiangolo.com/advanced/websockets/)
