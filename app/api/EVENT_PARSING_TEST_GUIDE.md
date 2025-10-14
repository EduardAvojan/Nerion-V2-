# Event Parsing Test Guide

## Overview

The terminal server now parses terminal output and broadcasts structured events to the dashboard in real-time. This creates a connected experience where terminal commands automatically update dashboard panels.

## How It Works

1. Terminal output is captured from the PTY
2. ANSI escape codes are stripped for clean parsing
3. Output is matched against regex patterns in `output_parser.py`
4. Matching patterns generate structured events
5. Events are broadcast to all connected event WebSocket clients
6. Dashboard panels update in real-time

## Testing Event Parsing

### Setup

1. Open Mission Control: `http://localhost:3000`
2. Open browser console to see event logs (optional)
3. Use the embedded terminal to run test commands

### Test Commands

Here are commands you can type in the terminal that will trigger dashboard events:

#### Health/Signal Updates

```bash
# Test voice status
echo "Voice Stack: Ready"
echo "Voice Stack: Online"
echo "Voice Stack: Offline"

# Test network status
echo "Network Gate: Online"
echo "Network Gate: Active"
echo "Network Gate: Offline"

# Test coverage metric
echo "Coverage: 85%"
echo "Coverage: 95%"

# Test error count
echo "Errors: 5"
```

**Expected**: Signal Health Panel should update showing voice/network status changes.

#### Autonomous Actions

```bash
# Test autonomous fix detection
echo "[AUTONOMOUS] Fixed bug in auth.py"
echo "[AUTONOMOUS] Patched issue in database.py"
echo "[AUTONOMOUS] Repaired error in utils.py"

# Test autonomous deployment
echo "[AUTONOMOUS] Deployed patch to production"
echo "[AUTONOMOUS] Applied fix to staging"

# Test threat detection
echo "[AUTONOMOUS] Detected 3 threats"
```

**Expected**:
- Immune Vitals Panel should increment auto-fixes counter
- Autonomous actions should appear in activity feed

#### Memory Operations

```bash
# Test memory pinning
echo "Pinned: Use TypeScript for new files"
echo "Pinned: Database: PostgreSQL"

# Test learning
echo "Learned: Prefer async/await over callbacks"
echo "Learned: Testing: pytest"

# Test memory count
echo "Memory: 234 entries"
```

**Expected**: Memory Snapshot Panel should update with new entries and counts.

#### Artifacts

```bash
# Test artifact creation
echo "Created artifact: security_audit.md"
echo "Created artifact: refactor_plan.json"
echo "Saved report: bug_analysis.json"
echo "Saved analysis: performance_metrics.md"
```

**Expected**: Artifacts Panel should show new artifacts.

#### Upgrades

```bash
# Test upgrade proposals
echo "Upgrade ready: Add type hints to utils module"
echo "Applied upgrade: Implement caching layer"
```

**Expected**: Upgrade Lane Panel should display new proposal.

#### Learning Events

```bash
# Test learning timeline
echo "Learned: Prefer pytest over unittest"
echo "Tool adjustment: Use ruff for linting"
```

**Expected**: Learning Timeline Panel should show new events.

## Event Types

The parser generates these event types:

- `health_update` - Build health, coverage metrics
- `signal_update` - Component status (voice, network, errors)
- `autonomous_action` - Autonomous fixes, deployments
- `immune_update` - Threats detected, fixes counter
- `memory_update` - Memory entries, pins, learned facts
- `artifact_created` - New artifacts generated
- `upgrade_ready` - Upgrade proposals
- `upgrade_applied` - Applied upgrades
- `learning_event` - Learning timeline entries

## Advanced Testing

### Multi-line Output

The parser handles multi-line output correctly:

```bash
cat <<EOF
Voice Stack: Ready
Network Gate: Online
Coverage: 92%
Memory: 245 entries
EOF
```

**Expected**: Multiple dashboard panels update simultaneously.

### Real Nerion Commands

Once Nerion CLI commands are implemented, they will automatically trigger events:

```bash
nerion status
nerion health
nerion memory --list
nerion autonomous-fixes
```

### Custom Event Patterns

To add new patterns:

1. Edit `/app/api/output_parser.py`
2. Add regex pattern to `self.patterns` dict
3. Add parsing logic in `parse_line()` method
4. Restart backend server
5. Test with matching terminal output

## Debugging

### Enable Event Logging

In browser console, monitor WebSocket events:

```javascript
// Add to browser console
const ws = new WebSocket('ws://localhost:8000/api/events');
ws.onmessage = (event) => {
  console.log('Dashboard event:', JSON.parse(event.data));
};
```

### Check Server Logs

Backend logs show:
- Event client connections/disconnections
- Parsing errors (if any)
- WebSocket activity

```bash
# View backend logs
# Check the terminal where terminal_server.py is running
```

### Verify Pattern Matching

Test patterns directly in Python:

```python
from output_parser import parse_output

# Test a pattern
events = parse_output("Voice Stack: Ready")
print(events)
# Expected: [{'type': 'signal_update', 'data': {'voice': 'online'}}]
```

## Troubleshooting

### Events not appearing in dashboard

1. Check browser console for WebSocket errors
2. Verify backend is running: `http://localhost:8000/`
3. Reload the frontend page
4. Check that events WebSocket is connected

### Pattern not matching

1. Check ANSI codes aren't interfering (they should be stripped)
2. Verify regex pattern in `output_parser.py`
3. Test pattern with `re.search()` directly
4. Check for typos in echo command

### Multiple clients not receiving events

1. Verify `event_clients` set is being updated
2. Check `broadcast_event()` function for errors
3. Monitor server logs for connection messages

## Next Steps

1. **Integrate Real Nerion Commands**: When Nerion CLI is built, commands will naturally trigger events
2. **Add Dashboard Animations**: Visual feedback when events update panels
3. **Event History**: Store recent events for replay
4. **Event Filtering**: Allow users to filter which events update which panels
5. **Custom Event Actions**: Allow panels to trigger actions based on events

## Example Test Session

```bash
# In the Mission Control terminal:

# Update all signals
echo "Voice Stack: Online"
echo "Network Gate: Active"
echo "Coverage: 98%"

# Show autonomous action
echo "[AUTONOMOUS] Fixed critical bug in auth.py"

# Add memory
echo "Learned: Always use type hints"
echo "Pinned: Testing framework: pytest"

# Create artifact
echo "Created artifact: security_report.md"

# Show upgrade
echo "Upgrade ready: Implement Redis caching"
```

**Expected**: All dashboard panels update in real-time as commands execute.

## Architecture

```
Terminal PTY → Output bytes → decode('utf-8')
    ↓
strip_ansi(text) → Clean text
    ↓
output_parser.parse_buffer(text) → List[Event]
    ↓
broadcast_event(event) → All event_clients
    ↓
WebSocket → Frontend React → Dashboard panels update
```

This creates a bidirectional, real-time connection between the terminal and the dashboard, making them feel like parts of a unified system rather than separate interfaces.
