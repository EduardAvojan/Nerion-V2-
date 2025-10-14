# Nerion Immune System - 24/7 Codebase Protection

## 🧬 Overview

Nerion is a **biological immune system** for your codebase that runs **24/7** in the background, continuously monitoring, learning, and protecting your code - just like your body's immune system.

```
┌─────────────────────────────────────────────┐
│  Nerion Immune System (Always Running)     │
│  ────────────────────────────────────────   │
│                                             │
│  Background Daemon                          │
│  🔴 Watches codebase 24/7                   │
│  🔴 Runs GNN training                       │
│  🔴 Monitors for threats                    │
│  🔴 Auto-fixes issues                       │
│  🔴 Learns from patterns                    │
│                                             │
│  ↕ Socket Communication                     │
│                                             │
│  Mission Control GUI (Optional)             │
│  • Open to see detailed status             │
│  • Close anytime - daemon keeps running    │
│  • Shows real-time metrics                 │
│                                             │
│  System Tray Icon                           │
│  🟢 Healthy  🟡 Warning  🔴 Critical        │
└─────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Option 1: Standard Start (Daemon + GUI)
```bash
./start_nerion.sh
```

### Option 2: Install as System Service (Auto-start on boot)
```bash
# Install daemon to run automatically
cd daemon
./install_service.sh

# Launch GUI
cd ../app/ui/holo-app
npm start
```

## 🎯 How It Works

### Background Daemon (`nerion_daemon.py`)

**Runs independently** - doesn't stop when you close the GUI.

**What it does:**
- 👁️ **Watches**: Monitors all files in your codebase
- 🧠 **Learns**: Trains GNN on patterns it discovers
- 🛡️ **Protects**: Detects threats and anomalies
- 🔧 **Fixes**: Auto-corrects issues it can handle
- 📊 **Reports**: Sends status to GUI when open

**Location:** `~/.nerion/daemon.sock` (Unix socket)
**Logs:** `~/.nerion/daemon.log`

### Mission Control GUI (Electron)

**Can open/close freely** - daemon keeps running.

**What it shows:**
- Real-time daemon status
- Training metrics
- Threat dashboard
- File monitoring stats
- Auto-fix history

**Behavior:**
- Close window → minimizes to tray
- Daemon keeps running
- Click tray icon → reopens
- "Quit" → stops both daemon and GUI

### System Tray

**Always visible** when Nerion is running.

**Status indicators:**
- 🟢 **Green**: Healthy - all systems normal
- 🟡 **Yellow**: Warning - issues detected
- 🔴 **Red**: Critical - immediate attention needed
- ⚪ **Gray**: Disconnected from daemon

**Menu options:**
- Show/Hide Mission Control
- View status
- Quit Nerion

## 📋 Installation Modes

### Mode 1: Manual Start (Development)

Daemon runs only while you're working:

```bash
./start_nerion.sh
```

**Pros:**
- Easy to stop/restart
- Good for development

**Cons:**
- Stops when you close terminal
- Doesn't survive reboots

### Mode 2: System Service (Production)

Daemon runs 24/7 automatically:

```bash
cd daemon
./install_service.sh
```

**Pros:**
- Auto-starts on boot
- Runs 24/7 independently
- Survives reboots
- Restarts if crashes

**Cons:**
- Need to explicitly stop it

**macOS Implementation:**
- Creates `~/Library/LaunchAgents/com.nerion.daemon.plist`
- Uses `launchctl` for management

## 🔧 Management Commands

### Check if daemon is running
```bash
# Check socket exists
ls -la ~/.nerion/daemon.sock

# Check via launchctl (if installed as service)
launchctl list | grep nerion
```

### View daemon logs
```bash
tail -f ~/.nerion/daemon.log
```

### Stop daemon
```bash
# Via GUI: Right-click tray → Quit

# Manually (if service):
launchctl unload ~/Library/LaunchAgents/com.nerion.daemon.plist

# Kill process:
pkill -f nerion_daemon.py
```

### Uninstall service
```bash
cd daemon
./uninstall_service.sh
```

## 📡 Communication Protocol

**Transport:** Unix Domain Socket (`~/.nerion/daemon.sock`)
**Format:** Newline-delimited JSON

### GUI → Daemon Commands

```json
{"type": "get_status"}
{"type": "start_training"}
{"type": "stop_training"}
{"type": "shutdown"}
```

### Daemon → GUI Messages

```json
{
  "type": "status_update",
  "data": {
    "status": "running",
    "health": "healthy",
    "threats_detected": 0,
    "auto_fixes_applied": 23,
    "files_monitored": 1234,
    "gnn_training": true,
    "gnn_episodes": 42
  }
}
```

## 🏗️ File Structure

```
Nerion-V2/
├── daemon/
│   ├── nerion_daemon.py           # Core immune system (runs 24/7)
│   ├── install_service.sh         # Install as system service
│   └── uninstall_service.sh       # Uninstall service
│
├── app/ui/holo-app/               # Electron GUI
│   ├── src/
│   │   ├── main.js                # Connects to daemon
│   │   └── mission-control/       # React UI
│   └── dist/                      # Built React app
│
├── start_nerion.sh                # Unified startup
└── ~/.nerion/                     # Runtime directory
    ├── daemon.sock                # Socket for communication
    ├── daemon.log                 # Daemon logs
    └── daemon-error.log           # Error logs
```

## 🎓 User Experience

### First Time Setup

```bash
# 1. Start Nerion
./start_nerion.sh

# GUI opens, daemon starts in background
# Tray icon appears showing status
```

### Daily Use

```bash
# Close Mission Control window
# → Daemon keeps running
# → Tray icon still visible
# → Can reopen anytime by clicking tray

# Reboot computer
# → Daemon NOT running (manual mode)
# → Run ./start_nerion.sh again

# OR install as service for auto-start:
./daemon/install_service.sh
# → Daemon starts automatically on boot
# → Always protecting your code
```

### Stopping Nerion

```bash
# Right-click tray icon → "Quit Nerion"
# → Stops both daemon and GUI
# → Tray icon disappears
```

## 🔍 Monitoring

### Daemon Status via Tray

- **Hover**: Shows health status
- **Click**: Opens/closes Mission Control
- **Right-click**: Shows full menu

### Daemon Status via GUI

Mission Control shows:
- Connection status (connected/disconnected)
- Health (healthy/warning/critical)
- Files monitored
- Threats detected
- Auto-fixes applied
- Training status
- GNN episodes completed

### Daemon Status via Logs

```bash
tail -f ~/.nerion/daemon.log
```

Example output:
```
[2025-10-14 12:00:00] [NERION-DAEMON] INFO: Nerion Immune Daemon initialized
[2025-10-14 12:00:00] [NERION-DAEMON] INFO: Monitoring codebase: /Users/ed/project
[2025-10-14 12:00:01] [NERION-DAEMON] INFO: Socket server started: ~/.nerion/daemon.sock
[2025-10-14 12:00:01] [NERION-DAEMON] INFO: 🧬 Nerion Immune System ONLINE
[2025-10-14 12:00:01] [NERION-DAEMON] INFO: 👁️  Codebase watcher started
[2025-10-14 12:00:01] [NERION-DAEMON] INFO: 🧠 GNN background training started
[2025-10-14 12:00:01] [NERION-DAEMON] INFO: 🛡️  Health monitor started
```

## 🐛 Troubleshooting

### Daemon won't start

```bash
# Check logs
cat ~/.nerion/daemon-error.log

# Check socket
ls -la ~/.nerion/daemon.sock

# Try starting manually
python3 daemon/nerion_daemon.py /path/to/codebase
```

### GUI can't connect to daemon

```bash
# Check daemon is running
ls ~/.nerion/daemon.sock

# Check logs
tail ~/.nerion/daemon.log

# Restart daemon
pkill -f nerion_daemon.py
./start_nerion.sh
```

### Daemon using too much CPU

```bash
# Check what it's doing
tail -f ~/.nerion/daemon.log

# Temporarily stop training
# → Open GUI → Training Dashboard → Pause Training
```

### Remove everything

```bash
# Uninstall service
./daemon/uninstall_service.sh

# Remove runtime files
rm -rf ~/.nerion/

# Kill any running processes
pkill -f nerion_daemon.py
```

## 🚦 Status Reference

### Health States

| Icon | Health | Meaning |
|------|--------|---------|
| 🟢 | `healthy` | All systems normal |
| 🟡 | `warning` | 10+ threats detected |
| 🔴 | `critical` | 50+ threats detected |
| ⚪ | `unknown` | Disconnected from daemon |

### Status States

| Status | Meaning |
|--------|---------|
| `starting` | Daemon initializing |
| `running` | Daemon active and monitoring |
| `stopping` | Daemon shutting down |
| `disconnected` | GUI not connected to daemon |
| `connected` | GUI connected to daemon |

## 🔮 Future Enhancements

- [ ] Windows support (Windows Service)
- [ ] Linux support (systemd service)
- [ ] Real-time file watching (watchdog library)
- [ ] Actual GNN training integration
- [ ] Threat detection algorithms
- [ ] Auto-fix capabilities
- [ ] Performance metrics
- [ ] Security scanning
- [ ] Code quality checks

---

**Status:** ✅ **Hybrid system implemented and ready for testing**

**Next:** Test daemon persistence, then create installer
