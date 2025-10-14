#!/bin/bash
# Nerion Unified Startup Script
# Starts the immune system daemon + GUI

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DAEMON_SOCKET="$HOME/.nerion/daemon.sock"

echo "🧬 Starting Nerion..."
echo ""

# Check if daemon is already running
if [ -S "$DAEMON_SOCKET" ]; then
    echo "✅ Daemon already running"
else
    echo "🚀 Starting daemon..."

    # Check if daemon is installed as LaunchAgent
    if [ -f "$HOME/Library/LaunchAgents/com.nerion.daemon.plist" ]; then
        # Start via launchctl
        launchctl load "$HOME/Library/LaunchAgents/com.nerion.daemon.plist" 2>/dev/null || true
        echo "✅ Daemon started via LaunchAgent"
    else
        # Start daemon manually in background
        python3 "$SCRIPT_DIR/daemon/nerion_daemon.py" "$SCRIPT_DIR" &
        echo "✅ Daemon started manually"
        echo "   (To auto-start on boot, run: ./daemon/install_service.sh)"
    fi

    # Wait for socket to be created
    echo "   Waiting for daemon to initialize..."
    for i in {1..10}; do
        if [ -S "$DAEMON_SOCKET" ]; then
            break
        fi
        sleep 1
    done

    if [ ! -S "$DAEMON_SOCKET" ]; then
        echo "⚠️  Warning: Daemon socket not found after 10 seconds"
        echo "   Check logs: tail -f ~/.nerion/daemon.log"
    fi
fi

# Start Electron GUI
echo "🖥️  Launching Mission Control..."
cd "$SCRIPT_DIR/app/ui/holo-app"
npm start &

echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║  ✅ Nerion Running                                 ║"
echo "╠════════════════════════════════════════════════════╣"
echo "║  Daemon: Running in background                     ║"
echo "║  GUI: Opening...                                   ║"
echo "║                                                    ║"
echo "║  • Close GUI window → Nerion keeps running        ║"
echo "║  • Check tray icon for status                     ║"
echo "║  • To stop: Right-click tray → Quit               ║"
echo "║                                                    ║"
echo "║  Logs: ~/.nerion/daemon.log                        ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""
