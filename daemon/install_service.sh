#!/bin/bash
# Install Nerion Immune System as a macOS LaunchAgent
# This makes Nerion start automatically on boot and run 24/7

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DAEMON_PATH="$SCRIPT_DIR/nerion_daemon.py"
PLIST_PATH="$HOME/Library/LaunchAgents/com.nerion.daemon.plist"
LOG_DIR="$HOME/.nerion"

echo "🧬 Installing Nerion Immune System Daemon..."
echo ""

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Create LaunchAgent plist
cat > "$PLIST_PATH" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.nerion.daemon</string>

    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>$DAEMON_PATH</string>
        <string>$PROJECT_ROOT</string>
    </array>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
        <key>Crashed</key>
        <true/>
    </dict>

    <key>StandardOutPath</key>
    <string>$LOG_DIR/daemon.log</string>

    <key>StandardErrorPath</key>
    <string>$LOG_DIR/daemon-error.log</string>

    <key>WorkingDirectory</key>
    <string>$PROJECT_ROOT</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
EOF

echo "✅ LaunchAgent plist created: $PLIST_PATH"

# Load the LaunchAgent
launchctl unload "$PLIST_PATH" 2>/dev/null || true
launchctl load "$PLIST_PATH"

echo "✅ LaunchAgent loaded"
echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║  🧬 Nerion Immune System Installed                ║"
echo "╠════════════════════════════════════════════════════╣"
echo "║  The daemon will now:                              ║"
echo "║  • Start automatically on boot                     ║"
echo "║  • Run continuously in background                  ║"
echo "║  • Monitor your codebase 24/7                      ║"
echo "║  • Restart automatically if it crashes             ║"
echo "║                                                    ║"
echo "║  Logs: ~/.nerion/daemon.log                        ║"
echo "║  Status: Check system tray or run:                ║"
echo "║    launchctl list | grep nerion                    ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""
echo "Nerion is now running! Check the system tray icon."
