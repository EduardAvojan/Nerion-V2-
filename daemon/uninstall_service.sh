#!/bin/bash
# Uninstall Nerion Immune System daemon

set -e

PLIST_PATH="$HOME/Library/LaunchAgents/com.nerion.daemon.plist"

echo "🛑 Uninstalling Nerion Immune System Daemon..."
echo ""

if [ -f "$PLIST_PATH" ]; then
    # Unload the LaunchAgent
    launchctl unload "$PLIST_PATH" 2>/dev/null || true
    echo "✅ Daemon stopped"

    # Remove plist
    rm "$PLIST_PATH"
    echo "✅ LaunchAgent removed"
else
    echo "⚠️  LaunchAgent not found (already uninstalled?)"
fi

echo ""
echo "Nerion immune system daemon has been uninstalled."
echo "Logs are still available at: ~/.nerion/"
