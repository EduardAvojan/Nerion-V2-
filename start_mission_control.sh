#!/bin/bash

# Nerion Mission Control Startup Script
# Starts Daemon, API Server, and Frontend in separate tabs

echo "🚀 Starting Nerion Mission Control..."

# Get project root
PROJECT_ROOT="/Users/ed/Nerion-V2"
cd "$PROJECT_ROOT"

# Function to open a new tab and run a command
open_tab() {
    local title="$1"
    local cmd="$2"
    
    # Use a simpler AppleScript command to avoid quoting hell
    osascript <<EOF
    tell application "Terminal"
        do script "cd $PROJECT_ROOT; $cmd"
    end tell
EOF
}

# 1. Start API Server (Terminal Backend)
echo "🔌 Starting API Server..."
open_tab "Nerion API" "python app/api/terminal_server.py"

# 2. Start Frontend (React)
echo "💻 Starting Frontend..."
open_tab "Nerion Frontend" "cd app/web && npm run dev"

echo "✅ API and Frontend launching in new tabs..."
echo "🚀 Starting Daemon in THIS terminal..."
echo "   (Press Ctrl+C to stop the Daemon)"
echo ""
sleep 2

# 3. Start Daemon (in current shell)
python daemon/nerion_daemon.py
