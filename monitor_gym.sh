#!/bin/bash
# monitor_gym.sh - Dashboard for Nerion's Training Gym

echo "=================================================="
echo "🏋️  NERION GYM MONITOR"
echo "=================================================="

# 1. Check Process
PID=$(ps aux | grep "nerion_daemon.py --gym" | grep -v grep | awk '{print $2}')
if [ -z "$PID" ]; then
    echo "🔴 Status: STOPPED"
else
    echo "🟢 Status: RUNNING (PID: $PID)"
    # Show runtime
    ps -p $PID -o etime= | awk '{print "   Runtime: " $1}'
fi

echo "--------------------------------------------------"

# 2. Check Model Growth
MODEL_PATH="models/nerion_immune_brain.pt"
if [ -f "$MODEL_PATH" ]; then
    SIZE=$(ls -lh "$MODEL_PATH" | awk '{print $5}')
    TIME=$(ls -lT "$MODEL_PATH" | awk '{print $6, $7, $8}')
    echo "🧠 Brain Model: $SIZE (Last Updated: $TIME)"
else
    echo "🧠 Brain Model: Not created yet"
fi

# 3. Check Memory Growth
MEMORY_PATH="data/episodic_memory/episodes.jsonl"
if [ -f "$MEMORY_PATH" ]; then
    SIZE=$(ls -lh "$MEMORY_PATH" | awk '{print $5}')
    COUNT=$(wc -l < "$MEMORY_PATH")
    echo "📚 Episodic Memory: $SIZE ($COUNT episodes)"
else
    echo "📚 Episodic Memory: Empty"
fi

echo "--------------------------------------------------"
echo "📜 Recent Activity (gym.log):"
echo "--------------------------------------------------"
tail -n 10 gym.log
echo "=================================================="
echo "Press Ctrl+C to exit monitor (Daemon keeps running)"
