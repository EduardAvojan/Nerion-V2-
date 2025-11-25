#!/bin/bash
# run_gym.sh
export PYTHONUNBUFFERED=1
nohup python daemon/nerion_daemon.py --gym > gym.log 2>&1 &
echo $! > gym.pid
echo "Gym started with PID $(cat gym.pid)"
