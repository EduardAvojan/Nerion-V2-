#!/usr/bin/env python3
"""
Nerion Immune System Daemon 🛡️
------------------------------
The "Heartbeat" of the autonomous coding system.
Runs in the background and periodically triggers the Universal Fixer
to evolve the codebase (Quality, Security, Types, Performance).
"""

import os
import sys
import time
import random
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Configuration
DAEMON_LOG = "daemon.log"
MIN_INTERVAL_SECONDS = 3600  # 1 hour
MAX_INTERVAL_SECONDS = 14400 # 4 hours
DO_NOT_DISTURB_FILE = "NERION_DND"

# Evolution Vectors
VECTORS = [
    "evolve_quality",
    "evolve_quality", # Higher weight for quality
    "evolve_types",
    "evolve_security",
    "evolve_perf"
]

# Setup Logging
logging.basicConfig(
    filename=DAEMON_LOG,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NerionDaemon")

def setup_console_logging():
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)

def get_random_target_file(root_dir: str = ".") -> Optional[str]:
    """Pick a random Python file from the project."""
    excluded_dirs = {".git", ".venv", "venv", "env", "__pycache__", "node_modules", "tmp", "temp", "build", "dist"}
    candidates = []
    
    for p in Path(root_dir).rglob("*.py"):
        # Check exclusions
        if any(part in excluded_dirs for part in p.parts):
            continue
        # Skip tests for now (we want to evolve source code)
        if "test" in p.name or "tests" in p.parts:
            continue
        candidates.append(str(p))
    
    if not candidates:
        return None
    return random.choice(candidates)

def run_evolution_cycle():
    """Run a single evolution cycle."""
    if os.path.exists(DO_NOT_DISTURB_FILE):
        logger.info("😴 Do Not Disturb mode active. Skipping cycle.")
        return

    target_file = get_random_target_file()
    if not target_file:
        logger.warning("⚠️  No suitable target files found.")
        return

    vector = random.choice(VECTORS)
    
    logger.info(f"🧬 Starting Evolution Cycle: {vector} on {target_file}")
    
    try:
        # Run Universal Fixer
        cmd = [
            sys.executable,
            "nerion_digital_physicist/universal_fixer.py",
            target_file,
            "--mode", vector
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            logger.info(f"✅ Evolution Successful: {vector} on {target_file}")
            logger.info(f"Output:\n{result.stdout}")
        else:
            logger.error(f"❌ Evolution Failed: {vector} on {target_file}")
            logger.error(f"Error:\n{result.stderr}")
            
    except Exception as e:
        logger.error(f"💥 Daemon Exception: {e}")

def main():
    setup_console_logging()
    logger.info("🛡️  Nerion Immune Daemon Started")
    logger.info(f"   Interval: {MIN_INTERVAL_SECONDS}-{MAX_INTERVAL_SECONDS}s")
    logger.info("   Press Ctrl+C to stop.")
    
    try:
        while True:
            run_evolution_cycle()
            
            # Sleep for random interval
            sleep_time = random.randint(MIN_INTERVAL_SECONDS, MAX_INTERVAL_SECONDS)
            next_run = datetime.fromtimestamp(time.time() + sleep_time)
            logger.info(f"💤 Sleeping for {sleep_time}s. Next run at {next_run.strftime('%H:%M:%S')}")
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        logger.info("🛑 Daemon stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
