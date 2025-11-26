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

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from nerion_digital_physicist.memory.episodic_memory import EpisodicMemory
from nerion_digital_physicist.agent.data import create_graph_data_from_source

# Configuration
DAEMON_LOG = "daemon.log"
MIN_INTERVAL_SECONDS = 3600  # 1 hour
MAX_INTERVAL_SECONDS = 14400 # 4 hours
DO_NOT_DISTURB_FILE = "NERION_DND"

# Evolution Vectors (equal weight for balanced training)
VECTORS = [
    "evolve_quality",
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

# Global Memory Component
MEMORY = None

def setup_memory():
    global MEMORY
    try:
        project_root = Path(__file__).parent.parent
        memory_path = project_root / "data" / "episodic_memory"
        MEMORY = EpisodicMemory(storage_path=memory_path)
        logger.info(f"🧠 Episodic Memory initialized at {memory_path}")
    except Exception as e:
        logger.warning(f"Could not initialize EpisodicMemory: {e}")

def setup_console_logging():
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)

def get_random_target_file(root_dir: str = ".") -> Optional[tuple[str, str]]:
    """Pick a random code file from the project. Returns (file_path, language)."""
    excluded_dirs = {".git", ".venv", "venv", "env", "__pycache__", "node_modules", "tmp", "temp", "build", "dist"}
    
    # Language -> file extensions mapping
    language_exts = {
        "python": ["*.py"],
        "javascript": ["*.js", "*.jsx"],
        "typescript": ["*.ts", "*.tsx"]
    }
    
    candidates = []  # List of (file_path, language) tuples
    
    for lang, patterns in language_exts.items():
        for pattern in patterns:
            for p in Path(root_dir).rglob(pattern):
                # Check exclusions
                if any(part in excluded_dirs for part in p.parts):
                    continue
                # Skip tests
                if "test" in p.name or "tests" in p.parts:
                    continue
                candidates.append((str(p), lang))
    
    if not candidates:
        return None
    return random.choice(candidates)

def run_evolution_cycle(root_dir: str = "."):
    """Run a single evolution cycle."""
    if os.path.exists(DO_NOT_DISTURB_FILE):
        logger.info("😴 Do Not Disturb mode active. Skipping cycle.")
        return

    result = get_random_target_file(root_dir)
    if not result:
        logger.warning("⚠️  No suitable target files found.")
        return
    
    target_file, language = result
    vector = random.choice(VECTORS)
    
    logger.info(f"🧬 Starting Evolution Cycle: {vector} on {target_file} ({language})")
    
    try:
        # Run Universal Fixer with language parameter
        cmd = [
            sys.executable,
            "nerion_digital_physicist/universal_fixer.py",
            target_file,
            "--mode", vector,
            "--language", language
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
            timeout=300  # 5 minute timeout to prevent hanging
        )

        if result.returncode == 0:
            logger.info(f"✅ Evolution Successful: {vector} on {target_file}")
            if result.stdout.strip():
                logger.info(f"Output:\n{result.stdout}")
            if result.stderr.strip():
                logger.info(f"Logs:\n{result.stderr}")
        else:
            logger.error(f"❌ Evolution Failed: {vector} on {target_file}")
            logger.error(f"Error:\n{result.stderr}")

    except subprocess.TimeoutExpired:
        logger.error(f"⏰ Evolution Timeout: {vector} on {target_file} (exceeded 5 min)")
            
    except Exception as e:
        logger.error(f"💥 Daemon Exception: {e}")

def dream_cycle():
    """Run a dreaming cycle (Memory Consolidation)."""
    if not MEMORY: return
    
    logger.info("🌙 Entering Dream Cycle (Memory Consolidation)...")
    try:
        # Consolidate memories into principles
        principles = MEMORY.consolidate_memory()
        if principles:
            logger.info(f"💡 Extracted {len(principles)} new principles from experience.")
            for p in principles:
                logger.info(f"   - {p.description}")
        else:
            logger.info("   No new principles extracted.")
            
    except Exception as e:
        logger.error(f"Nightmare (Dream Cycle Failed): {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Nerion Immune Daemon")
    parser.add_argument("--gym", action="store_true", help="Run in Gym Mode (Training Ground only)")
    args = parser.parse_args()

    setup_console_logging()
    
    if args.gym:
        logger.info("🏋️  GYM MODE ACTIVATED: Training on open source code in training_ground/")
        target_root = "training_ground"
        if not os.path.exists(target_root):
            logger.error(f"❌ Training ground not found at {target_root}")
            sys.exit(1)
        # Faster training in Gym
        global MIN_INTERVAL_SECONDS, MAX_INTERVAL_SECONDS
        MIN_INTERVAL_SECONDS = 10
        MAX_INTERVAL_SECONDS = 30
    else:
        logger.info("🛡️  Nerion Immune Daemon Started")
        target_root = "."

    logger.info(f"   Interval: {MIN_INTERVAL_SECONDS}-{MAX_INTERVAL_SECONDS}s")
    logger.info("   Press Ctrl+C to stop.")
    
    try:
        setup_memory()
        cycle_count = 0
        
        while True:
            # Pass target_root to get_random_target_file (need to update function signature first)
            # But wait, get_random_target_file takes root_dir arg.
            # We need to update run_evolution_cycle to accept root_dir.
            run_evolution_cycle(root_dir=target_root)
            cycle_count += 1
            
            # Dream every 4 cycles (approx every 4-16 hours)
            if cycle_count % 4 == 0:
                dream_cycle()
            
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
