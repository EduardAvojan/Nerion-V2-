#!/bin/zsh
# This script runs one learning cycle for Nerion.
# It is intended to be called by a cron job.

# Source the user's shell configuration to make the 'conda' command available.
# This is crucial for cron environments.
source /Users/ed/.zshrc

# Activate the correct conda environment.
conda activate base

# Define the absolute path to the project root.
PROJECT_ROOT="/Users/ed/Nerion-V2"

# Navigate to the project root to ensure all relative paths work correctly.
cd "$PROJECT_ROOT"

# Explicitly load environment variables from .env file if it exists.
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Run the learning orchestrator module, passing along any arguments given to this script.
python -m nerion_digital_physicist.learning_orchestrator "$@"