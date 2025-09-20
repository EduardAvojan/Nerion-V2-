#!/usr/bin/env bash
set -euo pipefail

if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama CLI not found. Install Ollama first: https://ollama.com" >&2
  exit 1
fi

MODEL="${1:-deepseek-coder-v2}"
echo "Pulling model: ${MODEL}"
ollama pull "${MODEL}"
echo "Done. Set NERION_CODER_MODEL=${MODEL} and restart Nerion."

