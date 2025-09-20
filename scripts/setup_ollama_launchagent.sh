#!/usr/bin/env bash

# Set up or remove a user LaunchAgent that keeps the Ollama daemon running.
# Usage:
#   ./scripts/setup_ollama_launchagent.sh install
#   ./scripts/setup_ollama_launchagent.sh remove

set -euo pipefail

usage() {
  echo "Usage: $0 <install|remove>" >&2
  exit 64
}

if [[ $# -ne 1 ]]; then
  usage
fi

cmd=$1

if [[ $(uname -s) != "Darwin" ]]; then
  echo "LaunchAgents are only supported on macOS." >&2
  exit 65
fi

agent_dir="$HOME/Library/LaunchAgents"
plist_name="io.nerion.ollama.autostart.plist"
plist_path="$agent_dir/$plist_name"

ollama_bin=$(command -v ollama || true)
if [[ -z "$ollama_bin" ]]; then
  echo "Could not find the 'ollama' binary in PATH. Install Ollama first." >&2
  exit 66
fi

ensure_agent_dir() {
  mkdir -p "$agent_dir"
}

write_plist() {
  cat >"$plist_path" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>io.nerion.ollama.autostart</string>
  <key>ProgramArguments</key>
  <array>
    <string>$ollama_bin</string>
    <string>serve</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <dict>
    <key>SuccessfulExit</key>
    <false/>
  </dict>
  <key>StandardErrorPath</key>
  <string>$HOME/Library/Logs/ollama-launchagent.log</string>
  <key>StandardOutPath</key>
  <string>$HOME/Library/Logs/ollama-launchagent.log</string>
</dict>
</plist>
EOF
}

case "$cmd" in
  install)
    ensure_agent_dir
    write_plist
    launchctl unload "$plist_path" >/dev/null 2>&1 || true
    launchctl load -w "$plist_path"
    echo "Installed LaunchAgent: $plist_path"
    echo "Ollama daemon should now start automatically at login."
    ;;
  remove)
    if [[ -f "$plist_path" ]]; then
      launchctl unload "$plist_path" >/dev/null 2>&1 || true
      rm -f "$plist_path"
      echo "Removed LaunchAgent: $plist_path"
    else
      echo "LaunchAgent not found at $plist_path" >&2
    fi
    ;;
  *)
    usage
    ;;
esac
