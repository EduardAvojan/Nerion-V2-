#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

# -------- helpers --------
RED=$'\e[31m'; GREEN=$'\e[32m'; YELLOW=$'\e[33m'; RESET=$'\e[0m'
ok()   { echo "✔ $1"; }
bad()  { echo "✖ $1"; }
hdr()  { echo; echo "===== $1 ====="; }

run_check() {
  local title="$1"; shift
  hdr "$title"
  if "$@"; then ok "$title"; else bad "$title"; fi
}

# -------- checks --------

check_legacy_name_scan() {
  # scan for text "Jarvis" ignoring backups, archives, git, venv and this script
  local out
  out="$(grep -R -I -n -i \
      --exclude-dir=backups \
      --exclude-dir=_archive \
      --exclude-dir=.git \
      --exclude-dir=.venv \
      --exclude="*.zip" \
      --exclude="*.tar.gz" \
      --exclude="*.pyc" \
      --exclude="health.sh" \
      -- 'Jarvis' . || true)"
  if [[ -n "$out" ]]; then
    echo "$out"
    return 1
  fi
}

check_legacy_filename_scan() {
  if find . \
      -type f -iname "*jarvis*" \
      -not -path "./backups/*" \
      -not -path "./_archive/*" \
      -not -path "./.git/*" \
      -not -path "./.venv/*" \
      | grep -q . ; then
    return 1
  fi
}

check_env() {
  echo "Python $(python -V 2>&1 | awk '{print $2}')"
  echo "executable: $(command -v python)"
}

check_ruff() {
  if ! command -v ruff >/dev/null 2>&1; then
    echo "${YELLOW}ruff not found in PATH${RESET}"
    return 1
  fi
  ruff check .
}

check_pytest() {
  pytest -q
}

check_tts_import() {
  python - <<'PY'
try:
    import pyttsx3  # noqa
    print("pyttsx3 import OK")
except Exception as e:
    raise SystemExit(f"pyttsx3 import failed: {e}")
PY
}

check_stt_import() {
  python - <<'PY'
try:
    import speech_recognition  # noqa
    print("speech_recognition import OK")
except Exception as e:
    raise SystemExit(f"speech_recognition import failed: {e}")
PY
}

check_memory_rw() {
  python - <<'PY'
import json, os, tempfile
fd, p = tempfile.mkstemp(prefix="nerion_health_", suffix=".json")
os.close(fd)
try:
    with open(p, "w", encoding="utf-8") as f: json.dump({"ok": True}, f)
    with open(p, "r", encoding="utf-8") as f: data = json.load(f)
    assert data.get("ok") is True
    print("memory R/W OK")
finally:
    try: os.remove(p)
    except OSError: pass
PY
}

check_selfcoder_dry_run() {
  # Ensure CLI loads and core subcommands dry-run successfully
  python -m app.nerion_autocoder selfcheck || return 1
  python -m app.nerion_autocoder apply --request "normalize imports" --file app/nerion_chat.py --dry-run || return 1
  python -m app.nerion_autocoder add logging --file app/nerion_chat.py --dry-run || return 1
  python -m app.nerion_autocoder rename \
    --old-module oldmod --old-attr oldfunc \
    --new-module newmod --new-attr newfunc \
    --file app/nerion_chat.py --dry-run || return 1
}

# -------- run all --------
run_check "Legacy name scan (Jarvis → Nerion)"   check_legacy_name_scan
run_check "Legacy filename scan (jarvis → nerion)" check_legacy_filename_scan
run_check "Environment sanity"                   check_env
run_check "Ruff lint"                            check_ruff
run_check "Unit tests (pytest)"                  check_pytest
run_check "TTS library import (pyttsx3)"         check_tts_import
run_check "STT library import (speech_recognition)" check_stt_import
run_check "Memory read/write (temp JSON)"        check_memory_rw
run_check "Self-coder dry-run"                   check_selfcoder_dry_run