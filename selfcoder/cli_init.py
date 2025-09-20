import pathlib
import sys

POLICY_TEMPLATE = """\
policy:
  deny_paths:
    - "plugins/**"
    - ".git/**"
  allow_paths:
    - "app/**"
    - "selfcoder/**"
    - "core/**"
  allow_actions:
    - add_module_docstring
    - add_function_docstring
    - insert_function
    - insert_class
    - replace_node
    - apply_unified_diff
  limits:
    max_file_bytes: 200000
    max_total_bytes: 500000
  secrets_block: true
  pii_block: true
"""

SETTINGS_TEMPLATE = """\
voice:
  backend: ${NERION_TTS_BACKEND:-pyttsx3}
stt:
  backend: ${NERION_STT_BACKEND:-whisper}
profile: balanced
"""

def write_if_missing(path: pathlib.Path, content: str):
    if path.exists():
        print(f"skip (exists): {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"wrote: {path}")

def main(argv=None):
    root = pathlib.Path.cwd()
    write_if_missing(root/".nerion/policy.yaml", POLICY_TEMPLATE)
    write_if_missing(root/"app/settings.yaml", SETTINGS_TEMPLATE)
    print("NERION: init complete.\nNext: `nerion doctor`")
    return 0

if __name__ == "__main__":
    sys.exit(main())
