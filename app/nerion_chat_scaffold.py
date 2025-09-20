"""Entry point for voice-first Nerion runtime (scaffold).
- Loads settings
- Stubs voice I/O
- Routes simple commands
"""
import os
import sys
import json
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

def load_settings():
    import yaml
    with open(ROOT / 'app' / 'settings.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    try:
        settings = load_settings()
    except Exception as e:
        print('[Nerion] Failed to load settings.yaml:', e, file=sys.stderr)
        sys.exit(1)
    print('[Nerion] Online. Voice-first scaffold running.')
    print('[Nerion] Settings summary:', json.dumps({'always_speak': settings.get('voice', {}).get('always_speak', True), 'wake_words': settings.get('wake_words', [])}, indent=2))
    print("\nType a command (e.g., 'remember X', 'forget X', 'upgrade yourself with ...', 'exit'):")
    while True:
        try:
            line = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n[Nerion] Goodbye.')
            break
        if not line:
            continue
        if line.lower() in {'exit', 'quit', 'goodbye', 'talk later', 'see you'}:
            print('[Nerion] Shutting down. See you.')
            break
        if line.lower().startswith('remember '):
            print('[Nerion] (memory stub) Remembered:', line[9:])
            continue
        if line.lower().startswith('forget '):
            print('[Nerion] (memory stub) Forgot:', line[7:])
            continue
        if line.lower().startswith('nerion, upgrade yourself') or line.lower().startswith('upgrade yourself'):
            print('[Nerion] (self-coder stub) Planning safe AST edits...')
            print('[Nerion] (self-coder stub) Static checks & tests would run here.')
            print('[Nerion] (self-coder stub) No-op in scaffold.')
            continue
        print('[Nerion] You said:', line)
if __name__ == '__main__':
    main()