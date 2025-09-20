from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
import zipfile
from datetime import datetime, timezone
from core.ui.progress import progress
from core.ui.messages import fmt as _fmt_msg
from core.ui.messages import Result as _MsgRes


BREW_FORMULA = """
class Nerion < Formula
  desc "Nerion local AI agent"
  homepage "https://example.com/nerion"
  url "https://example.com/nerion/archive/v0.0.1.tar.gz"
  sha256 "REPLACE_WITH_SHA256"
  license "MIT"

  depends_on "python@3.11"

  def install
    system "pip3", "install", ".", "--prefix=#{prefix}"
    bin.install_symlink Dir["#{prefix}/bin/nerion"]
  end
end
""".strip()

APP_RUNNER = """#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR/.."
export TOKENIZERS_PARALLELISM=false
exec python3 -m app.nerion_chat "$@"
""".strip()


def cmd_scaffold(args: argparse.Namespace) -> int:
    out = Path('out/package')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'brew_formula.rb').write_text(BREW_FORMULA, encoding='utf-8')

    app_dir = out / 'Nerion.app' / 'Contents' / 'MacOS'
    app_dir.mkdir(parents=True, exist_ok=True)
    runner = app_dir / 'nerion'
    runner.write_text(APP_RUNNER, encoding='utf-8')
    try:
        runner.chmod(0o755)
    except Exception:
        pass
    print(f"[package] wrote: {out}")
    print(_fmt_msg('package', 'scaffold', _MsgRes.OK, out.as_posix()))
    print("- brew_formula.rb: edit url/sha256, then `brew create --set-name nerion` or use a tap")
    print("- Nerion.app: minimal runner; wire HOLO UI as needed")
    print("- pipx: `pipx install .` (ensure pyproject scripts expose 'nerion')")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser('package', help='packaging helpers (scaffold and run bundles)')
    sp = p.add_subparsers(dest='package_cmd', required=True)
    sc = sp.add_parser('scaffold', help='write packaging scaffolds to out/package')
    sc.set_defaults(func=cmd_scaffold)

    # ---------------- pack run bundle ----------------
    def _collect_paths() -> List[Path]:
        root = Path('.')
        candidates: List[Path] = []
        def _maybe(p: Path):
            try:
                if p.exists():
                    candidates.append(p)
            except Exception:
                pass
        # Artifacts and plan cache
        _maybe(root / '.nerion' / 'plan_cache.json')
        arts = (root / '.nerion' / 'artifacts')
        if arts.exists():
            for f in arts.glob('*.json'):
                _maybe(f)
        # Index
        _maybe(root / 'out' / 'index' / 'index.json')
        # Logs / experience
        _maybe(root / 'out' / 'experience' / 'log.jsonl')
        # Learning prefs
        _maybe(root / 'out' / 'learning' / 'prefs.json')
        # Voice latency
        _maybe(root / 'out' / 'voice' / 'latency.jsonl')
        # Bench artifacts
        bench = root / 'out' / 'bench'
        if bench.exists():
            for f in bench.rglob('*'):
                if f.is_file():
                    _maybe(f)
        # Settings snapshot for reproducibility
        _maybe(root / 'app' / 'settings.yaml')
        return candidates

    def _rel_under_repo(p: Path) -> Path:
        try:
            # Compute a sensible relative path for zip entries
            root = Path('.').resolve()
            pr = p.resolve()
            return pr.relative_to(root)
        except Exception:
            return Path(p.name)

    def cmd_pack_run(_args: argparse.Namespace) -> int:
        root = Path('.')
        out_dir = root / 'out' / 'package'
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        bundle = out_dir / f'run_{ts}.zip'
        files = _collect_paths()
        count = 0
        with progress("package: pack.run"):
            with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as z:
                for f in files:
                    try:
                        arcname = _rel_under_repo(f).as_posix()
                        z.write(f, arcname)
                        count += 1
                    except Exception:
                        continue
        print(f"[package] wrote: {bundle} ({count} files)")
        print(_fmt_msg('package', 'pack.run', _MsgRes.OK, f"files={count}"))
        # Print a tiny JSON for programmatic use
        try:
            import json as _json
            print(_json.dumps({'bundle': str(bundle), 'files': count}, ensure_ascii=False))
        except Exception:
            pass
        return 0

    pk = sp.add_parser('pack', help='bundle run artifacts (plan/logs/index) into a zip under out/package')
    pksp = pk.add_subparsers(dest='pack_cmd', required=True)
    run = pksp.add_parser('run', help='zip run artifacts to out/package/run_*.zip')
    run.set_defaults(func=cmd_pack_run)
