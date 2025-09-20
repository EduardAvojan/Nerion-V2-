# plugins/security.py
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional, Tuple, Dict
import hashlib
import builtins as _builtins
import os as _os
from contextlib import contextmanager

class PluginNotAllowed(PermissionError): ...
class PluginSandboxViolation(PermissionError): ...

def load_allowlist_data(repo_root: Path) -> Dict:
    p = (repo_root / "plugins" / "allowlist.json")
    if not p.exists():
        return {"allowed": [], "hashes": {}}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"allowed": [], "hashes": {}}
        data.setdefault("allowed", [])
        data.setdefault("hashes", {})
        return data
    except Exception:
        return {"allowed": [], "hashes": {}}


def _load_allowlist(repo_root: Path) -> Tuple[set[str], Optional[Path], Dict[str, str]]:
    data = load_allowlist_data(repo_root)
    allowed = set(x for x in data.get("allowed", []) if isinstance(x, str))
    hashes = {k: str(v) for k, v in (data.get("hashes", {}) or {}).items() if isinstance(k, str)}
    if not allowed:
        test_mode_dir = None
        plugins_dir_env = os.environ.get("NERION_PLUGINS_DIR")
        if plugins_dir_env:
            test_mode_dir = Path(plugins_dir_env).resolve()
        return allowed, test_mode_dir, hashes
    return allowed, None, hashes


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def assert_plugin_allowed(repo_root: Path, module_name: str, module_path: Optional[Path] = None) -> None:
    allowed, test_mode_dir, hashes = _load_allowlist(repo_root)
    if module_name not in allowed:
        if test_mode_dir and module_path is not None:
            try:
                resolved_module_path = module_path.resolve()
            except Exception:
                resolved_module_path = module_path
            if str(resolved_module_path).startswith(str(test_mode_dir)):
                return
        raise PluginNotAllowed(f"Plugin not allowlisted: {module_name}")
    # Optionally: ensure it lives under <repo>/plugins/
    if module_path is not None:
        plugins_dir = (repo_root / "plugins").resolve()
        resolved = module_path.resolve()
        # Deny symlinked plugin files
        if module_path.is_symlink():
            raise PluginNotAllowed(f"Plugin path is a symlink: {module_path}")
        if not str(resolved).startswith(str(plugins_dir)):
            raise PluginNotAllowed(f"Plugin path outside plugins dir: {module_path}")
        # Hash pinning (optional or required via env). In test-mode (NERION_PLUGINS_DIR),
        # skip hash requirement to keep local/dev flows convenient.
        expected = hashes.get(module_name)
        require_hash = (os.getenv('NERION_PLUGINS_REQUIRE_HASH') or '').strip().lower() in {'1','true','yes','on'} and (test_mode_dir is None)
        if expected or require_hash:
            try:
                actual = _sha256(resolved)
                if not expected and require_hash:
                    raise PluginNotAllowed(f"Missing hash for {module_name} and NERION_PLUGINS_REQUIRE_HASH is set")
                if expected and actual.lower() != str(expected).strip().lower():
                    raise PluginNotAllowed(f"Hash mismatch for {module_name}: expected {expected} got {actual}")
            except Exception as e:
                raise PluginNotAllowed(f"Failed to verify hash for {module_name}: {e}")

@contextmanager
def plugin_sandbox(repo_root: Path):
    """Runtime FS sandbox for plugin code: deny writes outside plugins/.

    Only intercepts common filesystem writes: open(..., 'w/a/x/+'), os.remove, os.rename, os.replace.
    Reads are allowed. This is a best-effort in-process guard (not a kernel sandbox).
    """
    root = Path(repo_root).resolve()
    allowed = (root / 'plugins').resolve()

    def _guard_path(p: object) -> None:
        try:
            fp = Path(p).resolve()
        except Exception:
            raise PluginSandboxViolation(f"unresolvable path: {p!r}")
        if not str(fp).startswith(str(allowed)):
            raise PluginSandboxViolation(f"write outside plugins dir: {fp}")

    _orig_open = _builtins.open
    _orig_remove = _os.remove
    _orig_rename = _os.rename
    _orig_replace = _os.replace

    def _open_guard(path, mode='r', *a, **k):
        m = str(mode or 'r')
        if any(flag in m for flag in ('w', 'a', 'x', '+')):
            _guard_path(path)
        return _orig_open(path, mode, *a, **k)

    def _rm_guard(path, *a, **k):
        _guard_path(path)
        return _orig_remove(path, *a, **k)

    def _rename_guard(src, dst, *a, **k):
        _guard_path(src)
        _guard_path(dst)
        return _orig_rename(src, dst, *a, **k)

    def _replace_guard(src, dst, *a, **k):
        _guard_path(src)
        _guard_path(dst)
        return _orig_replace(src, dst, *a, **k)

    try:
        _builtins.open = _open_guard  # type: ignore
        _os.remove = _rm_guard  # type: ignore
        _os.rename = _rename_guard  # type: ignore
        _os.replace = _replace_guard  # type: ignore
        yield
    finally:
        _builtins.open = _orig_open  # type: ignore
        _os.remove = _orig_remove  # type: ignore
        _os.rename = _orig_rename  # type: ignore
        _os.replace = _orig_replace  # type: ignore
