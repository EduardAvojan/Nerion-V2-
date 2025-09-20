import ast
import sys
import pathlib

ROOT = pathlib.Path(".").resolve()
SHUTIL_FUNCS = {"copy", "copy2", "move", "rmtree"}
OS_FUNCS = {"remove", "rename", "mkdir", "makedirs"}

# Skip noisy or non-repo-edit paths (snapshots, tests, caches, etc.)
EXCLUDE_PARTS = {".venv", "node_modules", "__pycache__", ".git", ".nerion", "dist", "build", "out"}
EXCLUDE_SUBPATHS = {
    "selfcoder/tests",
    "tests/",
    "backups/snapshots",
    "out/security_audit",
    ".nerion/snapshots",
}


def _is_open_write(call: ast.Call) -> bool:
    """Return True iff an `open()` call looks like it may write."""
    if not (isinstance(call.func, ast.Name) and call.func.id == "open"):
        return False

    mode = None
    # Positional mode
    if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant) and isinstance(call.args[1].value, str):
        mode = call.args[1].value
    else:
        # Keyword mode
        for kw in call.keywords or []:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                mode = kw.value.value
                break

    if mode is None:
        # No mode => defaults to "r" (read-only) in Python; don't flag.
        return False
    return any(ch in mode for ch in ("w", "a", "x", "+"))


def is_writer_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    # open(..., mode="w"/"a"/"x"/"+")
    if _is_open_write(node):
        return True
    # Path(...).write_text / write_bytes
    if isinstance(node.func, ast.Attribute) and node.func.attr in ("write_text", "write_bytes"):
        return True
    # shutil.* writers
    if (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "shutil"
        and node.func.attr in SHUTIL_FUNCS
    ):
        return True
    # os.* mutators
    if (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "os"
        and node.func.attr in OS_FUNCS
    ):
        return True
    return False


def has_guard_call_in_func(func: ast.FunctionDef) -> bool:
    SAFE_HELPERS = {
        # io_safe wrappers
        "write_text","write_bytes","open_write",
        "mkdir","makedirs","remove","rename","copy","copy2","move","rmtree",
        # explicit guards
        "ensure_in_repo_auto","ensure_in_repo",
    }
    for n in ast.walk(func):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute) and f.attr in SAFE_HELPERS:
                return True
            if isinstance(f, ast.Name) and f.id in SAFE_HELPERS:
                return True
    return False


def scan_file(path: pathlib.Path):
    try:
        src = path.read_text(encoding="utf-8")
    except Exception:
        return []
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return []

    issues = []
    # Consider any function (top-level, nested, or class method)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            writes = any(is_writer_call(n) for n in ast.walk(node) if isinstance(n, ast.Call))
            if writes and not has_guard_call_in_func(node):
                issues.append((path, node.name, node.lineno))
    return issues


def _should_skip(p: pathlib.Path) -> bool:
    sp = p.as_posix()
    for part in p.parts:
        if part in EXCLUDE_PARTS:
            return True
    for sub in EXCLUDE_SUBPATHS:
        if sub in sp:
            return True
    return False


def main():
    targets = [p for p in ROOT.rglob("*.py") if not _should_skip(p)]
    all_issues = []
    for p in targets:
        all_issues.extend(scan_file(p))

    if all_issues:
        print("Unguarded writers (no ensure_in_repo_* in same function):")
        for path, fn, ln in all_issues:
            print(f"{path}:{ln} in {fn}()")
        sys.exit(1)
    else:
        print("✅ No obvious unguarded writers found (heuristic).")


if __name__ == "__main__":
    main()
