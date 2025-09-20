"""
Built-in offline diff proposer for bench repair.

Generates one or more minimal unified diffs based on common Python failures.
Focuses on safe, surgical edits and never touches tests.

Current fixers:
  - Unresolved import / ModuleNotFoundError → add `import X`
  - NameError for common aliases (np, pd) → add canonical import

API mirrors optional plugin:
  - propose_diff(ctx) -> str | None
  - propose_diff_multi(ctx) -> list[str]

The ctx is the JSON object written by triage with keys like:
  { 'failures': [...], '_task_dir': '/abs/path/to/task' }
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import re
import os
import difflib


_TEST_PAT = re.compile(r"(^|/)tests(/|$)|(^|/)test_.*\\.py$")


def _is_source_file(p: Path) -> bool:
    if p.suffix != ".py":
        return False
    posix = p.as_posix()
    return _TEST_PAT.search(posix) is None


def _select_top_frame_file(frames: List[Dict[str, Any]], task_dir: Path) -> Optional[Path]:
    for fr in frames:
        f = Path(fr.get("file"))
        try:
            f = f if f.is_absolute() else (task_dir / f)
        except Exception:
            continue
        if _is_source_file(f) and f.exists():
            return f
    return None


def _parse_failures(ctx: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Path]:
    failures = list(ctx.get("failures") or ctx.get("failed") or [])
    task_dir = Path(ctx.get("_task_dir") or ".").resolve()
    return failures, task_dir


def _iter_source_files(task_dir: Path):
    for root, dirs, files in os.walk(task_dir):
        rp = Path(root)
        # prune tests
        if _TEST_PAT.search(rp.as_posix()):
            continue
        for fn in files:
            if fn.endswith('.py'):
                p = rp / fn
                if _is_source_file(p):
                    yield p


def _grep_rank(task_dir: Path, tokens: List[str], max_files: int = 8) -> List[Path]:
    """Rank candidate files by simple substring frequency of tokens.
    Returns a list of Paths sorted by score (desc)."""
    if not tokens:
        return []
    scores: List[Tuple[int, Path]] = []
    toks = [t for t in tokens if isinstance(t, str) and t.strip()]
    if not toks:
        return []
    for fp in _iter_source_files(task_dir):
        try:
            text = fp.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        sc = 0
        for t in toks:
            sc += text.count(t)
        if sc > 0:
            scores.append((sc, fp))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scores[:max_files]]


def _rank_focus_file(frames: List[Dict[str, Any]], task_dir: Path, tokens: List[str]) -> Optional[Path]:
    tgt = _select_top_frame_file(frames, task_dir)
    if tgt:
        return tgt
    cands = _grep_rank(task_dir, tokens)
    if cands:
        return cands[0]
    # As a last resort, consult index graph for symbol tokens
    try:
        from selfcoder.analysis import index_api as _idxapi  # lazy import
        for t in tokens:
            if not t or not isinstance(t, str):
                continue
            aff = _idxapi.affected(task_dir, t, transitive=True)
            for s in aff:
                p = Path(s)
                if p.exists() and _is_source_file(p):
                    return p
    except Exception:
        pass
    return None


def _file_src(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _insert_import(src: str, module: str, alias: Optional[str] = None) -> Optional[str]:
    """Insert a top-level import if not present. Place after module docstring and before code.
    Returns new source or None if no change."""
    if not module or not isinstance(src, str):
        return None
    line = f"import {module}"
    if alias:
        line = f"import {module} as {alias}"
    # already present?
    patt = re.compile(rf"^\s*import\s+{re.escape(module)}(\s+as\s+{re.escape(alias)}\s*)?$", re.M) if alias else re.compile(rf"^\s*import\s+{re.escape(module)}\b", re.M)
    if patt.search(src):
        return None
    # find insertion point: after module docstring and existing imports block
    lines = src.splitlines()
    i = 0
    # skip shebang/encoding
    if i < len(lines) and lines[i].startswith("#!"):
        i += 1
    if i < len(lines) and lines[i].startswith("# -*- coding"):
        i += 1
    # module docstring
    if i < len(lines) and re.match(r"^\s*\"\"\"|^\s*'\'\'", lines[i]):
        quote = "\"\"\"" if lines[i].lstrip().startswith("\"\"\"") else "'''"
        i += 1
        while i < len(lines) and quote not in lines[i]:
            i += 1
        if i < len(lines):
            i += 1
    # skip future/from/import lines block
    while i < len(lines) and re.match(r"^\s*(from\s+\S+\s+import\s+|import\s+)", lines[i]):
        i += 1
    new_lines = lines[:i] + [line] + lines[i:]
    return "\n".join(new_lines) + ("\n" if src.endswith("\n") else "")


def _unified_diff(a_path: Path, a_old: str, a_new: str) -> str:
    import difflib
    a = a_old.splitlines(keepends=True)
    b = a_new.splitlines(keepends=True)
    path = a_path.as_posix()
    # Normalize to relative-looking path (strip leading slash for readability)
    if path.startswith('/'):
        path = path.lstrip('/')
    return "".join(difflib.unified_diff(a, b, fromfile=f"a/{path}", tofile=f"b/{path}"))


def _fix_unresolved_import(ctx: Dict[str, Any]) -> List[str]:
    failures, task_dir = _parse_failures(ctx)
    diffs: List[str] = []
    # Patterns
    pat_mod1 = re.compile(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]")
    pat_mod2 = re.compile(r"ImportError: No module named\s+([^\s]+)")
    for rec in failures:
        tb = rec.get("traceback") or rec.get("error") or rec.get("message") or ""
        m = pat_mod1.search(str(tb)) or pat_mod2.search(str(tb))
        if not m:
            continue
        mod = m.group(1).strip()
        frames = rec.get("frames") or rec.get("stack") or []
        target = _rank_focus_file(frames, task_dir, tokens=[mod])
        if not target:
            continue
        src = _file_src(target)
        if not src:
            continue
        new_src = _insert_import(src, mod)
        if new_src and new_src != src:
            disp = target
            try:
                disp = target.relative_to(task_dir)
            except Exception:
                pass
            diffs.append(_unified_diff(disp, src, new_src))
    return diffs


_ALIAS_IMPORTS = {
    "np": ("numpy", "np"),
    "pd": ("pandas", "pd"),
}


def _fix_nameerror_common_alias(ctx: Dict[str, Any]) -> List[str]:
    failures, task_dir = _parse_failures(ctx)
    diffs: List[str] = []
    pat = re.compile(r"NameError: name ['\"]([^'\"]+)['\"] is not defined")
    for rec in failures:
        tb = rec.get("traceback") or rec.get("error") or rec.get("message") or ""
        m = pat.search(str(tb))
        if not m:
            continue
        name = m.group(1).strip()
        if name not in _ALIAS_IMPORTS:
            continue
        mod, alias = _ALIAS_IMPORTS[name]
        frames = rec.get("frames") or rec.get("stack") or []
        target = _rank_focus_file(frames, task_dir, tokens=[name])
        if not target:
            continue
        src = _file_src(target)
        if not src:
            continue
        new_src = _insert_import(src, mod, alias=alias)
        if new_src and new_src != src:
            disp = target
            try:
                disp = target.relative_to(task_dir)
            except Exception:
                pass
            diffs.append(_unified_diff(disp, src, new_src))
    return diffs


def _fix_nameerror_safe_rename(ctx: Dict[str, Any]) -> List[str]:
    """Attempt a conservative safe rename on a NameError by fixing a likely typo.

    Heuristic:
      - Parse NameError: name 'X' is not defined
      - Identify focus file via frames/grep/index ranking
      - On the failing line, if a close lexical match Y exists in the file (recent lines)
        or function parameters/assignments, replace X -> Y on that line only.
    """
    failures, task_dir = _parse_failures(ctx)
    diffs: List[str] = []
    pat = re.compile(r"NameError: name ['\"]([^'\"]+)['\"] is not defined")
    for rec in failures:
        tb = rec.get("traceback") or rec.get("error") or rec.get("message") or ""
        m = pat.search(str(tb))
        if not m:
            continue
        miss = m.group(1).strip()
        # Choose target file
        frames = rec.get("frames") or rec.get("stack") or []
        target = _rank_focus_file(frames, task_dir, tokens=[miss])
        if not target:
            continue
        line_no = None
        try:
            for fr in frames:
                f = Path(fr.get("file") or "")
                if f and (f == target or target.as_posix().endswith(Path(f).as_posix())):
                    line_no = int(fr.get("line") or 0) or None
                    if line_no:
                        break
        except Exception:
            line_no = None
        src = _file_src(target)
        if not src:
            continue
        lines = src.splitlines()
        # Collect candidates: names in recent window + params/assignments
        window_start = max(0, (line_no or 1) - 25)
        window = "\n".join(lines[window_start:(line_no or 1)])
        # identifiers around
        ids = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", window))
        # Prefer function params if available
        def_idx = _find_enclosing_function(lines, line_no or 1)
        if def_idx is not None:
            sig = lines[def_idx]
            mparams = re.search(r"def\s+\w+\s*\(([^)]*)\)", sig)
            if mparams:
                for p in mparams.group(1).split(","):
                    nm = p.strip().split("=")[0].strip()
                    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", nm):
                        ids.add(nm)
        # Remove obvious builtins/common words
        ids.discard(miss)
        if not ids:
            continue
        # Pick closest match
        cand = None
        try:
            matches = difflib.get_close_matches(miss, sorted(ids), n=1, cutoff=0.8)
            cand = matches[0] if matches else None
        except Exception:
            cand = None
        if not cand:
            continue
        if not line_no or not (1 <= line_no <= len(lines)):
            continue
        text = lines[line_no - 1]
        if miss not in text:
            continue
        new_text = text.replace(miss, cand, 1)
        if new_text == text:
            continue
        new_src = "\n".join(lines[: line_no - 1] + [new_text] + lines[line_no:])
        if new_src != src:
            disp = target
            try:
                disp = target.relative_to(task_dir)
            except Exception:
                pass
            diffs.append(_unified_diff(disp, src, new_src))
    return diffs


def propose_diff(ctx: Dict[str, Any]) -> Optional[str]:
    """Return a single best diff or None."""
    cands = propose_diff_multi(ctx)
    return cands[0] if cands else None


def propose_diff_multi(ctx: Dict[str, Any]) -> List[str]:
    """Return multiple candidate diffs (strings)."""
    diffs: List[str] = []
    # Run fixers in priority order
    for fixer in (
        _fix_unresolved_import,
        _fix_nameerror_common_alias,
        _fix_nameerror_safe_rename,
        _fix_none_attrerror_guard,
        _fix_index_guard,
        _fix_valueerror_cast_guard,
        _fix_typeerror_binary_op_guard,
        _fix_numeric_tolerance_line,
    ):
        try:
            out = fixer(ctx)
            for d in out or []:
                if isinstance(d, str) and d.strip() and d not in diffs:
                    diffs.append(d)
        except Exception:
            continue
    return diffs


__all__ = ["propose_diff", "propose_diff_multi"]


# ---- Additional fixers ----------------------------------------------------


def _find_enclosing_function(lines: List[str], line_no_1based: int) -> Optional[int]:
    """Return 0-based index of the 'def' line that encloses the given line, or None."""
    i = max(0, int(line_no_1based) - 1)
    for j in range(i, -1, -1):
        if re.match(r"^\s*def\s+\w+\s*\(", lines[j]):
            return j
    return None


def _after_def_insertion_idx(lines: List[str], def_idx: int) -> int:
    """Return index after def and any docstring, suitable for inserting a guard."""
    i = def_idx + 1
    # Skip blank/comment lines
    while i < len(lines) and (lines[i].strip() == "" or lines[i].lstrip().startswith("#")):
        i += 1
    # Skip docstring block if present
    if i < len(lines) and re.match(r"^\s*[\"\']{3}", lines[i]):
        quote = "\"\"\"" if lines[i].lstrip().startswith("\"\"\"") else "'''"
        i += 1
        while i < len(lines) and quote not in lines[i]:
            i += 1
        if i < len(lines):
            i += 1
    return i


def _insert_lines(src: str, insert_at: int, new_lines: List[str]) -> str:
    L = src.splitlines()
    L[insert_at:insert_at] = new_lines
    text = "\n".join(L)
    if src.endswith("\n"):
        text += "\n"
    return text


def _fix_none_attrerror_guard(ctx: Dict[str, Any]) -> List[str]:
    """Insert `if var is None: return None` at start of the enclosing function.

    Looks for AttributeError mentioning 'NoneType' and extracts a probable variable from
    the failing line (token before a dot).
    """
    failures, task_dir = _parse_failures(ctx)
    diffs: List[str] = []
    pat = re.compile(r"AttributeError: 'NoneType' object has no attribute '([A-Za-z_][A-Za-z0-9_]*)'")
    for rec in failures:
        tb = str(rec.get("traceback") or rec.get("error") or rec.get("message") or "")
        m_attr = pat.search(tb)
        if not m_attr:
            continue
        frames = rec.get("frames") or rec.get("stack") or []
        # Prefer explicit frame; fallback to grep by attribute name
        target = _select_top_frame_file(frames, task_dir)
        if not target:
            target = _rank_focus_file([], task_dir, tokens=[m_attr.group(1)])
        if not target:
            continue
        line_no = None
        try:
            for fr in frames:
                if Path(fr.get("file")) == target or (target in (task_dir / Path(fr.get("file") or "")).resolve().parents):
                    line_no = int(fr.get("line")) if fr.get("line") else None
                    if line_no:
                        break
        except Exception:
            line_no = None
        src = _file_src(target)
        if not src:
            continue
        lines = src.splitlines()
        var = None
        if line_no and 1 <= line_no <= len(lines):
            text = lines[line_no - 1]
            m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\.", text)
            if m:
                var = m.group(1)
        if not var:
            continue
        def_idx = _find_enclosing_function(lines, line_no or 1)
        if def_idx is None:
            continue
        def_indent = len(lines[def_idx]) - len(lines[def_idx].lstrip(" "))
        insert_at = _after_def_insertion_idx(lines, def_idx)
        guard = [" " * (def_indent + 4) + f"if {var} is None:", " " * (def_indent + 8) + "return None"]
        new_src = _insert_lines(src, insert_at, guard)
        if new_src != src:
            disp = target
            try:
                disp = target.relative_to(task_dir)
            except Exception:
                pass
            diffs.append(_unified_diff(disp, src, new_src))
    return diffs


def _fix_index_guard(ctx: Dict[str, Any]) -> List[str]:
    """Insert an index bounds guard for simple subscript patterns on failing line."""
    failures, task_dir = _parse_failures(ctx)
    diffs: List[str] = []
    pat_idx1 = re.compile(r"IndexError: .*out of range|out of bounds", re.I)
    for rec in failures:
        tb = str(rec.get("traceback") or rec.get("error") or rec.get("message") or "")
        if not pat_idx1.search(tb):
            continue
        frames = rec.get("frames") or rec.get("stack") or []
        target = _select_top_frame_file(frames, task_dir)
        if not target:
            continue
        line_no = None
        try:
            for fr in frames:
                if Path(fr.get("file")) == target:
                    line_no = int(fr.get("line")) if fr.get("line") else None
                    if line_no:
                        break
        except Exception:
            line_no = None
        src = _file_src(target)
        if not src or not line_no or line_no < 1:
            continue
        lines = src.splitlines()
        if line_no > len(lines):
            continue
        text = lines[line_no - 1]
        m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*([^\]]+)\s*\]", text)
        if not m:
            continue
        arr, idx_expr = m.group(1), m.group(2)
        def_idx = _find_enclosing_function(lines, line_no)
        if def_idx is None:
            continue
        def_indent = len(lines[def_idx]) - len(lines[def_idx].lstrip(" "))
        insert_at = _after_def_insertion_idx(lines, def_idx)
        guard = [
            " " * (def_indent + 4) + f"if not (0 <= int({idx_expr}) < len({arr})):",
            " " * (def_indent + 8) + "return None",
        ]
        new_src = _insert_lines(src, insert_at, guard)
        if new_src != src:
            disp = target
            try:
                disp = target.relative_to(task_dir)
            except Exception:
                pass
            diffs.append(_unified_diff(disp, src, new_src))
    return diffs


def _fix_numeric_tolerance_line(ctx: Dict[str, Any]) -> List[str]:
    """Replace direct float equality on a source line with math.isclose (best-effort).

    Only patches non-test files and only the single failing line if it contains '==' or '!='.
    """
    failures, task_dir = _parse_failures(ctx)
    diffs: List[str] = []
    for rec in failures:
        # Heuristic: look for AssertionError in tests but patch a non-test source frame
        tb = str(rec.get("traceback") or rec.get("error") or rec.get("message") or "")
        if "AssertionError" not in tb:
            continue
        frames = rec.get("frames") or rec.get("stack") or []
        target = _select_top_frame_file(frames, task_dir)
        if not target:
            continue
        line_no = None
        try:
            for fr in frames:
                if Path(fr.get("file")) == target:
                    line_no = int(fr.get("line")) if fr.get("line") else None
                    if line_no:
                        break
        except Exception:
            line_no = None
        src = _file_src(target)
        if not src or not line_no or line_no < 1:
            continue
        lines = src.splitlines()
        if line_no > len(lines):
            continue
        text = lines[line_no - 1]
        if "==" not in text and "!=" not in text:
            continue
        # Best-effort: require both sides look like expressions (letters/digits/._())
        m = re.search(r"(.+?)(==|!=)(.+)", text)
        if not m:
            continue
        lhs, _op, rhs = m.group(1).strip(), m.group(2), m.group(3).strip()
        if not lhs or not rhs:
            continue
        # Replace direct equality with tolerance-based check
        repl = f"math.isclose({lhs}, {rhs}, rel_tol=1e-9, abs_tol=1e-12)"
        new_line = re.sub(r"(.+?)(==|!=)(.+)", repl, text)
        if new_line == text:
            continue
        # Add import math at top if missing
        new_src = "\n".join(lines[: line_no - 1] + [new_line] + lines[line_no:])
        if "import math" not in new_src:
            new_src = ("import math\n" + new_src)
        if new_src != src:
            disp = target
            try:
                disp = target.relative_to(task_dir)
            except Exception:
                pass
            diffs.append(_unified_diff(disp, src, new_src))
    return diffs


def _fix_valueerror_cast_guard(ctx: Dict[str, Any]) -> List[str]:
    """Wrap simple int()/float() casts by converting argument earlier with try/except.

    Inserts at function start:
        try: x = int(x)
        except (ValueError, TypeError): return None
    """
    failures, task_dir = _parse_failures(ctx)
    diffs: List[str] = []
    pat_val = re.compile(r"ValueError: (invalid literal for (int|float)\(\).+|could not convert string to (int|float).+)", re.I)
    for rec in failures:
        tb = str(rec.get("traceback") or rec.get("error") or rec.get("message") or "")
        if not pat_val.search(tb):
            continue
        frames = rec.get("frames") or rec.get("stack") or []
        target = _select_top_frame_file(frames, task_dir)
        if not target:
            # Grep by common cast patterns (int( / float()), since fn isn't known yet)
            target = _rank_focus_file([], task_dir, tokens=["int(", "float("])
        if not target:
            continue
        line_no = None
        try:
            for fr in frames:
                if Path(fr.get("file")) == target:
                    line_no = int(fr.get("line")) if fr.get("line") else None
                    if line_no:
                        break
        except Exception:
            line_no = None
        src = _file_src(target)
        if not src or not line_no:
            continue
        lines = src.splitlines()
        text = lines[line_no - 1]
        m = re.search(r"\b(int|float)\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", text)
        if not m:
            continue
        fn, var = m.group(1), m.group(2)
        def_idx = _find_enclosing_function(lines, line_no)
        if def_idx is None:
            continue
        def_indent = len(lines[def_idx]) - len(lines[def_idx].lstrip(" "))
        insert_at = _after_def_insertion_idx(lines, def_idx)
        guard = [
            " " * (def_indent + 4) + "try:",
            " " * (def_indent + 8) + f"{var} = {fn}({var})",
            " " * (def_indent + 4) + "except (ValueError, TypeError):",
            " " * (def_indent + 8) + "return None",
        ]
        new_src = _insert_lines(src, insert_at, guard)
        if new_src != src:
            disp = target
            try:
                disp = target.relative_to(task_dir)
            except Exception:
                pass
            diffs.append(_unified_diff(disp, src, new_src))
    return diffs


def _fix_typeerror_binary_op_guard(ctx: Dict[str, Any]) -> List[str]:
    """Coerce string operands to float for simple binary ops; otherwise return None.

    Inserts at function start checks on operands found on failing line.
    """
    failures, task_dir = _parse_failures(ctx)
    diffs: List[str] = []
    pat = re.compile(r"TypeError: unsupported operand type\(s\) for [+\-*/]")
    for rec in failures:
        tb = str(rec.get("traceback") or rec.get("error") or rec.get("message") or "")
        if not pat.search(tb):
            continue
        frames = rec.get("frames") or rec.get("stack") or []
        target = _select_top_frame_file(frames, task_dir)
        if not target:
            continue
        line_no = None
        try:
            for fr in frames:
                if Path(fr.get("file")) == target:
                    line_no = int(fr.get("line")) if fr.get("line") else None
                    if line_no:
                        break
        except Exception:
            line_no = None
        src = _file_src(target)
        if not src or not line_no:
            continue
        lines = src.splitlines()
        text = lines[line_no - 1]
        m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*([+\-*/])\s*([A-Za-z_][A-Za-z0-9_]*)", text)
        if not m:
            continue
        a, _op, b = m.group(1), m.group(2), m.group(3)
        def_idx = _find_enclosing_function(lines, line_no)
        if def_idx is None:
            continue
        def_indent = len(lines[def_idx]) - len(lines[def_idx].lstrip(" "))
        insert_at = _after_def_insertion_idx(lines, def_idx)
        guard = [
            " " * (def_indent + 4) + f"if isinstance({a}, str):",
            " " * (def_indent + 8) + "try:",
            " " * (def_indent + 12) + f"{a} = float({a})",
            " " * (def_indent + 8) + "except Exception:",
            " " * (def_indent + 12) + "return None",
            " " * (def_indent + 4) + f"if isinstance({b}, str):",
            " " * (def_indent + 8) + "try:",
            " " * (def_indent + 12) + f"{b} = float({b})",
            " " * (def_indent + 8) + "except Exception:",
            " " * (def_indent + 12) + "return None",
        ]
        new_src = _insert_lines(src, insert_at, guard)
        if new_src != src:
            disp = target
            try:
                disp = target.relative_to(task_dir)
            except Exception:
                pass
            diffs.append(_unified_diff(disp, src, new_src))
    return diffs
