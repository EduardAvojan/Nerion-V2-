"""Unified-diff text patch application utilities.

Conservative, repo-safe patch applicator that supports a common subset of
unified diff format:

  --- a/path
  +++ b/path
  @@ -l,s +l2,s2 @@
   context
  -removed
  +added

Limitations:
- Does not handle binary patches, file renames, or mode changes.
- Handles modify/create of text files present under the repo root.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from ops.security import fs_guard


@dataclass
class PatchHunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    lines: List[str]


@dataclass
class FilePatch:
    path: Path
    hunks: List[PatchHunk]


def _parse_unified_diff(diff_text: str, repo_root: Path) -> List[FilePatch]:
    files: List[FilePatch] = []
    lines = diff_text.splitlines()
    i = 0
    cur_path: Optional[Path] = None
    cur_hunks: List[PatchHunk] = []
    while i < len(lines):
        ln = lines[i]
        if ln.startswith('--- '):
            # Expect +++ on next line
            if i + 1 < len(lines) and lines[i + 1].startswith('+++ '):
                # Extract the b/ path; fallback to --- path if needed
                a = ln[4:].strip()
                b = lines[i + 1][4:].strip()
                # Strip leading a/ b/ prefixes if present
                def _clean(p: str) -> str:
                    p = p.split('\t', 1)[0]
                    if p.startswith('a/') or p.startswith('b/'):
                        return p[2:]
                    return p
                chosen = _clean(b) or _clean(a)
                try:
                    safe = fs_guard.ensure_in_repo(repo_root, chosen)
                except Exception:
                    # If the path escapes, skip this file patch entirely
                    safe = None
                cur_path = safe
                cur_hunks = []
                i += 2
                continue
        if ln.startswith('@@') and cur_path is not None:
            # Parse hunk header: @@ -l,s +l2,s2 @@ ...
            # Allow missing ",s" part (defaults to 1)
            hdr = ln
            try:
                # Extract between - and +, then after +
                seg = hdr.split('@@', 1)[1].split('@@')[0].strip()
                # e.g., "-12,3 +14,4"
                left, right = seg.split('+')
                left = left.strip().lstrip('-')
                right = right.strip()
                def _parse_pair(s: str) -> Tuple[int, int]:
                    if ',' in s:
                        a, b = s.split(',', 1)
                        return int(a), int(b)
                    return int(s), 1
                old_start, old_len = _parse_pair(left)
                new_start, new_len = _parse_pair(right)
            except Exception:
                # Malformed header — abort this hunk
                i += 1
                continue
            i += 1
            hunk_lines: List[str] = []
            while i < len(lines):
                line = lines[i]
                if line.startswith('@@') or line.startswith('--- '):
                    break
                if line and line[0] in {' ', '+', '-'}:
                    # Keep the full line content including leading marker and trailing text
                    hunk_lines.append(line)
                else:
                    # Lines outside markers are ignored within a hunk
                    hunk_lines.append(' ' + line)
                i += 1
            cur_hunks.append(PatchHunk(old_start, old_len, new_start, new_len, hunk_lines))
            # Do not continue here; next loop step will handle next hunk or file header
            continue
        # If we reached a non-hunk line after +++ and we have hunks collected, finalize the file patch
        if cur_path is not None and cur_hunks and (ln.startswith('--- ') or i == len(lines) - 1):
            files.append(FilePatch(path=cur_path, hunks=cur_hunks))
            cur_path = None
            cur_hunks = []
        i += 1
    # Finalize last file if any hunks collected
    if cur_path is not None and cur_hunks:
        files.append(FilePatch(path=cur_path, hunks=cur_hunks))
    return files


def _apply_hunks_to_text(orig: str, hunks: List[PatchHunk]) -> Optional[str]:
    """Apply hunks to a single file's text. Returns new text or None on mismatch.

    Conservative: verifies context lines and removed lines match exactly.
    """
    src = orig.splitlines(keepends=True)
    out: List[str] = []
    cursor = 0  # current index in src
    for h in hunks:
        old_start_idx = max(0, h.old_start - 1)
        # Copy unchanged block up to the hunk start
        if old_start_idx < cursor:
            # Overlapping or out-of-order hunks
            return None
        out.extend(src[cursor:old_start_idx])
        cursor = old_start_idx
        # Apply the hunk
        for line in h.lines:
            if not line:
                continue
            tag = line[0]
            body = line[1:]
            if tag == ' ':
                # Context: must match
                if cursor >= len(src) or src[cursor] != (body + ('\n' if not body.endswith('\n') else '')):
                    # Try forgiving check (strip universal newline)
                    if cursor >= len(src) or src[cursor].rstrip('\n') != body:
                        return None
                out.append(src[cursor])
                cursor += 1
            elif tag == '-':
                # Removal: must match and skip
                if cursor >= len(src) or src[cursor] != (body + ('\n' if not body.endswith('\n') else '')):
                    if cursor >= len(src) or src[cursor].rstrip('\n') != body:
                        return None
                cursor += 1
            elif tag == '+':
                # Addition
                out.append(body + ('\n' if not body.endswith('\n') else ''))
            else:
                # Unknown marker; treat as context line
                out.append(body + ('\n' if not body.endswith('\n') else ''))
    # Append remainder
    out.extend(src[cursor:])
    text = ''.join(out)
    if not text.endswith('\n'):
        text += '\n'
    return text


def preview_unified_diff(diff_text: str, repo_root: Path) -> Tuple[Dict[Path, Tuple[str, str]], List[str]]:
    """Compute old/new text for each file affected by the diff without writing.

    Returns ({path: (before, after)}, errors)
    """
    errors: List[str] = []
    previews: Dict[Path, Tuple[str, str]] = {}
    file_patches = _parse_unified_diff(diff_text, repo_root)
    for fp in file_patches:
        try:
            p = fs_guard.ensure_in_repo(repo_root, fp.path)
        except Exception:
            errors.append(f"path_outside_repo:{fp.path}")
            continue
        try:
            before = Path(p).read_text(encoding='utf-8')
        except FileNotFoundError:
            errors.append(f"missing:{p}")
            continue
        except Exception as e:
            errors.append(f"read_error:{p}:{e}")
            continue
        after = _apply_hunks_to_text(before, fp.hunks)
        if after is None:
            errors.append(f"mismatch:{p}")
            continue
        previews[Path(p)] = (before, after)
    return previews, errors

