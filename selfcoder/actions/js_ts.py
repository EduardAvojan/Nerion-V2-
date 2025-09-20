"""Minimal JS/TS transformers (textual, conservative).

Supported kinds (subset):
- add_module_docstring: prepend a top comment (/** ... */)
- insert_function: append a function skeleton with optional doc comment
- rename_symbol: naive token rename with word boundaries
 - insert_class: append a class skeleton with optional doc comment
 - insert_import: insert a simple import (default/named/namespace) if missing

These are intentionally lightweight to avoid heavy JS/TS parser deps.
"""

from __future__ import annotations

import re
from typing import Dict, Any, List, Optional
import os
try:
    from .js_ts_node import apply_actions_js_ts_node as _apply_node
except Exception:  # pragma: no cover
    _apply_node = None  # type: ignore


def _prepend_doc(source: str, doc: str) -> str:
    if not doc:
        return source
    head = source.lstrip()
    # If file already starts with a comment block, keep it; else add one
    if head.startswith("/*") or head.startswith("//"):
        return source
    block = f"/** {doc} */\n"
    # Preserve shebang if present
    if source.startswith("#!"):
        lines = source.splitlines(True)
        return lines[0] + block + "".join(lines[1:])
    return block + source


def _append_function(source: str, name: str, doc: str | None = None) -> str:
    if not name or not re.match(r"^[A-Za-z_$][A-Za-z0-9_$]*$", name):
        return source
    doc_block = f"/** {doc} */\n" if (doc and str(doc).strip()) else ""
    snippet = (
        f"\n{doc_block}export function {name}() {{\n  // TODO\n}}\n"
    )
    out = source
    if not out.endswith("\n"):
        out += "\n"
    out += snippet
    return out


def _rename_symbol(source: str, old: str, new: str) -> str:
    if not old or not new or old == new:
        return source
    # Token-boundary replacement: avoid replacing property accesses like obj.old by requiring a boundary
    # Note: This does not skip strings/comments; kept simple by design.
    pattern = re.compile(rf"\b{re.escape(old)}\b")
    return pattern.sub(new, source)


def _append_class(source: str, name: str, doc: Optional[str] = None) -> str:
    if not name or not re.match(r"^[A-Za-z_$][A-Za-z0-9_$]*$", name):
        return source
    # Idempotency: if class already exists, no-op
    if re.search(rf"\bclass\s+{re.escape(name)}\b", source):
        return source
    doc_block = f"/** {doc} */\n" if (doc and str(doc).strip()) else ""
    snippet = f"\n{doc_block}export class {name} {{\n  constructor() {{}}\n}}\n"
    out = source
    if not out.endswith("\n"):
        out += "\n"
    out += snippet
    return out


def _insert_import(source: str, module: str, *, default: Optional[str] = None, named: Optional[List[str]] = None, namespace: Optional[str] = None) -> str:
    """Insert an import statement if not already present.

    Forms supported:
      - default:   import React from 'react';
      - named:     import { useState, useEffect } from 'react';
      - namespace: import * as fs from 'fs';
    Precedence: namespace > default+named > default only > named only.
    """
    mod = (module or '').strip()
    if not mod:
        return source
    # If file already imports from module, be conservative and no-op
    if re.search(rf"^\s*import\b[\s\S]*?from\s+['\"]{re.escape(mod)}['\"]\s*;?\s*$", source, flags=re.M):
        return source
    if namespace:
        imp = f"import * as {namespace} from '{mod}';"
    else:
        parts = []
        if default:
            parts.append(default)
        if named:
            names = ', '.join(sorted(set(str(n).strip() for n in named if str(n).strip())))
            if names:
                parts.append(f"{{ {names} }}")
        if not parts:
            # Fallback to side-effect import
            imp = f"import '{mod}';"
        elif len(parts) == 1:
            imp = f"import {parts[0]} from '{mod}';"
        else:
            # default + named
            imp = f"import {parts[0]}, {parts[1]} from '{mod}';"
    # Insert after existing import block if present
    lines = source.splitlines(True)
    i = 0
    # Skip top-of-file comments (// and /* ... */) and blank lines
    while i < len(lines):
        s = lines[i].lstrip()
        if s.startswith('//') or lines[i].strip() == '':
            i += 1
            continue
        if s.startswith('/*'):
            # advance until closing */
            i += 1
            while i < len(lines) and '*/' not in lines[i]:
                i += 1
            if i < len(lines):
                i += 1
            continue
        break
    # Now advance past existing import block (contiguous imports)
    while i < len(lines) and lines[i].lstrip().startswith('import '):
        i += 1
    new_lines = lines[:i] + [imp + "\n"] + lines[i:]
    out = ''.join(new_lines)
    if not out.endswith('\n'):
        out += '\n'
    return out


def apply_actions_js_ts(source: str, actions: List[Dict[str, Any]]) -> str:
    # Optional Node bridge (ts-morph) for more precise transforms
    try:
        if (_apply_node is not None) and (os.getenv('NERION_JS_TS_NODE') or '').strip().lower() in {'1','true','yes','on'}:
            new_src = _apply_node(source, actions)
            if isinstance(new_src, str) and new_src:
                return new_src if new_src.endswith('\n') else (new_src + '\n')
    except Exception:
        pass
    updated = source
    for a in actions or []:
        if not isinstance(a, dict):
            continue
        kind = a.get("kind") or a.get("action")
        payload = a.get("payload") or {}
        try:
            if kind == "add_module_docstring":
                updated = _prepend_doc(updated, str(payload.get("doc") or "").strip())
            elif kind == "insert_function":
                nm = payload.get("name") or payload.get("symbol") or payload.get("function")
                updated = _append_function(updated, str(nm or "").strip(), str(payload.get("doc") or "").strip() or None)
            elif kind == "insert_class":
                nm = payload.get("name") or payload.get("symbol") or payload.get("class")
                updated = _append_class(updated, str(nm or "").strip(), str(payload.get("doc") or "").strip() or None)
            elif kind == "insert_import":
                mod = str(payload.get('module') or payload.get('from') or '').strip()
                default = payload.get('default')
                named = payload.get('named')
                if isinstance(named, str):
                    named = [named]
                namespace = payload.get('namespace')
                updated = _insert_import(updated, mod, default=str(default) if default else None, named=[str(n) for n in (named or [])] or None, namespace=str(namespace) if namespace else None)
            elif kind == "rename_symbol":
                old = payload.get("from") or payload.get("old") or payload.get("symbol")
                new = payload.get("to") or payload.get("new")
                if old and new:
                    updated = _rename_symbol(updated, str(old), str(new))
        except Exception:
            # Skip action on error to keep operation safe
            continue
    if not updated.endswith("\n"):
        updated += "\n"
    return updated
