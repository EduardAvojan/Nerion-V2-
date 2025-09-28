"""AST patch helpers for rename actions.

This module provides a helper to post-process updated source text based on action payloads,
fixing bare-name references when from-import attributes are renamed.
"""

from __future__ import annotations
from typing import List, Dict, Any


def postprocess_attr_renames(updated: str, actions: Optional[List[Dict[str, Any]]]) -> str:
    """Fix bare-name references when an action renames a from-import attribute.

    Returns the potentially updated source code string.
    """
    try:
        # Extract potential mapping(s) from actions
        mappings = []  # list of tuples: (old_mod, new_mod, attr_old, attr_new)
        for a in actions or []:
            if not isinstance(a, dict):
                continue
            payload = (a.get("payload") or {})
            old_mod = (
                payload.get("old")
                or payload.get("old_module")
                or payload.get("module_old")
            )
            new_mod = (
                payload.get("new")
                or payload.get("new_module")
                or payload.get("module_new")
            )
            attr_old = (
                payload.get("attr_old")
                or payload.get("attr-old")
                or payload.get("old_attr")
            )
            attr_new = (
                payload.get("attr_new")
                or payload.get("attr-new")
                or payload.get("new_attr")
            )
            if attr_old and attr_new:
                mappings.append((old_mod, new_mod, attr_old, attr_new))

        if mappings:
            import ast
            tree = ast.parse(updated)

            class _AttrRename(ast.NodeTransformer):
                def __init__(self, maps):
                    super().__init__()
                    self.maps = maps  # list of (old_mod, new_mod, attr_old, attr_new)
                    self.active = {}  # type: dict[str, str]

                def visit_ImportFrom(self, node):  # type: ignore[override]
                    for (old_mod, new_mod, a_old, a_new) in self.maps:
                        if old_mod and node.module == old_mod:
                            if new_mod:
                                node.module = new_mod
                            for alias in node.names:
                                if alias.asname is None and alias.name == a_old:
                                    alias.name = a_new
                                    self.active[a_old] = a_new
                                elif alias.asname is None and alias.name == a_new:
                                    self.active[a_old] = a_new
                        if new_mod and node.module == new_mod:
                            for alias in node.names:
                                if alias.asname is None and alias.name == a_old:
                                    alias.name = a_new
                                    self.active[a_old] = a_new
                                elif alias.asname is None and alias.name == a_new:
                                    self.active[a_old] = a_new
                    return node

                def visit_Name(self, node):  # type: ignore[override]
                    if isinstance(node.ctx, ast.Load) and node.id in self.active:
                        return ast.copy_location(ast.Name(id=self.active[node.id], ctx=node.ctx), node)
                    return node

            new_tree = _AttrRename(mappings).visit(tree)
            ast.fix_missing_locations(new_tree)
            updated = ast.unparse(new_tree)
    except Exception:
        # Fail-safe: if anything goes wrong, fall back to the current updated text
        pass

    return updated