from __future__ import annotations
from pathlib import Path
from selfcoder.plans.schema import validate_plan
from ops.security import fs_guard
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from selfcoder.actions.transformers import (
    apply_actions_via_ast,
    ModuleDocstringAdder,
    FunctionDocstringAdder,
)

# Import all orchestration utilities from modular package
from selfcoder.orchestration import (
    env_true as _env_true,
    prepare_for_prompt,
    run_actions_on_file,
    run_actions_on_files,
    apply_plan,
    run_ast_actions,
    dry_run_orchestrate,
    OrchestrateResult,
    REPO_ROOT,
    _normalize_actions_for_apply,
    run_batch_actions_on_files,
    apply_actions_preview as _apply_actions_preview,
    unified_diff_for_file as _unified_diff_for_file,
)

# A lightweight registry for higher-level callers (optional use)
_TRANSFORMERS: Dict[str, Any] = {
    "add_module_docstring": ModuleDocstringAdder,
    "add_function_docstring": FunctionDocstringAdder,
}

# Backwards compatibility: Orchestrator class
class Orchestrator:
    """Minimal orchestrator preserved for backward compatibility.

    Applies in-file AST actions to files and (optionally) writes the results
    when `preview` is False. Cross-file operations are not implemented in this
    shim; callers requiring them should use newer utilities.
    """

    def __init__(self, actions: List[Dict[str, Any]] | None = None) -> None:
        self.actions: List[Dict[str, Any]] = list(actions or [])

    @staticmethod
    def apply_plan(plan: Dict[str, Any], *, dry_run: bool = False) -> List[Path]:
        """Compatibility shim: delegate to module-level apply_plan."""
        return apply_plan(plan, dry_run=dry_run)

    def apply_in_file(self, source: str) -> Tuple[str, int]:
        new_src = apply_actions_via_ast(source, self.actions)
        known = sum(
            1 for a in self.actions if isinstance(a, dict) and a.get("kind") in _TRANSFORMERS
        )
        return new_src, known

    def orchestrate_files(
        self,
        files: List[Path],
        root: Path | None = None,
        preview: bool = True,
    ) -> OrchestrateResult:
        modified: List[Path] = []
        in_file_count = 0

        for p in files:
            jp = fs_guard.ensure_in_repo_auto(str(p))
            try:
                text = jp.read_text(encoding="utf-8")
            except Exception:
                continue
            new_text, n = self.apply_in_file(text)
            in_file_count += int(n)
            if new_text != text:
                try:
                    if not preview:
                        jp.write_text(new_text, encoding="utf-8")
                    modified.append(jp)
                except Exception:
                    # Ignore write errors at this layer
                    pass

        # Cross-file not supported in this shim; report zero.
        return OrchestrateResult(
            modified_files=modified,
            in_file_edits=in_file_count,
            crossfile_edits=0,
        )


__all__ = [
    "apply_actions_via_ast",
    "_TRANSFORMERS",
    "run_actions_on_file",
    "run_actions_on_files",
    "apply_plan",
    "run_ast_actions",
    "dry_run_orchestrate",
    "Orchestrator",
    "OrchestrateResult",
    "prepare_for_prompt",
    "REPO_ROOT",
    "_normalize_actions_for_apply",
    "run_batch_actions_on_files",
    "_apply_actions_preview",
    "_unified_diff_for_file",
]
