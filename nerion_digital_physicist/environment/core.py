"""Scope-aware environment utilities for Nerion."""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

import asttokens

from ..agent.data import create_graph_data_from_source
from ..agent.semantics import SemanticEmbedder, get_global_embedder
from .actions import Action, StatefulRenameVisitor
from .generative import GenerativeActionEngine, apply_function_body


class EnvironmentV2:
    """Environment that applies precise AST-guided text edits and runs tests."""

    def __init__(
        self,
        file_to_modify: str = "logic_v2.py",
        *,
        embedder: SemanticEmbedder | None = None,
    ):
        self._root = Path(__file__).resolve().parent
        self.file_path = self._root / file_to_modify
        self.original_code = self.file_path.read_text(encoding="utf-8")
        self.embedder = embedder or get_global_embedder()
        self._generator = GenerativeActionEngine()
        self._last_metadata: dict[str, object] = {}

    def _read_current_source(self) -> str:
        return self.file_path.read_text(encoding="utf-8")

    def _transform_source(
        self, source_code: str, action: Action, *, verbose: bool = True
    ) -> tuple[str, dict[str, object]]:
        """Return the transformed source and metadata without touching disk."""

        atok = asttokens.ASTTokens(source_code, parse=True)
        modified_code = source_code
        metadata: dict[str, object] = {
            "action_tags": ("structural",),
            "action": action.name,
            "lint_passed": True,
            "validation_passed": True,
        }

        if action == Action.RENAME_LOCAL_VARIABLE_IN_ADD:
            visitor = StatefulRenameVisitor(
                target_function="add_scoped",
                old_name="value",
                new_name="input_val",
            )
            visitor.visit(atok.tree)

            for node in sorted(
                visitor.nodes_to_rename,
                key=lambda n: atok.get_text_range(n)[0],
                reverse=True,
            ):
                if isinstance(node, ast.arg):
                    start = node.first_token.startpos  # type: ignore[attr-defined]
                    end = node.first_token.endpos  # type: ignore[attr-defined]
                else:
                    start, end = atok.get_text_range(node)
                modified_code = f"{modified_code[:start]}{visitor.new_name}{modified_code[end:]}"
        elif action == Action.CHANGE_OPERATOR_MULTIPLY_TO_ADD:
            modified_code = modified_code.replace("value - decrement", "value + decrement", 1)
            metadata["action_tags"] = ("mutator", "failure_mode")
        elif action == Action.IMPLEMENT_MULTIPLY_DOCSTRING:
            metadata["action_tags"] = ("generative", "llm")
            docstring = self._extract_docstring(source_code, atok, "multiply_scoped")
            signature = "multiply_scoped(value: int, multiplier: int) -> int"
            generated = self._generator.generate_body(
                function_name="multiply_scoped",
                signature=signature,
                docstring=docstring,
            )
            body_lines = generated.lines
            metadata.update(
                {
                    "generative_provider": generated.metadata.get("provider"),
                    "generative_model": generated.metadata.get("model"),
                    "generative_latency_s": generated.metadata.get("latency_s"),
                    "generative_used_fallback": generated.metadata.get("used_fallback", False),
                    "generative_error": generated.metadata.get("error"),
                }
            )
            modified_code = apply_function_body(
                modified_code,
                function_name="multiply_scoped",
                body_lines=body_lines,
            )
        else:  # pragma: no cover - defensive default for future actions
            metadata["action_tags"] = ("structural", "unknown")

        self._last_metadata = metadata
        return modified_code, metadata

    def preview_action_source(self, action: Action) -> str:
        """Return the source code that would result from applying `action`."""

        current_source = self._read_current_source()
        transformed, _ = self._transform_source(current_source, action, verbose=False)
        return transformed

    def preview_action_graph(self, action: Action):
        """Return a PyG graph for the hypothetical post-action state."""

        transformed_source = self.preview_action_source(action)
        return create_graph_data_from_source(
            transformed_source, embedder=self.embedder
        )

    def _apply_action(self, action: Action, *, verbose: bool = True) -> str:
        """Apply an action, write it to disk, and return the modified code."""

        current_source = self._read_current_source()
        modified_code, metadata = self._transform_source(
            current_source, action, verbose=verbose
        )

        lint_passed, lint_error = self._lint_source(modified_code)
        metadata["lint_passed"] = lint_passed
        if lint_error:
            metadata["lint_error"] = lint_error
        if not lint_passed:
            if verbose:
                print(f"  -> Lint failure: {lint_error}")
            self._last_metadata = metadata
            return current_source

        if action == Action.IMPLEMENT_MULTIPLY_DOCSTRING:
            validation_passed, validation_error = self._validate_generated_code(modified_code)
            metadata["validation_passed"] = validation_passed
            if validation_error:
                metadata["validation_error"] = validation_error
            if not validation_passed:
                if verbose:
                    print(f"  -> Validation failure: {validation_error}")
                self._last_metadata = metadata
                return current_source

        if verbose:
            print("--- Generated Code ---")
            print(modified_code)
            print("----------------------")

        self.file_path.write_text(modified_code, encoding="utf-8")

        self._last_metadata = metadata
        return modified_code

    def _run_tests(self) -> bool:
        env = os.environ.copy()
        python_path = env.get("PYTHONPATH", "")
        base_dir = str(self._root)
        env["PYTHONPATH"] = f"{base_dir}:{python_path}" if python_path else base_dir

        result = subprocess.run(
            ["pytest", "-q", str(self._root / "test_logic_v2.py")],
            capture_output=True,
            text=True,
            env=env,
        )
        return result.returncode == 0

    def _restore_file(self):
        self.file_path.write_text(self.original_code, encoding="utf-8")

    def step(self, action: Action, *, verbose: bool = True) -> bool:
        if verbose:
            print(f"Executing action: {action.name}")
        self._apply_action(action, verbose=verbose)
        metadata = self._last_metadata
        if not metadata.get("lint_passed", True):
            if verbose:
                print("  -> Skipping tests due to lint failure.")
            self._restore_file()
            if verbose:
                print("  -> Environment restored to original state.")
            return False
        if not metadata.get("validation_passed", True):
            if verbose:
                print("  -> Skipping tests due to validation failure.")
            self._restore_file()
            if verbose:
                print("  -> Environment restored to original state.")
            return False
        outcome_is_success = self._run_tests()
        if verbose:
            print(f"  -> Outcome: {'Tests Passed' if outcome_is_success else 'Tests FAILED'}")
        self._restore_file()
        if verbose:
            print("  -> Environment restored to original state.")
        return outcome_is_success

    def last_action_metadata(self) -> dict[str, object]:
        """Return metadata recorded during the most recent action."""

        return dict(self._last_metadata)

    @staticmethod
    def _extract_docstring(
        source_code: str, atok: asttokens.ASTTokens, function_name: str
    ) -> str:
        tree = atok.tree
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return ast.get_docstring(node) or ""
        return ""

    def _lint_source(self, code: str) -> tuple[bool, str | None]:
        try:
            compile(code, self.file_path, "exec")
        except SyntaxError as exc:
            message = f"{exc.msg} (line {exc.lineno})"
            return False, message
        return True, None

    def _validate_generated_code(self, code: str) -> tuple[bool, str | None]:
        namespace: dict[str, object] = {}
        try:
            exec(compile(code, self.file_path, "exec"), namespace, namespace)
        except Exception as exc:  # pragma: no cover - defensive guard
            return False, f"execution_error: {exc}"

        func = namespace.get("multiply_scoped")
        if not callable(func):
            return False, "missing multiply_scoped"

        try:
            result = func(3, 4)  # type: ignore[misc]
        except Exception as exc:  # pragma: no cover - guard unexpected runtime errors
            return False, f"call_error: {exc}"
        if result != 12:
            return False, f"unexpected_result: {result}"

        try:
            func(2, -1)  # type: ignore[misc]
        except ValueError:
            return True, None
        except Exception as exc:  # pragma: no cover
            return False, f"wrong_exception: {exc}"
        else:
            return False, "missing_value_error"


if __name__ == "__main__":
    env = EnvironmentV2()
    env.step(Action.RENAME_LOCAL_VARIABLE_IN_ADD)
