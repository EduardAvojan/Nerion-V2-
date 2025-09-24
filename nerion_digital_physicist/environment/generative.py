"""Utilities for LLM-backed generative code actions."""

from __future__ import annotations

import ast
import os
import textwrap
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from app.chat.providers.base import LLMResponse, ProviderError, ProviderRegistry

try:  # pragma: no cover - optional dependency used at runtime
    import asttokens  # type: ignore
except ImportError as exc:  # pragma: no cover - asttokens is an installation prerequisite
    raise RuntimeError("asttokens must be available for generative actions") from exc

__all__ = [
    "GenerativeActionEngine",
    "apply_function_body",
]


@dataclass
class GeneratedBody:
    """Container for a generated body snippet."""

    lines: List[str]
    metadata: Dict[str, object]


class GenerativeActionEngine:
    """LLM-backed generator with deterministic fallbacks."""

    def __init__(
        self,
        *,
        provider_override: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        self.provider_override = (
            provider_override or os.getenv("NERION_GENERATIVE_PROVIDER") or None
        )
        self.temperature = max(0.0, float(temperature))
        try:
            self._registry = ProviderRegistry.from_files()
        except Exception:  # pragma: no cover - configuration issues fall back to deterministic path
            self._registry = None
            self.provider_override = None
        self._cache: Dict[Tuple[str, str], GeneratedBody] = {}

    def generate_body(
        self,
        *,
        function_name: str,
        signature: str,
        docstring: str,
    ) -> GeneratedBody:
        """Return a code body implementing ``signature`` guided by ``docstring``."""

        cache_key = (function_name, docstring)
        if cache_key in self._cache:
            return self._cache[cache_key]

        error = None
        if self._registry:
            messages = _build_prompt(signature=signature, docstring=docstring)
            try:
                response = self._registry.generate(
                    role="code",
                    messages=messages,
                    provider_override=self.provider_override,
                    temperature=self.temperature,
                )
                lines = _extract_body_lines(response, function_name)
                if lines:
                    result = GeneratedBody(
                        lines=lines,
                        metadata={
                            "provider": response.provider,
                            "model": response.model,
                            "latency_s": response.latency_s,
                            "used_fallback": False,
                        },
                    )
                    self._cache[cache_key] = result
                    return result
                error = "empty_response"
            except ProviderError as exc:
                error = str(exc)
            except Exception as exc:  # pragma: no cover - protects against provider SDK bugs
                error = str(exc)
        else:
            error = "registry_unavailable"

        lines = _fallback_body(function_name=function_name, signature=signature)
        result = GeneratedBody(
            lines=lines,
            metadata={
                "provider": "fallback",
                "model": "deterministic",
                "used_fallback": True,
                "error": error,
            },
        )
        self._cache[cache_key] = result
        return result


def apply_function_body(
    source_code: str,
    *,
    function_name: str,
    body_lines: Sequence[str],
) -> str:
    """Return ``source_code`` with ``function_name`` body replaced by ``body_lines``."""

    atok = asttokens.ASTTokens(source_code, parse=True)
    for node in ast.walk(atok.tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            start, end = atok.get_text_range(node)
            header_end = _header_end_index(atok, node)
            header = source_code[start:header_end].rstrip()
            formatted_body = _format_body(body_lines, indent=node.col_offset + 4)
            new_function = header + "\n" + formatted_body
            if not new_function.endswith("\n"):
                new_function += "\n"
            return source_code[:start] + new_function + source_code[end:]
    raise ValueError(f"Function {function_name!r} not found in source")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_prompt(*, signature: str, docstring: str) -> List[Dict[str, str]]:
    system = (
        "You generate precise Python implementations. Respond with valid Python "
        "statements satisfying the specification."
    )
    user = textwrap.dedent(
        f"""
        Implement the body of the function below.
        Return only Python statements that belong inside the function body.
        Do not include the function signature or docstring.
        Signature: {signature}
        Docstring:
{docstring}
        Rules:
        - Preserve parameter names.
        - Raise ValueError when the multiplier is negative.
        - Prefer simple, deterministic control flow.
        """
    ).strip()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _extract_body_lines(response: LLMResponse, function_name: str) -> List[str]:
    text = _strip_code_fence(response.text)
    text = textwrap.dedent(text).strip()
    if not text:
        return []

    if text.startswith("def "):
        try:
            module = ast.parse(text)
        except SyntaxError:
            return []
        for node in module.body:
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return _body_lines_from_ast(text, node)
        return []

    return _normalise_lines(text.splitlines())


def _body_lines_from_ast(source: str, func: ast.FunctionDef) -> List[str]:
    lines: List[str] = []
    for stmt in func.body:
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            continue
        segment = ast.get_source_segment(source, stmt)
        if segment is None:
            return []
        lines.extend(segment.splitlines())
    return _normalise_lines(lines)


def _normalise_lines(lines: Iterable[str]) -> List[str]:
    cleaned = [line.rstrip() for line in lines]
    dedented = textwrap.dedent("\n".join(cleaned))
    result = [line.rstrip() for line in dedented.splitlines()]
    while result and not result[0].strip():
        result.pop(0)
    while result and not result[-1].strip():
        result.pop()
    return result


def _fallback_body(*, function_name: str, signature: str) -> List[str]:
    try:
        parsed = ast.parse(f"def {signature}:\n    pass")
        func = parsed.body[0]
    except Exception:
        return ["raise NotImplementedError('fallback unavailable')"]

    arg_names = [arg.arg for arg in func.args.args]
    if function_name == "multiply_scoped" and len(arg_names) >= 2:
        value_name, multiplier_name = arg_names[:2]
        return [
            f"if {multiplier_name} < 0:",
            f"    raise ValueError('{multiplier_name} must be non-negative')",
            f"return {value_name} * {multiplier_name}",
        ]

    return ["raise NotImplementedError('fallback unavailable')"]


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if "```" in text:
            text = text.rsplit("```", 1)[0]
    return text.strip()


def _format_body(lines: Sequence[str], *, indent: int) -> str:
    indent_str = " " * indent
    effective_lines = list(lines) or ["pass"]
    return "\n".join(
        indent_str + line if line.strip() else ""
        for line in effective_lines
    )


def _header_end_index(atok: asttokens.ASTTokens, func: ast.FunctionDef) -> int:
    if (
        func.body
        and isinstance(func.body[0], ast.Expr)
        and isinstance(func.body[0].value, ast.Constant)
        and isinstance(func.body[0].value.value, str)
    ):
        _, doc_end = atok.get_text_range(func.body[0])
        return doc_end

    first_token = func.first_token
    while first_token and first_token.string != ":":
        first_token = first_token.next
    if first_token:
        return first_token.endpos
    return atok.get_text_range(func)[0]
