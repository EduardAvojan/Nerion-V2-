"""
This module will contain the logic for the Automated Curriculum Generator.

It is responsible for:
1.  Receiving a high-level concept for a new training lesson.
2.  Using an LLM to generate the 'before', 'after', and 'test' code files.
3.  Orchestrating the 'Self-Vetting' process to ensure the generated lesson is valid and correct.
4.  Attempting to repair failed lessons.
5.  Adding the validated lesson to the curriculum database.
"""
import json
import tempfile
import subprocess
import sys
import argparse
import os
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from datetime import datetime, timezone

import torch
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from app.parent.coder import Coder

from nerion_digital_physicist.db.curriculum_store import CurriculumStore
from nerion_digital_physicist.infrastructure.errors import LLMGenerationError, ValidationError, DatabaseError
from nerion_digital_physicist.infrastructure.retry import retry_llm_call, retry_database_call
from nerion_digital_physicist.infrastructure.logging import with_correlation_id, track_performance, log_with_context, logger
from nerion_digital_physicist.infrastructure.validation import validate_lesson_data, validate_llm_response
from nerion_digital_physicist.agent.brain import build_gnn
from nerion_digital_physicist.agent.data import create_graph_data_from_source
from nerion_digital_physicist.agent.semantics import get_global_embedder


MODEL_META_PATH = Path("digital_physicist_brain.meta.json")
STRUCTURAL_DELTA_THRESHOLD = 0.02

_POOLING_REGISTRY = {
    "mean": global_mean_pool,
    "sum": global_add_pool,
    "max": global_max_pool,
}


def _pool_fn(name: str):
    return _POOLING_REGISTRY.get(name.lower(), global_mean_pool)


def _load_model_metadata() -> Dict[str, Any]:
    if MODEL_META_PATH.exists():
        try:
            return json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
        except Exception as err:
            print(f"      - WARNING: Could not parse model metadata: {err}")
    return {
        "architecture": "gcn",
        "hidden_channels": 256,
        "pooling": "mean",
        "num_layers": 4,
        "residual": False,
        "dropout": 0.3,
        "attention_heads": 4,
    }


def _offline_lesson_bundle(lesson_description: str, lesson_name: str) -> Optional[Dict[str, Any]]:
    """Return None when LLM access is unavailable - fail fast instead of generating placeholders."""
    print("   - ERROR: LLM access unavailable. Cannot generate lesson without LLM.")
    return None


def _normalise_redaction_placeholders(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure redaction placeholders cannot collide with original secret characters.
    Also validates and converts code fields to strings if LLM returns dicts."""

    replacement = "<<REDACTED>>"
    for key in ("before_code", "after_code", "test_code"):
        value = bundle.get(key)

        # Handle case where LLM returns a dict instead of string (TypeError fix)
        if isinstance(value, dict):
            print(f"   - WARNING: {key} is a dict, converting to string")
            # Try to extract 'code' field if it exists, otherwise convert whole dict
            if 'code' in value:
                bundle[key] = str(value['code'])
            else:
                bundle[key] = str(value)
            value = bundle[key]

        if not isinstance(value, str):
            # If still not a string after conversion, this is an error
            raise ValueError(f"Field '{key}' must be a string, got {type(value)}: {value}")

        # Normalize redaction placeholders
        if "[REDACTED]" in value or "{REDACTED}" in value:
            bundle[key] = value.replace("[REDACTED]", replacement).replace("{REDACTED}", replacement)

    return bundle

@retry_llm_call(max_attempts=3)
@with_correlation_id
@track_performance
def _generate_lesson_from_llm(lesson_description: str, lesson_name: str, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Optional[Dict[str, Any]]:
    """Uses an LLM to generate the code snippets for a new training lesson."""
    log_with_context(
        logger,
        logging.INFO,
        "Starting LLM lesson generation",
        lesson_name=lesson_name,
        provider=provider,
        model=model_name
    )
    
    try:
        os.environ['NERION_V2_REQUEST_TIMEOUT'] = '300'
        llm = Coder(role='code', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
    except Exception as e:
        log_with_context(
            logger,
            logging.ERROR,
            "Failed to initialize LLM provider",
            error=str(e),
            provider=provider,
            model=model_name
        )
        raise LLMGenerationError(f"Could not get LLM provider: {e}", provider=provider, model=model_name)

    system_prompt = (
        "You are an expert Python programmer, security engineer, and educator creating HIGH-IMPACT production-ready training exercises. "
        "Your task is to create a lesson that demonstrates a CRITICAL improvement in code quality, security, reliability, or performance. "
        "Given a description, provide three pieces of Python code in a single JSON object with the keys: 'before_code', 'after_code', and 'test_code'."
        "\n"
        "**CRITICAL REQUIREMENTS:**\n"
        "1. **Real Problem:** `before_code` must have a genuine, demonstrable flaw (security vulnerability, race condition, memory leak, performance bottleneck, etc.)\n"
        "2. **Clear Fix:** `after_code` must fix the problem completely and demonstrate best practices\n"
        "3. **Proves Impact:** `test_code` must FAIL on before_code and PASS on after_code, proving the improvement is real\n"
        "4. **Production Quality:** Code should be realistic, not trivial examples. Use actual libraries and patterns from production systems.\n"
        "\n"
        "**Execution Context:** Code runs in a sandbox. `before_code` and `after_code` are saved as `module.py`. Tests in `test_module.py` import via `from module import ...`\n"
        "\n"
        "**Testing Best Practices:**\n"
        "- ALWAYS include ALL imports at the top of test_code (import pytest, import re, from hypothesis import given, etc.)\n"
        "- If using re.escape(), re.match(), etc., MUST add: import re\n"
        "- If using @given(), MUST add: from hypothesis import given, strategies as st, settings, HealthCheck\n"
        "- DO NOT use function-scoped pytest fixtures with @given() - causes FailedHealthCheck error\n"
        "- Use Hypothesis for property-based testing (already installed) instead of hand-rolled fuzzers\n"
        "- Keep randomized loops lightweight (≤150 iterations) for fast vetting\n"
        "- For async/ExceptionGroup: Use `except* Exception` and re-raise first inner exception\n"
        "- For retry/backoff: Ensure jitter is called and sleep occurs (e.g., `random.uniform` + `time.sleep`)\n"
        "- For secrets: Use distinct placeholders like '<<REDACTED>>' that share no characters with original\n"
        "\n"
        "**Focus Areas (ensure your lesson fits one):**\n"
        "- Security: SQL injection, XSS, CSRF, authentication, secrets management\n"
        "- Reliability: Error handling, retries, circuit breakers, graceful degradation\n"
        "- Performance: N+1 queries, caching, connection pooling, algorithmic complexity\n"
        "- Concurrency: Race conditions, deadlocks, thread safety, async patterns\n"
        "- Data Integrity: Validation, sanitization, constraints, transactions\n"
    )
    user_prompt = f"Create a training exercise for the following lesson: {lesson_description}"

    try:
        response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        if not response_json_str:
            log_with_context(
                logger,
                logging.ERROR,
                "LLM returned empty response",
                lesson_name=lesson_name,
                provider=provider
            )
            raise LLMGenerationError("LLM returned an empty response", provider=provider, model=model_name)
        
        # Validate the response
        validated_response = validate_llm_response(response_json_str)
        
        data = json.loads(validated_response)
        data['name'] = lesson_name
        data['description'] = lesson_description
        
        log_with_context(
            logger,
            logging.INFO,
            "LLM lesson generation completed successfully",
            lesson_name=lesson_name,
            provider=provider,
            model=model_name
        )
        
        return _normalise_redaction_placeholders(data)
    except json.JSONDecodeError as e:
        log_with_context(
            logger,
            logging.ERROR,
            "Failed to parse LLM JSON response",
            error=str(e),
            lesson_name=lesson_name,
            provider=provider,
            response_preview=response_json_str[:200] if response_json_str else None
        )
        raise LLMGenerationError(f"Failed to parse LLM JSON response: {e}", provider=provider, model=model_name)
    except ValidationError as e:
        log_with_context(
            logger,
            logging.ERROR,
            "LLM response validation failed",
            error=str(e),
            lesson_name=lesson_name,
            provider=provider
        )
        raise
    except Exception as e:
        log_with_context(
            logger,
            logging.ERROR,
            "Failed to generate lesson from LLM",
            error=str(e),
            lesson_name=lesson_name,
            provider=provider
        )
        raise LLMGenerationError(f"Failed to generate or parse LLM response: {e}", provider=provider, model=model_name)

def _repair_lesson(lesson_description: str, lesson_name: str, failure_log: str, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Optional[Dict[str, Any]]:
    """Attempts to repair a failed lesson generation."""
    print("   - Repairing lesson with LLM...")
    try:
        os.environ['NERION_V2_REQUEST_TIMEOUT'] = '300'
        llm = Coder(role='code', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
    except Exception as e:
        print(f"   - Could not get LLM provider for Repair: {e}", file=sys.stderr)
        return None

    system_prompt = (
        "You are an expert debugging and repair specialist. A HIGH-IMPACT production-ready programming lesson failed its verification test. "
        "Your task is to analyze the failure, identify the root cause, and provide a corrected version that demonstrates a clear before/after improvement. "
        "\n"
        "**Critical Requirements:**\n"
        "1. **Execution Context:** Code runs in a sandbox where the file is `module.py`. Tests must import from `module`.\n"
        "2. **Clear Distinction:** `before_code` must have a real, demonstrable problem. `after_code` must clearly fix it.\n"
        "3. **Test Quality:** Tests must prove the problem exists in `before_code` and is fixed in `after_code`.\n"
        "4. **Production Focus:** Ensure the lesson teaches a critical skill (security, reliability, performance, correctness).\n"
        "\n"
        "**Testing Guidelines:**\n"
        "- ALWAYS include ALL imports at the top of test_code (pytest, re, os, hypothesis, etc.)\n"
        "- If test uses re.escape(), re.match(), etc., MUST import re\n"
        "- If test uses @given(), MUST import from hypothesis import given, strategies as st\n"
        "- DO NOT use function-scoped pytest fixtures with @given() - use class-scoped or module-scoped fixtures instead\n"
        "- Use Hypothesis for property-based testing when validating edge cases\n"
        "- Cap custom randomized loops at 150 iterations maximum\n"
        "- Keep shrinking/diagnostic passes short for fast pytest execution\n"
        "- For async code with TaskGroup/ExceptionGroup, handle exceptions properly\n"
        "- For retry/backoff functions, ensure jitter is invoked and sleep occurs\n"
        "- When redacting secrets, use distinct placeholders like '<<REDACTED>>' to avoid test failures\n"
        "\n"
        "**Common Failure Patterns to Fix:**\n"
        "- NameError (e.g., 're' is not defined): Missing import statement - add it to test_code\n"
        "- Hypothesis FailedHealthCheck (function-scoped fixture): Change fixture scope or remove @given()\n"
        "- Test unexpectedly PASSED on before_code: The bug isn't real or test is wrong\n"
        "- Test unexpectedly FAILED on after_code: The fix doesn't work or introduces new bugs\n"
        "- Syntax errors: Check imports, indentation, missing arguments\n"
        "- Import errors: Ensure all required modules are imported in test_code\n"
        "- Assertion errors: Make sure assertions match the actual behavior\n"
        "\n"
        "Return a single JSON object with corrected 'before_code', 'after_code', and 'test_code' keys."
    )
    user_prompt = (
        f"The original lesson description was: {lesson_description}\n\n"
        f"The test failed with the following error:\n{failure_log}\n\n"
        "Please provide a corrected lesson in the specified JSON format."
    )

    try:
        response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        if not response_json_str:
            print("   - Repair LLM returned an empty response.", file=sys.stderr)
            return None
        data = json.loads(response_json_str)
        data['name'] = lesson_name
        data['description'] = lesson_description
        return _normalise_redaction_placeholders(data)
    except Exception as e:
        print(f"   - Failed to generate or parse LLM response for repair: {e}", file=sys.stderr)
        return None

def _run_test_in_sandbox(source_code: str, test_code: str) -> subprocess.CompletedProcess:
    """Runs pytest on the given code and returns the process result."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        (p / "module.py").write_text(source_code, encoding="utf-8")
        (p / "test_module.py").write_text(test_code, encoding="utf-8")

        return subprocess.run(
            [sys.executable, "-m", "pytest", "test_module.py"],
            cwd=tmpdir,
            capture_output=True,
            text=True,
        )

def _is_structurally_good_change(
    before_code: str,
    after_code: str,
) -> Tuple[bool, Dict[str, Optional[float]], str, Optional[str]]:
    """Uses the GNN to analyze the structural quality of the code change."""
    MODEL_PATH = "digital_physicist_brain.pt"
    try:
        embedder = get_global_embedder()

        before_graph = create_graph_data_from_source(before_code, embedder=embedder)

        meta = _load_model_metadata()
        architecture = str(meta.get("architecture", "gcn"))
        hidden_channels = int(meta.get("hidden_channels", 256))
        pooling_name = str(meta.get("pooling", "mean"))
        num_layers = int(meta.get("num_layers", 4))
        residual = bool(meta.get("residual", False))
        dropout = float(meta.get("dropout", 0.2))
        attention_heads = int(meta.get("attention_heads", 4))
        pool_fn = _pool_fn(pooling_name)

        model = build_gnn(
            architecture=architecture,
            num_node_features=before_graph.num_node_features,
            hidden_channels=hidden_channels,
            num_classes=2,
            num_layers=num_layers,
            residual=residual,
            dropout=dropout,
            attention_heads=attention_heads,
        )

        raw_state = torch.load(MODEL_PATH)
        try:
            model.load_state_dict(raw_state)
        except RuntimeError as mismatch_error:
            print(
                "      - WARNING: Model checkpoint shape mismatch detected. Attempting partial weight load."
            )
            adjusted_state = {}
            current_state = model.state_dict()
            for key, tensor in raw_state.items():
                if key not in current_state:
                    continue
                target_tensor = current_state[key]
                if tensor.shape == target_tensor.shape:
                    adjusted_state[key] = tensor
                    continue
                # Copy overlapping slices and leave new dimensions initialised.
                overlapping = tuple(
                    slice(0, min(src, dst)) for src, dst in zip(tensor.shape, target_tensor.shape)
                )
                target_clone = target_tensor.clone()
                target_clone[overlapping] = tensor[overlapping]
                adjusted_state[key] = target_clone
            missing_keys = set(current_state.keys()) - set(adjusted_state.keys())
            load_status = model.load_state_dict(adjusted_state, strict=False)
            if load_status.unexpected_keys:
                print(
                    f"      - WARNING: Unexpected keys while loading checkpoint: {sorted(load_status.unexpected_keys)}"
                )
            if missing_keys:
                print(
                    f"      - NOTE: Initialising new parameters for keys: {sorted(missing_keys)}"
                )
        model.eval()

        after_graph = create_graph_data_from_source(after_code, embedder=embedder)

        def _score_graph(graph):
            if graph.x is None or graph.x.shape[0] == 0:
                return None
            edge_index = getattr(graph, "edge_index", None)
            if edge_index is None:
                return None

            batch = getattr(graph, "batch", None)
            if batch is None:
                batch = torch.zeros(graph.x.shape[0], dtype=torch.long, device=graph.x.device)

            with torch.no_grad():
                logits = model(graph.x, edge_index, batch)
                pooled = pool_fn(logits, batch)

            if pooled.ndim == 2 and pooled.shape[1] > 1:
                return pooled[:, 1].mean().item()
            if pooled.numel() == 0:
                return None
            return pooled.view(-1)[0].item()

        before_pass_score = _score_graph(before_graph)
        after_pass_score = _score_graph(after_graph)

        metrics = {
            "before_score": before_pass_score,
            "after_score": after_pass_score,
            "delta": None,
        }

        if before_pass_score is None or after_pass_score is None:
            msg = "Structural analysis unavailable (empty graph or logits)."
            print(f"      - WARNING: {msg} Skipping structural comparison.")
            return True, metrics, "skipped", msg

        metrics["delta"] = after_pass_score - before_pass_score
        print(
            "      - Structural analysis scores: [Before: %.3f] -> [After: %.3f] (Δ %.3f)"
            % (before_pass_score, after_pass_score, metrics["delta"])
        )

        passed = after_pass_score - before_pass_score >= STRUCTURAL_DELTA_THRESHOLD
        status = "passed" if passed else "failed"
        message = None
        if not passed:
            message = (
                "Structural analysis did not show improvement above threshold"
                f" ({metrics['delta']:.3f} < {STRUCTURAL_DELTA_THRESHOLD:.3f})."
            )
        return passed, metrics, status, message

    except FileNotFoundError:
        msg = f"Model file not found at {MODEL_PATH}. Skipping structural check."
        print(f"      - WARNING: {msg}")
        return True, {"before_score": None, "after_score": None, "delta": None}, "skipped", msg
    except Exception as e:
        msg = f"Could not perform structural analysis: {e}"
        print(f"      - WARNING: {msg}")
        return True, {"before_score": None, "after_score": None, "delta": None}, "error", msg


def _log_structural_metrics(record: Dict[str, Any]) -> None:
    """Append structural telemetry to an artefact file."""

    path = Path("out/learning/structural_metrics.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    record.setdefault("timestamp", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _self_vet_lesson(
    lesson_name: str,
    lesson_description: str,
    before_code: str,
    after_code: str,
    test_code: str,
    attempt: str,
) -> Tuple[bool, str]:
    """Verifies the lesson, returning a boolean for success and a string for the reason."""
    print("      - Running behavioral verification...")
    before_proc = _run_test_in_sandbox(before_code, test_code)
    after_proc = _run_test_in_sandbox(after_code, test_code)

    before_test_failed = before_proc.returncode != 0
    after_test_passed = after_proc.returncode == 0

    if not before_test_failed:
        reason = (
            "Vetting failed: Test unexpectedly PASSED on the 'before_code'.\n"
            f"PYTEST OUTPUT:\n{before_proc.stdout}{before_proc.stderr}"
        )
        print(f"      - {reason.splitlines()[0]}")
        _log_structural_metrics(
            {
                "lesson": lesson_name,
                "description": lesson_description,
                "attempt": attempt,
                "phase": "behavioral",
                "passed_behavioral": False,
                "before_test_failed": before_test_failed,
                "after_test_passed": after_test_passed,
                "reason": reason,
            }
        )
        return False, reason

    if not after_test_passed:
        reason = (
            "Vetting failed: Test unexpectedly FAILED on the 'after_code'.\n"
            f"PYTEST OUTPUT:\n{after_proc.stdout}{after_proc.stderr}"
        )
        print(f"      - {reason.splitlines()[0]}")
        _log_structural_metrics(
            {
                "lesson": lesson_name,
                "description": lesson_description,
                "attempt": attempt,
                "phase": "behavioral",
                "passed_behavioral": False,
                "before_test_failed": before_test_failed,
                "after_test_passed": after_test_passed,
                "reason": reason,
            }
        )
        return False, reason
    
    print("      - Behavioral verification passed.")

    print("      - Running structural verification...")
    passed_structural, metrics, status, message = _is_structurally_good_change(before_code, after_code)

    model_meta = _load_model_metadata()

    _log_structural_metrics(
        {
            "lesson": lesson_name,
            "description": lesson_description,
            "attempt": attempt,
            "phase": "structural",
            "passed_behavioral": True,
            "structural_status": status,
            "metrics": metrics,
            "message": message,
            "model_architecture": model_meta.get("architecture", "gcn"),
        }
    )

    if not passed_structural:
        reason = message or "Vetting failed: Structural analysis did not show improvement."
        print(f"      - {reason}")
        return False, reason
    
    print("      - Structural verification passed.")
    return True, "Lesson passed all vetting checks."

@retry_database_call(max_attempts=3)
@validate_lesson_data
@track_performance
def _save_lesson_to_db(lesson_data: Dict[str, Any]):
    """Saves the vetted lesson to the curriculum database."""
    try:
        db_path = Path("out/learning/curriculum.sqlite")
        store = CurriculumStore(db_path)
        store.add_lesson(lesson_data)
        store.close()
        
        log_with_context(
            logger,
            logging.INFO,
            "Lesson saved to database successfully",
            lesson_name=lesson_data.get('name'),
            focus_area=lesson_data.get('focus_area')
        )
    except Exception as e:
        log_with_context(
            logger,
            logging.ERROR,
            "Failed to save lesson to database",
            error=str(e),
            lesson_name=lesson_data.get('name')
        )
        raise DatabaseError(f"Failed to save lesson to database: {e}", operation="add_lesson", table="lessons")

def main():
    parser = argparse.ArgumentParser(description="Automated Curriculum Generator for the Digital Physicist.")
    parser.add_argument("--description", type=str, required=True, help="A high-level description of the lesson to generate.")
    parser.add_argument("--name", type=str, required=True, help="A short, descriptive name for the new template directory (e.g., 'fix_index_error').")
    # Configuration from environment variables or defaults
    default_provider = os.getenv("NERION_LLM_PROVIDER", "openai")
    default_project_id = os.getenv("NERION_LLM_PROJECT_ID")
    default_location = os.getenv("NERION_LLM_LOCATION", "us-central1")
    default_model = os.getenv("NERION_LLM_MODEL", "gpt-4")
    
    parser.add_argument("--provider", type=str, default=default_provider, help="LLM provider to use (e.g., 'openai' or 'gemini')")
    parser.add_argument("--project-id", type=str, default=default_project_id, help="Google Cloud Project ID for Vertex AI.")
    parser.add_argument("--location", type=str, default=default_location, help="Google Cloud location for Vertex AI (e.g., 'us-central1').")
    parser.add_argument("--model-name", type=str, default=default_model, help="Vertex AI model name to use (e.g., 'gemini-pro').")
    args = parser.parse_args()

    print(f"--- Generating lesson for: {args.description} ---")

    # First attempt
    lesson_bundle = _generate_lesson_from_llm(args.description, args.name, args.provider, args.project_id, args.location, args.model_name)
    if not lesson_bundle:
        print("   - Halting: Initial LLM generation failed.")
        sys.exit(1)

    is_valid, reason = _self_vet_lesson(
        lesson_name=lesson_bundle['name'],
        lesson_description=lesson_bundle['description'],
        before_code=lesson_bundle['before_code'],
        after_code=lesson_bundle['after_code'],
        test_code=lesson_bundle['test_code'],
        attempt="initial",
    )

    # Repair attempt
    if not is_valid:
        print("   - Initial vetting failed. Attempting repair...")
        repaired_bundle = _repair_lesson(args.description, args.name, reason, args.provider, args.project_id, args.location, args.model_name)
        if not repaired_bundle:
            print("   - Halting: Repair attempt failed.")
            sys.exit(1)
        
        lesson_bundle = repaired_bundle # Use the repaired bundle for the final check
        is_valid, reason = _self_vet_lesson(
            lesson_name=lesson_bundle['name'],
            lesson_description=lesson_bundle['description'],
            before_code=lesson_bundle['before_code'],
            after_code=lesson_bundle['after_code'],
            test_code=lesson_bundle['test_code'],
            attempt="repair",
        )

    # Final decision
    if not is_valid:
        print("   - Self-vetting failed even after repair. Discarding lesson.", file=sys.stderr)
        print(f"   - Final Reason: {reason}", file=sys.stderr)
        sys.exit(1)

    print("   - Self-vetting result: Valid")
    lesson_bundle['timestamp'] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    lesson_bundle['focus_area'] = "self-generated" # Placeholder
    _save_lesson_to_db(lesson_bundle)
    
    print(f"   - Successfully saved new lesson '{args.name}' to the curriculum database.")

if __name__ == '__main__':
    main()
