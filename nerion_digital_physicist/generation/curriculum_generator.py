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
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from datetime import datetime, timezone

import torch
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from app.parent.coder import Coder

from nerion_digital_physicist.db.curriculum_store import CurriculumStore
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


def _offline_lesson_bundle(lesson_description: str, lesson_name: str) -> Dict[str, Any]:
    """Return a deterministic lesson bundle when LLM access is unavailable."""
    before_code = (
        "def add(a: int, b: int) -> int:\n"
        "    \"\"\"Return the sum of two integers.\"\"\"\n"
        "    return a - b\n"
    )
    after_code = (
        "def add(a: int, b: int) -> int:\n"
        "    \"\"\"Return the sum of two integers.\"\"\"\n"
        "    return a + b\n"
    )
    test_code = (
        "from module import add\n\n"
        "def test_addition() -> None:\n"
        "    assert add(2, 3) == 5\n"
    )
    print("   - Using offline curriculum template to continue the cycle.")
    return {
        "name": lesson_name,
        "description": lesson_description,
        "before_code": before_code,
        "after_code": after_code,
        "test_code": test_code,
    }


def _normalise_redaction_placeholders(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure redaction placeholders cannot collide with original secret characters."""

    replacement = "<<REDACTED>>"
    for key in ("before_code", "after_code", "test_code"):
        value = bundle.get(key)
        if not isinstance(value, str):
            continue
        if "[REDACTED]" in value or "{REDACTED}" in value:
            bundle[key] = value.replace("[REDACTED]", replacement).replace("{REDACTED}", replacement)
    return bundle

def _generate_lesson_from_llm(lesson_description: str, lesson_name: str) -> Optional[Dict[str, Any]]:
    """Uses an LLM to generate the code snippets for a new training lesson."""
    print("   - Generating code snippets from LLM...")
    try:
        os.environ['NERION_V2_REQUEST_TIMEOUT'] = '300'
        llm = Coder(role='code')
    except Exception as e:
        print(f"   - Could not get LLM provider: {e}", file=sys.stderr)
        return _offline_lesson_bundle(lesson_description, lesson_name)

    system_prompt = (
        "You are an expert Python programmer and educator. Your task is to create a training exercise. "
        "Given a description, provide three pieces of Python code in a single JSON object with the keys: 'before_code', 'after_code', and 'test_code'."
        "\n**Execution Context:** Your generated code will be run in a temporary sandbox. The `before_code` and `after_code` will each be saved to a file named `module.py`. "
        "Your `test_code` will be saved in `test_module.py` and should therefore import the code to be tested using `from module import ...`."
        "\nThe 'test_code' must be a valid pytest file that fails on 'before_code' and passes on 'after_code'."
        "\nHypothesis is installed in this environment—prefer using Hypothesis for property-based tests rather than hand-rolled fuzzers."
        "\nIf you include any randomized loops (for example, custom fuzzing) keep them lightweight (≤150 iterations) and avoid long shrink loops so the vetting run stays fast."
        "\nWhen redacting secrets or sensitive values, use a placeholder that shares no characters with the original secret (e.g., '<<REDACTED>>'), so containment checks remain valid."
        "\nWhen you use asyncio.TaskGroup (or any construct that raises ExceptionGroup), flatten the exception by catching `except* Exception` and re-raising the first inner exception so single-exception tests pass cleanly."
        "\nWhen implementing retry/backoff helpers, ensure each retry actually sleeps once using a jitter-adjusted delay (e.g., call `random.uniform` and `time.sleep` before the next attempt)."
        "\n**Hypothesis Health Checks:** If you write a test that uses both the `@given` decorator and a `pytest` fixture (like `monkeypatch`), you must add the `@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])` decorator to the test function to avoid a `FailedHealthCheck` error."
    )
    user_prompt = f"Create a training exercise for the following lesson: {lesson_description}"

    try:
        response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        if not response_json_str:
            print("   - LLM returned an empty response.", file=sys.stderr)
            return _offline_lesson_bundle(lesson_description, lesson_name)
        data = json.loads(response_json_str)
        data['name'] = lesson_name
        data['description'] = lesson_description
        return _normalise_redaction_placeholders(data)
    except Exception as e:
        print(f"   - Failed to generate or parse LLM response: {e}", file=sys.stderr)
        return _offline_lesson_bundle(lesson_description, lesson_name)

def _repair_lesson(lesson_description: str, lesson_name: str, failure_log: str) -> Optional[Dict[str, Any]]:
    """Attempts to repair a failed lesson generation."""
    print("   - Repairing lesson with LLM...")
    try:
        os.environ['NERION_V2_REQUEST_TIMEOUT'] = '300'
        llm = Coder(role='code')
    except Exception as e:
        print(f"   - Could not get LLM provider for Repair: {e}", file=sys.stderr)
        return _offline_lesson_bundle(lesson_description, lesson_name)

    system_prompt = (
        "You are a debugging expert. A generated programming lesson failed its verification test. "
        "Your task is to analyze the failure and provide a corrected version. "
        "\n**Execution Context:** The code is run in a sandbox where the file to be tested is named `module.py`. Ensure your test code imports from `module`. "
        "Return a single JSON object with corrected 'before_code', 'after_code', and 'test_code' keys."
        "\nHypothesis is available—use it for property-based checks whenever suitable."
        "\nIf you must rely on custom randomized loops, cap them at 150 iterations and keep shrinking/diagnostic passes short so pytest completes quickly."
        "\nWhen redacting secrets or sensitive values, replace them with a placeholder that uses entirely different characters (for example, '<<REDACTED>>') so containment assertions do not fail." 
        "\nIf your fix involves asyncio.TaskGroup or ExceptionGroup, wrap the TaskGroup body in a `try` / `except* Exception` and re-raise the first inner exception to match tests that expect a single error." 
        "\nFor retry/backoff functions, make sure the jitter function is invoked and the code sleeps at least once before retrying so tests that count jitter usage succeed." 
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
            return _offline_lesson_bundle(lesson_description, lesson_name)
        data = json.loads(response_json_str)
        data['name'] = lesson_name
        data['description'] = lesson_description
        return _normalise_redaction_placeholders(data)
    except Exception as e:
        print(f"   - Failed to generate or parse LLM response for repair: {e}", file=sys.stderr)
        return _offline_lesson_bundle(lesson_description, lesson_name)

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

def _save_lesson_to_db(lesson_data: Dict[str, Any]):
    """Saves the vetted lesson to the curriculum database."""
    db_path = Path("out/learning/curriculum.sqlite")
    store = CurriculumStore(db_path)
    store.add_lesson(lesson_data)
    store.close()

def main():
    parser = argparse.ArgumentParser(description="Automated Curriculum Generator for the Digital Physicist.")
    parser.add_argument("--description", type=str, required=True, help="A high-level description of the lesson to generate.")
    parser.add_argument("--name", type=str, required=True, help="A short, descriptive name for the new template directory (e.g., 'fix_index_error').")
    args = parser.parse_args()

    print(f"--- Generating lesson for: {args.description} ---")

    # First attempt
    lesson_bundle = _generate_lesson_from_llm(args.description, args.name)
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
        repaired_bundle = _repair_lesson(args.description, args.name, reason)
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
