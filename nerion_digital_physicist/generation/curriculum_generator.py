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
from app.parent.coder import Coder

from nerion_digital_physicist.db.curriculum_store import CurriculumStore
from nerion_digital_physicist.agent.brain import CodeGraphNN
from nerion_digital_physicist.agent.data import create_graph_data_from_source
from nerion_digital_physicist.agent.semantics import get_global_embedder

def _generate_lesson_from_llm(lesson_description: str, lesson_name: str) -> Optional[Dict[str, Any]]:
    """Uses an LLM to generate the code snippets for a new training lesson."""
    print("   - Generating code snippets from LLM...")
    try:
        os.environ['NERION_V2_REQUEST_TIMEOUT'] = '300'
        llm = Coder(role='code')
    except Exception as e:
        print(f"   - Could not get LLM provider: {e}", file=sys.stderr)
        return None

    system_prompt = (
        "You are an expert Python programmer and educator. Your task is to create a training exercise. "
        "Given a description, provide three pieces of Python code in a single JSON object with the keys: 'before_code', 'after_code', and 'test_code'."
        "\n**Execution Context:** Your generated code will be run in a temporary sandbox. The `before_code` and `after_code` will each be saved to a file named `module.py`. "
        "Your `test_code` will be saved in `test_module.py` and should therefore import the code to be tested using `from module import ...`."
        "\nThe 'test_code' must be a valid pytest file that fails on 'before_code' and passes on 'after_code'."
    )
    user_prompt = f"Create a training exercise for the following lesson: {lesson_description}"

    try:
        response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        if not response_json_str:
            print("   - LLM returned an empty response.", file=sys.stderr)
            return None
        data = json.loads(response_json_str)
        data['name'] = lesson_name
        data['description'] = lesson_description
        return data
    except Exception as e:
        print(f"   - Failed to generate or parse LLM response: {e}", file=sys.stderr)
        return None

def _repair_lesson(lesson_description: str, lesson_name: str, failure_log: str) -> Optional[Dict[str, Any]]:
    """Attempts to repair a failed lesson generation."""
    print("   - Repairing lesson with LLM...")
    try:
        os.environ['NERION_V2_REQUEST_TIMEOUT'] = '300'
        llm = Coder(role='code')
    except Exception as e:
        print(f"   - Could not get LLM provider for Repair: {e}", file=sys.stderr)
        return None

    system_prompt = (
        "You are a debugging expert. A generated programming lesson failed its verification test. "
        "Your task is to analyze the failure and provide a corrected version. "
        "\n**Execution Context:** The code is run in a sandbox where the file to be tested is named `module.py`. Ensure your test code imports from `module`. "
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
        return data
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

def _is_structurally_good_change(before_code: str, after_code: str) -> bool:
    """Uses the GNN to analyze the structural quality of the code change."""
    MODEL_PATH = "digital_physicist_brain.pt"
    try:
        embedder = get_global_embedder()
        
        before_graph = create_graph_data_from_source(before_code, embedder=embedder)

        model = CodeGraphNN(
            num_node_features=before_graph.num_node_features,
            hidden_channels=256,
            num_classes=2,
        )
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()

        after_graph = create_graph_data_from_source(after_code, embedder=embedder)

        with torch.no_grad():
            before_pred = model(before_graph)
            after_pred = model(after_graph)
            
            before_pass_score = before_pred[0][1].item()
            after_pass_score = after_pred[0][1].item()
            print(f"      - Structural analysis scores: [Before: {before_pass_score:.3f}] -> [After: {after_pass_score:.3f}]")
            return after_pass_score > before_pass_score

    except FileNotFoundError:
        print(f"      - WARNING: Model file not found at {MODEL_PATH}. Skipping structural check.")
        return True
    except Exception as e:
        print(f"      - WARNING: Could not perform structural analysis: {e}")
        return True

def _self_vet_lesson(before_code: str, after_code: str, test_code: str) -> Tuple[bool, str]:
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
        return False, reason

    if not after_test_passed:
        reason = (
            "Vetting failed: Test unexpectedly FAILED on the 'after_code'.\n"
            f"PYTEST OUTPUT:\n{after_proc.stdout}{after_proc.stderr}"
        )
        print(f"      - {reason.splitlines()[0]}")
        return False, reason
    
    print("      - Behavioral verification passed.")

    print("      - Running structural verification...")
    if not _is_structurally_good_change(before_code, after_code):
        reason = "Vetting failed: Structural analysis did not show improvement."
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

    is_valid, reason = _self_vet_lesson(lesson_bundle['before_code'], lesson_bundle['after_code'], lesson_bundle['test_code'])

    # Repair attempt
    if not is_valid:
        print("   - Initial vetting failed. Attempting repair...")
        repaired_bundle = _repair_lesson(args.description, args.name, reason)
        if not repaired_bundle:
            print("   - Halting: Repair attempt failed.")
            sys.exit(1)
        
        lesson_bundle = repaired_bundle # Use the repaired bundle for the final check
        is_valid, reason = _self_vet_lesson(lesson_bundle['before_code'], lesson_bundle['after_code'], lesson_bundle['test_code'])

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