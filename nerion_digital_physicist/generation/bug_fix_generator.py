"""
This module generates bug-fixing lessons for the Digital Physicist.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.parent.coder import Coder
from nerion_digital_physicist.db.curriculum_store import CurriculumStore

def _generate_bug_fix(name: str, description: str, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> dict[str, str] | None:
    """Generates a bug-fixing lesson using an LLM."""
    try:
        llm = Coder(role='coder', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
    except Exception as e:
        print(f"  - ERROR: Could not get LLM provider: {e}")
        return None

    system_prompt = (
        "You are an expert Python programmer. Your task is to generate a code snippet with a common off-by-one error, a pytest test that exposes the bug, and the fixed version of the code. "
        "The response must be a JSON object with three keys: 'buggy_code', 'test_code', and 'fixed_code'."
    )
    user_prompt = f"Generate a bug-fixing lesson for the following bug: {description}"

    print(f"--- PROMPT ---\n{system_prompt}\n\n{user_prompt}\n--------------")

    try:
        response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
    except Exception as e:
        print(f"  - ERROR: Failed to request bug fix from LLM: {e}")
        return None

    if not response_json_str:
        print("  - WARNING: Bug fix generation LLM returned an empty response.")
        return None

    try:
        bug_fix = json.loads(response_json_str)
    except json.JSONDecodeError as e:
        print(f"  - WARNING: Bug fix generation JSON parse failed: {e}.")
        return None

    if not all(k in bug_fix for k in ['buggy_code', 'test_code', 'fixed_code']):
        print("  - WARNING: LLM response missing required keys.")
        return None

    return bug_fix

def main():
    """Main function to generate and store a bug-fixing lesson."""
    parser = argparse.ArgumentParser(description="Generate a bug-fixing lesson.")
    parser.add_argument("--name", required=True, help="The name of the bug-fixing lesson.")
    parser.add_argument("--description", required=True, help="A description of the bug.")
    parser.add_argument("--provider", default=None, help="The LLM provider to use.")
    parser.add_argument("--project-id", type=str, default=None, help="Google Cloud Project ID for Vertex AI.")
    parser.add_argument("--location", type=str, default=None, help="Google Cloud location for Vertex AI (e.g., 'us-central1').")
    parser.add_argument("--model-name", type=str, default=None, help="Vertex AI model name to use (e.g., 'gemini-pro').")
    args = parser.parse_args()

    print(f"Generating bug-fixing lesson: {args.name}")

    bug_fix = _generate_bug_fix(args.name, args.description, args.provider, args.project_id, args.location, args.model_name)

    if bug_fix:
        store = CurriculumStore(Path("out/learning/curriculum.sqlite"))
        bug_fix_data = {
            "name": args.name,
            "description": args.description,
            "focus_area": "bug_fixing",
            "before_code": bug_fix["buggy_code"],
            "after_code": bug_fix["fixed_code"],
            "test_code": bug_fix["test_code"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        store.add_bug_fix(bug_fix_data)
        store.close()
        print(f"Successfully generated and stored bug-fixing lesson: {args.name}")
    else:
        print(f"Failed to generate bug-fixing lesson: {args.name}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()