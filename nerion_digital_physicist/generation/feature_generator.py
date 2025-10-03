"""
This module generates feature implementation lessons for the Digital Physicist.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.parent.coder import Coder
from nerion_digital_physicist.db.curriculum_store import CurriculumStore

def _generate_feature_implementation(name: str, description: str, provider: str | None = None) -> dict[str, str] | None:
    """Generates a feature implementation lesson using an LLM."""
    try:
        llm = Coder(role='coder')
    except Exception as e:
        print(f"  - ERROR: Could not get LLM provider: {e}", file=sys.stderr)
        return None

    system_prompt = (
        "You are an expert Python programmer. Your task is to generate a code snippet for a new feature, a pytest test that specifies the new feature, and the final version of the code with the feature implemented. "
        "The response must be a JSON object with three keys: 'initial_code', 'test_code', and 'final_code'."
    )
    user_prompt = f"Generate a feature implementation lesson for the following feature: {description}"

    print(f"--- PROMPT ---\n{system_prompt}\n\n{user_prompt}\n--------------")

    try:
        response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
    except Exception as e:
        print(f"  - ERROR: Failed to request feature implementation from LLM: {e}", file=sys.stderr)
        return None

    if not response_json_str:
        print("  - WARNING: Feature implementation generation LLM returned an empty response.", file=sys.stderr)
        return None

    try:
        feature_implementation = json.loads(response_json_str)
    except json.JSONDecodeError as e:
        print(f"  - WARNING: Feature implementation generation JSON parse failed: {e}.", file=sys.stderr)
        return None

    if not all(k in feature_implementation for k in ['initial_code', 'test_code', 'final_code']):
        print("  - WARNING: LLM response missing required keys.", file=sys.stderr)
        return None

    return feature_implementation

def main():
    """Main function to generate and store a feature implementation lesson."""
    parser = argparse.ArgumentParser(description="Generate a feature implementation lesson.")
    parser.add_argument("--name", required=True, help="The name of the feature implementation lesson.")
    parser.add_argument("--description", required=True, help="A description of the feature.")
    parser.add_argument("--provider", default=None, help="The LLM provider to use.")
    args = parser.parse_args()

    print(f"Generating feature implementation lesson: {args.name}")

    feature_implementation = _generate_feature_implementation(args.name, args.description, args.provider)

    if feature_implementation:
        store = CurriculumStore(Path("out/learning/curriculum.sqlite"))
        feature_data = {
            "name": args.name,
            "description": args.description,
            "focus_area": "feature_implementation",
            "initial_code": feature_implementation["initial_code"],
            "test_code": feature_implementation["test_code"],
            "final_code": feature_implementation["final_code"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        store.add_feature_implementation(feature_data)
        store.close()
        print(f"Successfully generated and stored feature implementation lesson: {args.name}")
    else:
        print(f"Failed to generate feature implementation lesson: {args.name}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
