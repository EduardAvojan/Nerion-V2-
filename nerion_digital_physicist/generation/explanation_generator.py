"""
This module generates code explanation lessons for the Digital Physicist.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.parent.coder import Coder
from nerion_digital_physicist.db.curriculum_store import CurriculumStore

def _generate_code_explanation(name: str, description: str, provider: str | None = None) -> dict[str, str] | None:
    """Generates a code explanation lesson using an LLM."""
    try:
        llm = Coder(role='coder')
    except Exception as e:
        print(f"  - ERROR: Could not get LLM provider: {e}", file=sys.stderr)
        return None

    system_prompt = (
        "You are an expert Python programmer. Your task is to generate a code snippet and a concise, accurate explanation of what it does. "
        "The response must be a JSON object with two keys: 'code_snippet' and 'explanation'."
    )
    user_prompt = f"Generate a code explanation lesson for the following concept: {description}"

    print(f"--- PROMPT ---\n{system_prompt}\n\n{user_prompt}\n--------------")

    try:
        response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
    except Exception as e:
        print(f"  - ERROR: Failed to request code explanation from LLM: {e}", file=sys.stderr)
        return None

    if not response_json_str:
        print("  - WARNING: Code explanation generation LLM returned an empty response.", file=sys.stderr)
        return None

    try:
        code_explanation = json.loads(response_json_str)
    except json.JSONDecodeError as e:
        print(f"  - WARNING: Code explanation generation JSON parse failed: {e}.", file=sys.stderr)
        return None

    if not all(k in code_explanation for k in ['code_snippet', 'explanation']):
        print("  - WARNING: LLM response missing required keys.", file=sys.stderr)
        return None

    return code_explanation

def main():
    """Main function to generate and store a code explanation lesson."""
    parser = argparse.ArgumentParser(description="Generate a code explanation lesson.")
    parser.add_argument("--name", required=True, help="The name of the code explanation lesson.")
    parser.add_argument("--description", required=True, help="A description of the concept.")
    parser.add_argument("--provider", default=None, help="The LLM provider to use.")
    args = parser.parse_args()

    print(f"Generating code explanation lesson: {args.name}")

    code_explanation = _generate_code_explanation(args.name, args.description, args.provider)

    if code_explanation:
        store = CurriculumStore(Path("out/learning/curriculum.sqlite"))
        explanation_data = {
            "name": args.name,
            "description": args.description,
            "focus_area": "code_comprehension",
            "code_snippet": code_explanation["code_snippet"],
            "explanation": code_explanation["explanation"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        store.add_code_explanation(explanation_data)
        store.close()
        print(f"Successfully generated and stored code explanation lesson: {args.name}")
    else:
        print(f"Failed to generate code explanation lesson: {args.name}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
