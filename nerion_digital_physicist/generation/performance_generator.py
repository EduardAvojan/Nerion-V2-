"""
This module generates performance optimization lessons for the Digital Physicist.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.parent.coder import Coder
from nerion_digital_physicist.db.curriculum_store import CurriculumStore

def _generate_performance_optimization(name: str, description: str, provider: str | None = None) -> dict[str, str] | None:
    """Generates a performance optimization lesson using an LLM."""
    try:
        llm = Coder(role='coder')
    except Exception as e:
        print(f"  - ERROR: Could not get LLM provider: {e}", file=sys.stderr)
        return None

    system_prompt = (
        "You are an expert Python programmer. Your task is to generate a code snippet with a known performance bottleneck, a pytest test that benchmarks the code's performance, and an optimized version of the code. "
        "The response must be a JSON object with three keys: 'inefficient_code', 'test_code', and 'optimized_code'."
    )
    user_prompt = f"Generate a performance optimization lesson for the following bottleneck: {description}"

    print(f"--- PROMPT ---\n{system_prompt}\n\n{user_prompt}\n--------------")

    try:
        response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
    except Exception as e:
        print(f"  - ERROR: Failed to request performance optimization from LLM: {e}", file=sys.stderr)
        return None

    if not response_json_str:
        print("  - WARNING: Performance optimization generation LLM returned an empty response.", file=sys.stderr)
        return None

    try:
        performance_optimization = json.loads(response_json_str)
    except json.JSONDecodeError as e:
        print(f"  - WARNING: Performance optimization generation JSON parse failed: {e}.", file=sys.stderr)
        return None

    if not all(k in performance_optimization for k in ['inefficient_code', 'test_code', 'optimized_code']):
        print("  - WARNING: LLM response missing required keys.", file=sys.stderr)
        return None

    return performance_optimization

def main():
    """Main function to generate and store a performance optimization lesson."""
    parser = argparse.ArgumentParser(description="Generate a performance optimization lesson.")
    parser.add_argument("--name", required=True, help="The name of the performance optimization lesson.")
    parser.add_argument("--description", required=True, help="A description of the bottleneck.")
    parser.add_argument("--provider", default=None, help="The LLM provider to use.")
    args = parser.parse_args()

    print(f"Generating performance optimization lesson: {args.name}")

    performance_optimization = _generate_performance_optimization(args.name, args.description, args.provider)

    if performance_optimization:
        store = CurriculumStore(Path("out/learning/curriculum.sqlite"))
        performance_data = {
            "name": args.name,
            "description": args.description,
            "focus_area": "performance_optimization",
            "inefficient_code": performance_optimization["inefficient_code"],
            "test_code": performance_optimization["test_code"],
            "optimized_code": performance_optimization["optimized_code"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        store.add_performance_optimization(performance_data)
        store.close()
        print(f"Successfully generated and stored performance optimization lesson: {args.name}")
    else:
        print(f"Failed to generate performance optimization lesson: {args.name}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
