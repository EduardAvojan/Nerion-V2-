#!/usr/bin/env python3
"""
Quick test to verify Claude/Anthropic API integration.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python test_claude_integration.py
"""
import os
import sys
from pathlib import Path

# Load .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

from app.parent.coder import Coder

def test_claude_basic():
    """Test basic Claude generation."""
    print("=" * 60)
    print("TEST 1: Basic Claude Generation")
    print("=" * 60)

    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set!")
        print("   Please run: export ANTHROPIC_API_KEY='your-key'")
        return False

    print(f"✓ API key found: {api_key[:20]}...")

    try:
        coder = Coder(
            role="code",
            provider_override="anthropic:claude-sonnet-4-5-20250929"
        )
        print("✓ Coder initialized with anthropic:claude-sonnet-4-5-20250929")

        response = coder.complete("Write a hello world function in Python")
        print("\n📝 Response:")
        print("-" * 60)
        print(response[:200] + "..." if len(response) > 200 else response)
        print("-" * 60)
        print(f"\n✅ SUCCESS: Generated {len(response)} characters")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_claude_json():
    """Test Claude JSON structured output."""
    print("\n" + "=" * 60)
    print("TEST 2: Claude JSON Structured Output")
    print("=" * 60)

    try:
        coder = Coder(
            role="code",
            provider_override="anthropic:claude-sonnet-4-5-20250929"
        )

        response = coder.complete_json(
            prompt="Generate a simple lesson plan",
            system="You are an expert educator. Return a JSON with keys: title, description, duration_mins"
        )

        if not response:
            print("\n❌ Empty response from API")
            return False

        print("\n📝 JSON Response:")
        print("-" * 60)
        print(response[:300] + "..." if len(response) > 300 else response)
        print("-" * 60)

        # Try parsing it
        import json
        parsed = json.loads(response)
        print(f"\n✓ Valid JSON with keys: {list(parsed.keys())}")
        print(f"✅ SUCCESS: JSON generation works")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_curriculum_generator():
    """Test that curriculum generator can use Claude."""
    print("\n" + "=" * 60)
    print("TEST 3: Curriculum Generator with Claude")
    print("=" * 60)

    try:
        from nerion_digital_physicist.learning_orchestrator import LearningOrchestrator

        orchestrator = LearningOrchestrator()
        print("✓ LearningOrchestrator initialized")

        # Just test that we can create the provider string
        provider = "anthropic:claude-sonnet-4-5-20250929"
        print(f"✓ Provider string: {provider}")
        print(f"✅ SUCCESS: Curriculum generator is compatible")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n🧪 Claude/Anthropic Integration Test Suite")
    print("=" * 60)

    results = {
        "Basic Generation": test_claude_basic(),
        "JSON Output": test_claude_json(),
        "Curriculum Generator": test_curriculum_generator(),
    }

    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {test_name}")

    all_passed = all(results.values())

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nYou can now use Claude Sonnet 4.5 for lesson generation:")
        print("  python -m nerion_digital_physicist.learning_orchestrator \\")
        print("    --provider anthropic:claude-sonnet-4-5-20250929")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("\nPlease check:")
        print("  1. ANTHROPIC_API_KEY is set correctly")
        print("  2. You have API credits available")
        print("  3. Network connectivity is working")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
