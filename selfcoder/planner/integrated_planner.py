"""
Integrated Planner with Chain-of-Thought Reasoning 🧠

This script wires up the dormant reasoning components:
1.  ChainOfThoughtReasoner -> Structured thinking process
2.  ExplainablePlanner -> Plan generation with justification

It demonstrates how the system "thinks" before it acts.
"""
import argparse
import logging
from typing import List, Dict, Any

from selfcoder.reasoning.chain_of_thought import ChainOfThoughtReasoner, ReasoningStep
from selfcoder.planner.explainable_planner import ExplainablePlanner, ExplainablePlan
from selfcoder.planner.planner import plan_edits_from_nl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ReasoningLoop")

class MockBasePlanner:
    """Mock base planner to simulate the underlying planning logic."""
    def plan(self, instruction: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        # In a real scenario, this would call the LLM or heuristic planner
        # For this integration test, we return a dummy plan
        return [
            {"action": "edit", "file": "target_file.py", "description": f"Implement {instruction}"}
        ]

def run_reasoning_loop(instruction: str, context_files: List[str]):
    logger.info(f"🤔 Received Instruction: '{instruction}'")
    
    # 1. Initialize Components
    # We wrap a mock base planner for this demonstration
    # In production, this would wrap the actual `plan_edits_from_nl` function
    base_planner = MockBasePlanner()
    
    planner = ExplainablePlanner(
        base_planner=base_planner,
        min_confidence_for_execution=0.75
    )
    
    logger.info("✓ Reasoning Engine Initialized")
    
    # 2. Generate Plan with Reasoning
    logger.info("Generating explainable plan...")
    
    # Create a dummy context
    context = {
        "files": context_files,
        "repository_state": "clean"
    }
    
    # Note: ExplainablePlanner.create_plan expects (task, context)
    # and returns an ExplainablePlan object
    plan: ExplainablePlan = planner.create_plan(instruction, context)
    
    # 3. Output Results
    logger.info("\n" + "="*40)
    logger.info("🧠 REASONING TRACE")
    logger.info("="*40)
    
    if plan.reasoning and plan.reasoning.reasoning_chain:
        for step in plan.reasoning.reasoning_chain:
            logger.info(f"[{step.step.name}] {step.content}")
            logger.info(f"  Confidence: {step.confidence:.2f}")
    else:
        logger.warning("No reasoning trace generated (Mock mode might be active)")
        
    logger.info("\n" + "="*40)
    logger.info("📋 GENERATED PLAN")
    logger.info("="*40)
    
    for action in plan.actions:
        logger.info(f"- {action.get('type', 'unknown')} {action.get('file', 'unknown')}: {action.get('reason', '')}")
        
    logger.info("\n" + "="*40)
    logger.info("🛡️ SAFETY ASSESSMENT")
    logger.info("="*40)
    logger.info(f"Risk Level: {plan.estimated_risk}")
    logger.info(f"Requires Human Review: {plan.requires_human_review}")
    logger.info(f"Explanation: {plan.reasoning.user_explanation}")

def main():
    parser = argparse.ArgumentParser(description="Run Integrated Reasoning Loop")
    parser.add_argument("instruction", type=str, help="Natural language instruction")
    parser.add_argument("--files", nargs="+", default=["main.py"], help="Context files")
    args = parser.parse_args()
    
    run_reasoning_loop(args.instruction, args.files)

if __name__ == "__main__":
    main()
