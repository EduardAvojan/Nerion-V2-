"""
Multi-Agent Agency Loop 🤖🤝🤖

This script wires up the dormant agency components:
1.  MultiAgentCoordinator -> Orchestrates the swarm
2.  SpecialistAgents -> Domain experts (Python, Security, Performance)

It demonstrates how multiple agents collaborate to solve a task.
"""
import argparse
import logging
import asyncio
from typing import List, Dict, Any

from nerion_digital_physicist.agents.coordinator import MultiAgentCoordinator
from nerion_digital_physicist.agents.specialists import (
    PythonSpecialist,
    SecuritySpecialist,
    PerformanceSpecialist
)
from nerion_digital_physicist.agents.protocol import TaskRequest, AgentRole

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AgencyLoop")

def run_agency_loop(task_description: str):
    logger.info(f"🤖 Received Task: '{task_description}'")
    
    # 1. Initialize Coordinator
    coordinator = MultiAgentCoordinator()
    
    # 2. Register Specialists
    python_agent = PythonSpecialist(agent_id="agent_python_01")
    security_agent = SecuritySpecialist(agent_id="agent_sec_01")
    perf_agent = PerformanceSpecialist(agent_id="agent_perf_01")
    
    coordinator.register_agent(python_agent)
    coordinator.register_agent(security_agent)
    coordinator.register_agent(perf_agent)
    
    logger.info(f"✓ Swarm Initialized with {len(coordinator.registry.agents)} agents")
    
    # 3. Submit Task
    logger.info("Submitting task to swarm...")
    
    request = TaskRequest(
        task_id="task_001",
        task_type="audit",
        context={"description": task_description},
        requester_id="user"
    )
    
    # 4. Execute Task (Synchronous)
    # The coordinator will decompose the task and assign it to agents
    responses = coordinator.assign_task(request)
    
    # 5. Output Results
    logger.info("\n" + "="*40)
    logger.info("🏁 SWARM RESULT")
    logger.info("="*40)
    
    # Aggregate results
    aggregated = coordinator.aggregate_responses(responses)
    logger.info(f"Aggregated Success: {aggregated.success}")
    logger.info(f"Aggregated Confidence: {aggregated.confidence:.2f}")
    
    logger.info("\n" + "="*40)
    logger.info("🗣️ AGENT CONTRIBUTIONS")
    logger.info("="*40)
    
    for response in responses:
        logger.info(f"[{response.responder_id}]")
        logger.info(f"  Success: {response.success}")
        logger.info(f"  Result: {str(response.result)[:100]}...") # Truncate

def main():
    parser = argparse.ArgumentParser(description="Run Multi-Agent Agency Loop")
    parser.add_argument("task", type=str, help="Task description for the swarm")
    args = parser.parse_args()
    
    # Run sync loop
    run_agency_loop(args.task)

if __name__ == "__main__":
    main()
