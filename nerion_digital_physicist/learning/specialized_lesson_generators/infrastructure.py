"""
C1-Level Specialized Lesson Generators - Infrastructure.

This module contains specialized generator methods for deployment, observability, and messaging.
"""
import json
from typing import Dict, Any, Optional

from app.parent.coder import Coder
from .cefr_prompts import CEFR_PROMPTS


class InfrastructureGenerators:
    """Generators for Infrastructure."""

    def _fallback_idea(self, inspiration: str) -> Dict[str, Any]:
        """Generate an offline fallback idea when LLM is unavailable."""
        base_slug = "_".join(part for part in inspiration.lower().split() if part)
        slug = base_slug or "core_systems"
        idea = {
            "name": f"offline_{slug}_hardening",
            "description": f"Develop automated drills to strengthen {inspiration} safeguards across the stack.",
            "source": inspiration,
        }
        print(f"  - Using offline fallback idea: {idea['name']}")
        return idea

    def _generate_deployment_idea(self, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Optional[Dict[str, Any]]:
        """Generates a deployment/CI-CD lesson idea."""
        print(f"[Idea Generation] Generating deployment/CI-CD concept...")

        try:
            llm = Coder(role='planner', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider: {e}")
            return None

        system_prompt = (
            "You are an expert DevOps engineer and curriculum designer. Your task is to devise a HIGH-IMPACT deployment/CI-CD lesson for an AI agent. "
            "Focus on deployment strategies that minimize downtime and risk. "
            "The lesson concept must be returned as a JSON object with two keys: 'name' (a short, snake_case identifier) and 'description' (a one-sentence explanation)."
        )
        user_prompt = (
            "Propose a HIGH-IMPACT deployment/CI-CD lesson. Examples:\n"
            "- Blue-green deployment for zero downtime\n"
            "- Canary releases with gradual rollout\n"
            "- Database migration with backward compatibility\n"
            "- Automated rollback on health check failure\n"
            "- Feature flags for dark launches\n"
            "- Container orchestration with Kubernetes\n"
            "Generate a similarly impactful deployment concept."
        )

        try:
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - ERROR: Failed to request idea from LLM: {e}")
            return self._fallback_idea("deployment_cicd")

        if not response_json_str:
            print("  - WARNING: Idea generation LLM returned an empty response. Using offline fallback.")
            return self._fallback_idea("deployment_cicd")

        try:
            idea = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"  - WARNING: Idea generation JSON parse failed: {e}. Using offline fallback.")
            return self._fallback_idea("deployment_cicd")

        idea['source'] = "deployment_cicd"
        if not all(k in idea for k in ['name', 'description']):
            print("  - WARNING: LLM response missing required keys. Using offline fallback.")
            return self._fallback_idea("deployment_cicd")

        print(f"  - Generated deployment idea: {idea['name']}")
        return idea

    def _generate_observability_idea(self, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Optional[Dict[str, Any]]:
        """Generates an observability/monitoring lesson idea."""
        print(f"[Idea Generation] Generating observability concept...")

        try:
            llm = Coder(role='planner', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider: {e}")
            return None

        system_prompt = (
            "You are an expert SRE/DevOps engineer and curriculum designer. Your task is to devise a HIGH-IMPACT observability lesson for an AI agent. "
            "Focus on monitoring, logging, and tracing patterns that enable rapid debugging and incident response. "
            "The lesson concept must be returned as a JSON object with two keys: 'name' (a short, snake_case identifier) and 'description' (a one-sentence explanation)."
        )
        user_prompt = (
            "Propose a HIGH-IMPACT observability lesson. Examples:\n"
            "- Structured logging with correlation IDs\n"
            "- Distributed tracing across microservices\n"
            "- RED metrics (Rate, Errors, Duration) for APIs\n"
            "- Setting alert thresholds with SLOs/SLIs\n"
            "- Log aggregation and centralized monitoring\n"
            "- Performance profiling and flame graphs\n"
            "Generate a similarly critical observability concept."
        )

        try:
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - ERROR: Failed to request idea from LLM: {e}")
            return self._fallback_idea("observability")

        if not response_json_str:
            print("  - WARNING: Idea generation LLM returned an empty response. Using offline fallback.")
            return self._fallback_idea("observability")

        try:
            idea = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"  - WARNING: Idea generation JSON parse failed: {e}. Using offline fallback.")
            return self._fallback_idea("observability")

        idea['source'] = "observability"
        if not all(k in idea for k in ['name', 'description']):
            print("  - WARNING: LLM response missing required keys. Using offline fallback.")
            return self._fallback_idea("observability")

        print(f"  - Generated observability idea: {idea['name']}")
        return idea

    def _generate_message_queue_idea(self, provider: str | None = None, project_id: str | None = None, location: str | None = None, model_name: str | None = None) -> Optional[Dict[str, Any]]:
        """Generates a message queue/event-driven lesson idea."""
        print(f"[Idea Generation] Generating message queue concept...")

        try:
            llm = Coder(role='planner', provider_override=provider, project_id=project_id, location=location, model_name=model_name)
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider: {e}")
            return None

        system_prompt = (
            "You are an expert distributed systems architect and curriculum designer. Your task is to devise a HIGH-IMPACT message queue lesson for an AI agent. "
            "Focus on async messaging patterns that improve scalability and reliability. "
            "The lesson concept must be returned as a JSON object with two keys: 'name' (a short, snake_case identifier) and 'description' (a one-sentence explanation)."
        )
        user_prompt = (
            "Propose a HIGH-IMPACT message queue lesson. Examples:\n"
            "- At-least-once vs exactly-once delivery guarantees\n"
            "- Dead letter queue for failed messages\n"
            "- Event sourcing with append-only log\n"
            "- Pub/sub pattern for decoupled services\n"
            "- Message ordering and partitioning strategies\n"
            "- Idempotent message processing\n"
            "Generate a similarly critical message queue concept."
        )

        try:
            response_json_str = llm.complete_json(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - ERROR: Failed to request idea from LLM: {e}")
            return self._fallback_idea("message_queues")

        if not response_json_str:
            print("  - WARNING: Idea generation LLM returned an empty response. Using offline fallback.")
            return self._fallback_idea("message_queues")

        try:
            idea = json.loads(response_json_str)
        except json.JSONDecodeError as e:
            print(f"  - WARNING: Idea generation JSON parse failed: {e}. Using offline fallback.")
            return self._fallback_idea("message_queues")

        idea['source'] = "message_queues"
        if not all(k in idea for k in ['name', 'description']):
            print("  - WARNING: LLM response missing required keys. Using offline fallback.")
            return self._fallback_idea("message_queues")

        print(f"  - Generated message queue idea: {idea['name']}")
        return idea

