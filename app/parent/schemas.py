

from __future__ import annotations
from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field, model_validator

# One step in the Parent's plan
class Step(BaseModel):
    """
    Represents a single step in the Parent's plan for the Child to execute.
    """
    action: Literal["tool_call", "ask_user", "respond"]
    tool: Optional[str] = Field(default=None, description="Name of tool to call if action=tool_call")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool call")
    summary: Optional[str] = Field(default=None, description="One-line human-readable summary of the step")
    # Optional execution controls
    continue_on_error: bool = Field(default=False, description="If true, record the error and continue to next step")
    timeout_s: Optional[float] = Field(default=None, description="Optional timeout (seconds) for this step")
    retry: int = Field(default=0, ge=0, le=5, description="Optional retry count on failure (no retry by default)")
    # Optional rationale and cost hinting
    why: Optional[str] = Field(default=None, description="Short rationale for selecting this tool/step")
    cost_hint_ms: Optional[int] = Field(default=None, description="Estimated latency/cost in milliseconds for this step")

    @model_validator(mode="after")
    def _require_tool_for_tool_call(self):
        if self.action == "tool_call" and not self.tool:
            raise ValueError("tool is required when action=tool_call")
        return self

# Parent's full decision object
class ParentDecision(BaseModel):
    """
    Full decision from the Parent, including intent, plan, and optional final response.
    """
    intent: str = Field(..., description="High-level intent label (e.g., web.search, code.refactor)")
    plan: List[Step] = Field(default_factory=list, description="Ordered list of steps the child should execute")
    final_response: Optional[str] = Field(default=None, description="If action=respond, the text to return to the user")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Parent confidence in the plan")
    requires_network: bool = Field(False, description="Whether this plan requires internet access")
    notes: Optional[str] = Field(default=None, description="Rationale, caveats, extra guidance")

    @model_validator(mode="after")
    def _plan_or_final_response(self):
        if not self.final_response and not self.plan:
            raise ValueError("plan must contain at least one step when final_response is not provided")
        return self
