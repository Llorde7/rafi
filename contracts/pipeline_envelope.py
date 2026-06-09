from typing import Optional, Any
from pydantic import BaseModel


class PipelineEnvelope(BaseModel):
    """
    Complete output of one pipeline turn.
    All agent outputs stacked — used for persistence and API response assembly.
    """

    # Raw input
    text:               str
    detected_language:  str
    tone_preference:    str

    # Stage 1 — Fused classifier + situation
    classifier_output:  Any   # Stage1Output
    situation_type:     str   # NEW
    situation_summary:  str   # NEW

    # Stage 3 — Fused causal + cognitive pattern + behavioral risk
    causal_output:      Any   # CausalOutput
    cognitive_pattern:  str   # NEW
    behavioral_risk:    str   # NEW

    # Stage 4 — Intent (NEW)
    intent_state:       str   # processing | action_seeking | transitioning | crisis

    # Stage 6 — Planner
    planner_output:     Any   # PlannerOutput (contains TechniqueCluster)

    # Stage 8 — TRACE
    trace_output:       Optional[Any]  # TraceOutput | None if escalated

    # Convenience accessors
    @property
    def response_text(self) -> Optional[str]:
        if self.trace_output:
            return self.trace_output.response_text
        return None

    @property
    def escalated(self) -> bool:
        return self.planner_output.escalate_to_safety if self.planner_output else False