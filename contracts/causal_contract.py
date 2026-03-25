from pydantic import BaseModel
from typing import Optional
from enum import Enum


class ConfidenceCategory(str, Enum):
    CONFIDENT      = "confident"
    PARTIAL        = "partial"
    INSUFFICIENT   = "insufficient"
    CONTRADICTORY  = "contradictory"


class CauseType(str, Enum):
    COGNITIVE_DISTORTION   = "cognitive_distortion"
    AVOIDANCE_BEHAVIOUR    = "avoidance_behaviour"
    UNRESOLVED_LOSS        = "unresolved_loss"
    SOMATIC_RESPONSE       = "somatic_response"
    INTERPERSONAL_CONFLICT = "interpersonal_conflict"
    IDENTITY_THREAT        = "identity_threat"
    AMBIGUOUS              = "ambiguous"


class PlannerInstruction(str, Enum):
    PROCEED    = "proceed"
    ASK_FIRST  = "ask_first"
    HOLD       = "hold"


class TriggerSpan(BaseModel):
    span: str
    emotion: str
    weight: float


class HistoryTurn(BaseModel):
    text: str
    top_emotion: str
    confidence: float
    cause_type: Optional[str]        = None
    temporal_pattern: Optional[str]  = None


class CausalInput(BaseModel):
    text: str
    top_emotions: list[dict]
    reasoning: str
    session_history: list[HistoryTurn] = []
    trajectory_context: Optional[str] = None  # formatted summary from trajectory_engine


class CausalOutput(BaseModel):
    confidence_score: float
    confidence_category: ConfidenceCategory
    trigger_spans: list[TriggerSpan]
    global_cause: str
    causal_chain: list[str]
    temporal_pattern: Optional[str]
    cause_type: CauseType
    clarifying_question: Optional[str]
    planner_instruction: PlannerInstruction
    _error: Optional[str] = None