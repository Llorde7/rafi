from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from datetime import datetime


class EmotionScore(BaseModel):
    emotion: str
    confidence: float


class ClassifyRequest(BaseModel):
    text: str
    user_id: Optional[str] = None
    session_id: Optional[UUID] = None

class CreateSessionRequest(BaseModel):
    user_id: Optional[str] = None
    language: str = "en"
    tone_preference: str = "friendly"   # "friendly" | "clinical"

class SessionResponse(BaseModel):
    session_id: UUID
    user_id: Optional[str]
    language: str
    tone_preference: str
    created_at: datetime


class TriggerSpan(BaseModel):
    span: str
    emotion: str
    weight: float


class CausalAnalysis(BaseModel):
    """Matches contracts.causal_contract.CausalOutput."""
    confidence_score: float
    confidence_category: str
    trigger_spans: list[TriggerSpan]
    global_cause: str
    causal_chain: list[str]
    temporal_pattern: Optional[str] = None
    cause_type: str
    cognitive_pattern: str = ""
    behavioral_risk: str = ""
    clarifying_question: Optional[str] = None
    planner_instruction: str
    error: Optional[str] = None


class PlannerTechnique(BaseModel):
    name: str
    modality: str
    purpose: str
    sequence_note: str


class PlannerTechniqueCluster(BaseModel):
    techniques: list[PlannerTechnique]
    cluster_rationale: str
    executor_instruction: str
    rag_context: Optional[str] = None
    kb_sources: list[str] = []


class PlannerResult(BaseModel):
    """Matches contracts.planner_contract.PlannerOutput."""
    framework: str
    intent_state_received: str = ""
    technique_cluster: Optional[PlannerTechniqueCluster] = None
    clarifying_question: Optional[str] = None
    clarifying_question_overridden: bool = False
    response_directive: str = ""
    escalate_to_safety: bool = False
    escalation_reason: Optional[str] = None
    kb_retrieval_attempted: bool = False
    error: Optional[str] = None


class TraceResult(BaseModel):
    response_text:         str
    strategy_used:         str
    language:              str
    contains_clarifying_q: bool = False
    trace_confidence:      str
    error:                 Optional[str] = None


class TurnResponse(BaseModel):
    turn_id: Optional[UUID] = None
    session_id: UUID
    text: Optional[str] = None
    translation: Optional[str]
    top_3: list[EmotionScore]
    reasoning: str
    causal_analysis: Optional[CausalAnalysis] = None
    planner_output: Optional[PlannerResult] = None
    trace_output: Optional[TraceResult] = None
    created_at: Optional[datetime] = None
    response_text: Optional[str] = None


class CausalRequest(BaseModel):
    session_id: UUID
    text: str
    top_3: list[EmotionScore]
    reasoning: Optional[str] = ""


class CausalResponse(BaseModel):
    session_id: UUID
    text: str
    confidence_score: float
    confidence_category: str
    trigger_spans: list[TriggerSpan]
    global_cause: str
    causal_chain: list[str]
    temporal_pattern: Optional[str] = None
    cause_type: str
    clarifying_question: Optional[str] = None
    planner_instruction: str
    error: Optional[str] = None


class SessionHistoryResponse(BaseModel):
    session_id: UUID
    language: str
    created_at: datetime
    turns: list[TurnResponse]