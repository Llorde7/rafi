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

class SessionResponse(BaseModel):
    session_id: UUID
    user_id: Optional[str]
    language: str
    created_at: datetime


class TriggerSpan(BaseModel):
    span: str
    emotion: str
    weight: float


class CausalAnalysis(BaseModel):
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


class TurnResponse(BaseModel):
    turn_id: UUID
    session_id: UUID
    text: str
    translation: Optional[str]
    top_3: list[EmotionScore]
    reasoning: str
    causal_analysis: Optional[CausalAnalysis] = None
    created_at: datetime


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
