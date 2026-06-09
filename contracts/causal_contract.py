from enum import Enum
from typing import Optional
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Existing enums — unchanged
# ---------------------------------------------------------------------------

class ConfidenceCategory(str, Enum):
    confident     = "confident"
    partial       = "partial"
    insufficient  = "insufficient"
    contradictory = "contradictory"


class CauseType(str, Enum):
    cognitive_distortion   = "cognitive_distortion"
    avoidance_behaviour    = "avoidance_behaviour"
    unresolved_loss        = "unresolved_loss"
    somatic_response       = "somatic_response"
    interpersonal_conflict = "interpersonal_conflict"
    identity_threat        = "identity_threat"
    ambiguous              = "ambiguous"


class PlannerInstruction(str, Enum):
    proceed    = "proceed"
    ask_first  = "ask_first"
    hold       = "hold"


# ---------------------------------------------------------------------------
# New enums — added this session
# ---------------------------------------------------------------------------

class CognitivePattern(str, Enum):
    catastrophising           = "catastrophising"
    all_or_nothing            = "all_or_nothing"
    mind_reading              = "mind_reading"
    personalisation           = "personalisation"
    should_statements         = "should_statements"
    emotional_reasoning       = "emotional_reasoning"
    avoidance_rationalisation = "avoidance_rationalisation"
    none_identified           = "none_identified"


class BehavioralRisk(str, Enum):
    shutdown_paralysis = "shutdown_paralysis"
    avoidance          = "avoidance"
    rumination_loop    = "rumination_loop"
    impulsive_exit     = "impulsive_exit"
    none_identified    = "none_identified"


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------

class TriggerSpan(BaseModel):
    span:    str    # verbatim substring of user text — hallucinated spans dropped in validation
    emotion: str
    weight:  float


class HistoryTurn(BaseModel):
    turn_index:  int = 0
    text:        str
    top_emotion: str
    cause_type:  str
    valence:     float = 0.0


class CausalInput(BaseModel):
    text:             str
    top_emotions:     list[dict]          # from Stage1Output — no classifier import
    reasoning:        str
    situation_type:   str                 # NEW — passed from Stage1Output
    session_history:  list[HistoryTurn]


class CausalOutput(BaseModel):
    confidence_score:      float
    confidence_category:   ConfidenceCategory
    trigger_spans:         list[TriggerSpan]
    global_cause:          str
    causal_chain:          list[str]
    temporal_pattern:      Optional[str]
    cause_type:            CauseType
    cognitive_pattern:     CognitivePattern   # NEW
    behavioral_risk:       BehavioralRisk     # NEW
    clarifying_question:   Optional[str]
    planner_instruction:   PlannerInstruction