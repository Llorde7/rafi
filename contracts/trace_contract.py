from enum import Enum
from typing import Optional
from pydantic import BaseModel
from contracts.planner_contract import TechniqueCluster


class TonePreference(str, Enum):
    friendly = "friendly"
    clinical = "clinical"


class DetectedLanguage(str, Enum):
    en      = "en"
    sw      = "sw"
    sheng   = "sheng"
    mixed   = "mixed"
    unknown = "unknown"


class TraceConfidence(str, Enum):
    high    = "high"
    medium  = "medium"
    low     = "low"
    fallback = "fallback"


class TraceTurn(BaseModel):
    student_message:  str
    trace_response:   str
    strategy_used:    str
    turn_index:       int


class TraceInput(BaseModel):
    text:                          str
    detected_language:             DetectedLanguage
    tone_preference:               TonePreference
    technique_cluster:             TechniqueCluster   # replaces strategy: str
    executor_instruction:          str                # unified instruction from planner
    global_cause:                  str
    trigger_spans:                 list[str]
    clarifying_question_from_cae:  Optional[str]
    intent_state:                  str                # processing | action_seeking | transitioning
    trace_history:                 list[TraceTurn]


class TraceOutput(BaseModel):
    response_text:        str
    strategy_used:        str   # primary technique name from cluster
    language:             str
    contains_clarifying_q: bool
    trace_confidence:     TraceConfidence
    error:                Optional[str]