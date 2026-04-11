"""
trace_contract.py
─────────────────
Contracts for TRACE — the empathetic response generator.

TRACE consumes PlannerOutput + session context and produces
the student-facing message.
"""

from pydantic import BaseModel
from typing import Optional
from enum import Enum


class TonePreference(str, Enum):
    FRIENDLY  = "friendly"   # warm, peer-like, approachable
    CLINICAL  = "clinical"   # measured, grounded, professional warmth


class TraceConfidence(str, Enum):
    HIGH   = "high"    # directive was clear, generation was unambiguous
    MEDIUM = "medium"  # directive had some ambiguity, best judgement applied
    LOW    = "low"     # directive was thin or contradictory, fallback applied


class DetectedLanguage(str, Enum):
    ENGLISH = "en"
    SWAHILI = "sw"


class TraceTurn(BaseModel):
    """
    A single turn of TRACE conversation history.
    Stored in Redis, passed into TRACE each call.
    Trimmed to last 6 turns (3 exchanges) before injection.
    """
    student_text:   str
    trace_response: str
    strategy_used:  str
    turn_number:    int


class TraceInput(BaseModel):
    # ── From planner ──────────────────────────────────────────────────────────
    response_directive:    str
    framework:             str                   # TherapeuticFramework.value
    strategy:              str                   # ResponseStrategy.value
    clarifying_question:   Optional[str]         # woven in when present
    kb_context:            Optional[str]         # psychoeducate only
    escalate_to_safety:    bool = False          # TRACE must not run if True

    # ── Student message (from envelope raw_text) ──────────────────────────────
    student_text: str

    # ── Session context ───────────────────────────────────────────────────────
    tone_preference:        TonePreference
    detected_language:      DetectedLanguage
    trace_history:          list[TraceTurn] = []      # last 6 turns max
    cross_session_summary:  Optional[str]  = None     # from UserEmotionalProfile


class TraceOutput(BaseModel):
    response_text:          str
    strategy_used:          str                  # what TRACE actually used — may differ from planner
    planner_strategy:       str                  # what the planner suggested
    strategy_overridden:    bool = False          # True if TRACE deviated
    override_reason:        Optional[str] = None  # why TRACE deviated
    language:               DetectedLanguage     # language TRACE responded in
    contains_clarifying_q:  bool = False         # True if question was woven in
    trace_confidence:       TraceConfidence = TraceConfidence.HIGH
    error:                  Optional[str]   = None