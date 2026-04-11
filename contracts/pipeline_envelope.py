from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from datetime import datetime

from contracts.classifier_contract import ClassifierOutput
from contracts.causal_contract import CausalOutput
from contracts.planner_contract import PlannerOutput
from contracts.trace_contract import TraceOutput, TonePreference, DetectedLanguage
from contracts.trajectory_contract import TrajectoryFlag


class PipelineEnvelope(BaseModel):
    turn_id: Optional[UUID]       = None
    session_id: Optional[str]     = None
    user_id: Optional[str]        = None
    timestamp: datetime           = datetime.utcnow()
    raw_text: str

    # ── Session-level context (set once at session create) ────────────────────
    tone_preference:   TonePreference   = TonePreference.FRIENDLY
    detected_language: DetectedLanguage = DetectedLanguage.ENGLISH

    # ── Agent outputs ─────────────────────────────────────────────────────────
    classifier_output: Optional[ClassifierOutput] = None
    causal_output:     Optional[CausalOutput]     = None
    planner_output:    Optional[PlannerOutput]    = None
    trace_output:      Optional[TraceOutput]      = None

    # ── Orchestrator signals ─────────────────────────────────────────────────
    escalation_flag: Optional[TrajectoryFlag] = None