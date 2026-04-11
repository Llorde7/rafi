from pydantic import BaseModel
from typing import Optional
from enum import Enum


class TherapeuticFramework(str, Enum):
    CBT              = "cbt"
    MI               = "mi"
    SOLUTION_FOCUSED = "solution_focused"
    PERSON_CENTRED   = "person_centred"
    NONE             = "none"              # active listening only


class ResponseStrategy(str, Enum):
    VALIDATE         = "validate"
    REFLECT          = "reflect"
    REFRAME          = "reframe"
    PROBE            = "probe"
    PSYCHOEDUCATE    = "psychoeducate"
    ACTIVE_LISTEN    = "active_listen"
    MOTIVATE         = "motivate"
    SOLUTION_ELICIT  = "solution_elicit"


class PlannerConfidence(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


class PlannerInput(BaseModel):
    # ── From classifier ───────────────────────────────────────────────────────
    text: str
    top_emotion: str                    # GoEmotionLabel.value
    emotion_confidence: float
    top_3_emotions: list[dict]          # [{emotion: str, confidence: float}, ...]

    # ── From causal engine ────────────────────────────────────────────────────
    global_cause: str
    causal_chain: list[str]
    cause_type: str                     # CauseType.value
    causal_confidence_score: float
    causal_confidence_category: str     # ConfidenceCategory.value
    causal_planner_instruction: str     # PlannerInstruction.value — advisory only
    clarifying_question: Optional[str]

    # ── From trajectory ───────────────────────────────────────────────────────
    trajectory_flag: str                # TrajectoryFlag.value
    valence_direction: str              # ValenceDirection.value
    current_arousal: str                # ArousalLevel.value
    current_valence: float              # weighted valence from latest turn
    shift_events: list[dict]
    turn_count: int
    cross_session_baseline: Optional[float]

    # ── Session framework ─────────────────────────────────────────────────────
    session_framework: str              # TherapeuticFramework.value
    session_framework_is_set: bool      # False = pre-framework mode
    session_framework_locked: bool      # True = within cooldown, cannot change
    session_framework_change_count: int # how many times it has changed


class PlannerOutput(BaseModel):
    framework: TherapeuticFramework
    strategy: ResponseStrategy
    planner_confidence: PlannerConfidence
    rationale: str
    clarifying_question: Optional[str]
    clarifying_question_overridden: bool = False
    response_directive: str
    escalate_to_safety: bool = False
    escalation_reason: Optional[str] = None

    # RAG context — populated when strategy=psychoeducate
    kb_context: Optional[str] = None
    kb_sources: list[str] = []
    kb_retrieval_attempted: bool = False

    error: Optional[str] = None