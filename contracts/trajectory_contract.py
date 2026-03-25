from pydantic import BaseModel
from typing import Optional
from enum import Enum


class ValenceDirection(str, Enum):
    POSITIVE   = "positive"    # moving toward positive emotions
    NEGATIVE   = "negative"    # moving toward negative emotions
    STABLE     = "stable"      # no consistent direction
    MIXED      = "mixed"       # oscillating / no clear pattern


class ArousalLevel(str, Enum):
    HIGH    = "high"     # animated, agitated, panicked
    MEDIUM  = "medium"
    LOW     = "low"      # flat, withdrawn, dissociated


class TrajectoryFlag(str, Enum):
    """
    Structural flags raised by the trajectory analyser.
    These are pattern observations, not clinical diagnoses.
    The orchestrator decides how to act on them.
    """
    NONE              = "none"
    ESCALATING        = "escalating"        # sustained negative valence increase
    DEESCALATING      = "de-escalating"     # recovering across turns
    EMOTION_SHIFT     = "emotion_shift"     # dominant emotion changed sharply
    SUSTAINED_NEGATIVE= "sustained_negative"# ≥3 consecutive high-negative turns
    AROUSAL_SPIKE     = "arousal_spike"     # arousal jumped ≥2 levels in 1 turn
    SUPPRESSION       = "suppression"       # repeated low confidence + high arousal


class EmotionShiftEvent(BaseModel):
    """Records a detected shift between dominant emotions."""
    turn_index: int
    from_emotion: str
    to_emotion: str
    from_valence: float     # approximate valence [-1, 1]
    to_valence: float
    magnitude: float        # abs(to_valence - from_valence)


# ─── Valence map ──────────────────────────────────────────────────────────────
# Approximate valence for the 28 GoEmotions labels + safety-net neutral.
# Range [-1.0, 1.0]. Not a clinical instrument — used for trajectory direction.

EMOTION_VALENCE: dict[str, float] = {
    # strongly negative
    "grief":         -0.95,
    "remorse":       -0.85,
    "fear":          -0.80,
    "sadness":       -0.78,
    "despair":       -0.90,
    "anger":         -0.70,
    "disgust":       -0.68,
    "embarrassment": -0.65,
    "disappointment":-0.60,
    "annoyance":     -0.55,
    "nervousness":   -0.50,
    "confusion":     -0.30,
    # mildly negative / ambiguous
    "disapproval":   -0.45,
    "realization":   -0.10,
    # neutral
    "neutral":        0.00,
    "surprise":       0.05,
    # mildly positive
    "curiosity":      0.25,
    "desire":         0.30,
    "relief":         0.45,
    "optimism":       0.55,
    # strongly positive
    "caring":         0.65,
    "amusement":      0.70,
    "approval":       0.65,
    "gratitude":      0.75,
    "admiration":     0.72,
    "love":           0.85,
    "excitement":     0.80,
    "joy":            0.90,
    "pride":          0.75,
}

AROUSAL_MAP: dict[str, ArousalLevel] = {
    "fear":          ArousalLevel.HIGH,
    "anger":         ArousalLevel.HIGH,
    "excitement":    ArousalLevel.HIGH,
    "nervousness":   ArousalLevel.HIGH,
    "disgust":       ArousalLevel.HIGH,
    "surprise":      ArousalLevel.MEDIUM,
    "joy":           ArousalLevel.MEDIUM,
    "amusement":     ArousalLevel.MEDIUM,
    "annoyance":     ArousalLevel.MEDIUM,
    "desire":        ArousalLevel.MEDIUM,
    "confusion":     ArousalLevel.MEDIUM,
    "sadness":       ArousalLevel.LOW,
    "grief":         ArousalLevel.LOW,
    "remorse":       ArousalLevel.LOW,
    "disappointment":ArousalLevel.LOW,
    "neutral":       ArousalLevel.LOW,
    "relief":        ArousalLevel.LOW,
}


# ─── Session-level trajectory ─────────────────────────────────────────────────

class SessionTrajectory(BaseModel):
    """
    Maintained in Redis for the duration of a session.
    Updated after every turn. Fed to causal engine as context.
    """
    session_id: str
    turn_count: int                     = 0
    dominant_emotions: list[str]        = []   # one per turn, in order
    valence_series: list[float]         = []   # one per turn, in order
    arousal_series: list[str]           = []   # ArousalLevel value per turn
    shift_events: list[EmotionShiftEvent] = []
    current_flag: TrajectoryFlag        = TrajectoryFlag.NONE
    valence_direction: ValenceDirection = ValenceDirection.STABLE
    current_arousal: ArousalLevel       = ArousalLevel.MEDIUM

    # cross-session baseline loaded at session start (may be None for new users)
    cross_session_baseline: Optional[float] = None
    sessions_seen: int                  = 0


# ─── Cross-session profile ────────────────────────────────────────────────────

class UserEmotionalProfile(BaseModel):
    """
    Persisted to Postgres at session close.
    Loaded at session start to seed cross-session baseline.
    """
    user_id: str
    sessions_seen: int              = 0
    mean_valence: float             = 0.0     # rolling mean across all sessions
    dominant_cause_types: list[str] = []      # most common cause types, ranked
    last_session_flag: Optional[str]= None    # TrajectoryFlag from last session
    last_session_end_emotion: Optional[str] = None