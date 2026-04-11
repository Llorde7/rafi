"""
session_framework_contract.py
──────────────────────────────
Session-level therapeutic framework.

Sits above the turn-by-turn planner strategy. Set once early in the
session (when causal confidence first reaches 'confident', or by turn 3),
then persists with a change cooldown and a hard cap of 3 changes.

The planner receives this each turn and is constrained to select only
strategies compatible with the current session framework. Framework
changes are decided in code (orchestrator), not by the LLM.
"""

from pydantic import BaseModel
from typing import Optional
from contracts.planner_contract import TherapeuticFramework


# ── Compatible strategies per framework ───────────────────────────────────────
# Hard constraints — planner output is overridden if it violates these.

COMPATIBLE_STRATEGIES: dict[str, list[str]] = {
    "cbt":              ["reframe", "psychoeducate", "probe", "validate"],
    "mi":               ["motivate", "probe", "reflect", "validate"],
    "solution_focused": ["solution_elicit", "validate", "probe", "psychoeducate"],
    "person_centred":   ["validate", "reflect", "active_listen", "psychoeducate"],
    "none":             ["active_listen", "validate"],
}

# Fallback strategy if planner selects incompatible one
FRAMEWORK_FALLBACK_STRATEGY: dict[str, str] = {
    "cbt":              "reframe",
    "mi":               "motivate",
    "solution_focused": "solution_elicit",
    "person_centred":   "validate",
    "none":             "active_listen",
}

SESSION_FRAMEWORK_CHANGE_CAP = 3   # max changes per session
SESSION_FRAMEWORK_LOCK_TURNS = 3   # turns locked after set or change
PRE_FRAMEWORK_TURN_CAP       = 3   # set framework by this turn at the latest


class SessionFramework(BaseModel):
    """
    Persisted in Redis for the lifetime of the session.
    """
    framework:               TherapeuticFramework = TherapeuticFramework.PERSON_CENTRED
    is_set:                  bool  = False   # False = pre-framework mode
    set_at_turn:             int   = 0
    cause_type_at_set:       str   = ""
    change_count:            int   = 0       # hard cap: SESSION_FRAMEWORK_CHANGE_CAP
    locked_until_turn:       int   = 0       # no changes before this turn number
    last_cause_type:         str   = ""      # cause_type from previous turn
    consecutive_cause_shifts: int  = 0       # turns with a different cause_type
    consecutive_emotion_shifts: int = 0      # turns with emotion_shift flag


def should_set_framework(
    sf: SessionFramework,
    turn_count: int,
    causal_confidence_category: str,
) -> bool:
    """
    Returns True if the session framework should be set this turn.
    Conditions: not yet set AND (causal is confident OR turn >= cap).
    """
    if sf.is_set:
        return False
    if causal_confidence_category == "confident":
        return True
    if turn_count >= PRE_FRAMEWORK_TURN_CAP:
        return True
    return False


def should_change_framework(
    sf: SessionFramework,
    turn_count: int,
    trajectory_flag: str,
    cause_type: str,
) -> bool:
    """
    Returns True if the session framework should change this turn.
    All three conditions must be evaluated — any one can trigger a change
    provided the cooldown and cap allow it.
    """
    if not sf.is_set:
        return False
    if sf.change_count >= SESSION_FRAMEWORK_CHANGE_CAP:
        return False
    if turn_count < sf.locked_until_turn:
        return False

    # Condition 1: sustained emotion shift (2+ consecutive turns)
    if sf.consecutive_emotion_shifts >= 2:
        return True

    # Condition 2: cause type changed for 2 consecutive turns
    if sf.consecutive_cause_shifts >= 2:
        return True

    # Condition 3: explicit de-escalation after negative phase
    if trajectory_flag == "de-escalating":
        return True

    return False


def select_framework_for_cause(cause_type: str) -> TherapeuticFramework:
    """
    Maps a cause_type to its most appropriate session framework.
    Used when setting or changing the session framework.
    """
    mapping = {
        "cognitive_distortion": TherapeuticFramework.CBT,
        "avoidance_behaviour":  TherapeuticFramework.MI,
        "identity_threat":      TherapeuticFramework.MI,
        "unresolved_loss":      TherapeuticFramework.PERSON_CENTRED,
        "somatic_response":     TherapeuticFramework.PERSON_CENTRED,
        "interpersonal_conflict": TherapeuticFramework.PERSON_CENTRED,
        "ambiguous":            TherapeuticFramework.PERSON_CENTRED,
    }
    return mapping.get(cause_type, TherapeuticFramework.PERSON_CENTRED)


def update_session_framework(
    sf: SessionFramework,
    turn_count: int,
    trajectory_flag: str,
    cause_type: str,
    causal_confidence_category: str,
    planner_suggested_framework: Optional[str] = None,
) -> SessionFramework:
    """
    Main update function. Called by orchestrator after each turn.
    Returns a new SessionFramework — does not mutate in place.

    Priority:
    1. If not set yet: set it.
    2. If change conditions met: change it.
    3. Otherwise: update shift counters only.
    """
    # ── Update shift counters first ───────────────────────────────────────────
    emotion_shifted = trajectory_flag == "emotion_shift"
    cause_shifted   = (sf.last_cause_type != "" and cause_type != sf.last_cause_type)

    new_consecutive_emotion = (sf.consecutive_emotion_shifts + 1) if emotion_shifted else 0
    new_consecutive_cause   = (sf.consecutive_cause_shifts + 1) if cause_shifted else 0

    # ── Set framework (first time) ────────────────────────────────────────────
    if should_set_framework(sf, turn_count, causal_confidence_category):
        new_framework = select_framework_for_cause(cause_type)
        # De-escalating trajectory at set time → prefer solution_focused
        if trajectory_flag == "de-escalating":
            new_framework = TherapeuticFramework.SOLUTION_FOCUSED
        return sf.model_copy(update={
            "framework":                 new_framework,
            "is_set":                    True,
            "set_at_turn":               turn_count,
            "cause_type_at_set":         cause_type,
            "locked_until_turn":         turn_count + SESSION_FRAMEWORK_LOCK_TURNS,
            "last_cause_type":           cause_type,
            "consecutive_cause_shifts":  0,
            "consecutive_emotion_shifts": 0,
        })

    # ── Change framework ──────────────────────────────────────────────────────
    if should_change_framework(sf, turn_count, trajectory_flag, cause_type):
        # De-escalation → solution_focused
        if trajectory_flag == "de-escalating":
            new_framework = TherapeuticFramework.SOLUTION_FOCUSED
        else:
            new_framework = select_framework_for_cause(cause_type)

        # Don't change to the same framework
        if new_framework == sf.framework:
            # Still reset shift counters
            return sf.model_copy(update={
                "last_cause_type":            cause_type,
                "consecutive_cause_shifts":   new_consecutive_cause,
                "consecutive_emotion_shifts": new_consecutive_emotion,
            })

        return sf.model_copy(update={
            "framework":                  new_framework,
            "change_count":               sf.change_count + 1,
            "locked_until_turn":          turn_count + SESSION_FRAMEWORK_LOCK_TURNS,
            "last_cause_type":            cause_type,
            "consecutive_cause_shifts":   0,
            "consecutive_emotion_shifts": 0,
        })

    # ── No change — update counters only ──────────────────────────────────────
    return sf.model_copy(update={
        "last_cause_type":            cause_type,
        "consecutive_cause_shifts":   new_consecutive_cause,
        "consecutive_emotion_shifts": new_consecutive_emotion,
    })


def enforce_strategy_compatibility(
    session_framework: str,
    selected_strategy: str,
) -> str:
    """
    Hard constraint. If the planner selected a strategy incompatible with
    the session framework, returns the framework's fallback strategy instead.
    Called in planner_engine after the LLM response is parsed.
    """
    compatible = COMPATIBLE_STRATEGIES.get(session_framework, [])
    if selected_strategy in compatible:
        return selected_strategy
    return FRAMEWORK_FALLBACK_STRATEGY.get(session_framework, "validate")