"""
trajectory_engine.py
────────────────────
Stateless functions that compute and update SessionTrajectory objects.
No I/O. No LLM calls. Pure pattern detection.

Called by the orchestrator after every turn.
"""

from contracts.trajectory_contract import (
    SessionTrajectory,
    EmotionShiftEvent,
    TrajectoryFlag,
    ValenceDirection,
    ArousalLevel,
    EMOTION_VALENCE,
    AROUSAL_MAP,
)

# ─── Constants ────────────────────────────────────────────────────────────────

SHIFT_MAGNITUDE_THRESHOLD = 0.40   # min valence jump to log a shift event
SUSTAINED_NEGATIVE_RUNS   = 3      # consecutive negative turns to flag
ESCALATION_WINDOW         = 3      # turns to look back for escalation trend
ESCALATION_SLOPE_THRESHOLD= 0.12   # mean per-turn valence drop to flag escalation


def get_valence(emotion: str) -> float:
    return EMOTION_VALENCE.get(emotion.lower(), 0.0)


def get_arousal(emotion: str) -> ArousalLevel:
    return AROUSAL_MAP.get(emotion.lower(), ArousalLevel.MEDIUM)


def arousal_to_int(a) -> int:
    key = a.value if hasattr(a, "value") else a
    return {"low": 0, "medium": 1, "high": 2}[key]


# ─── Core update ─────────────────────────────────────────────────────────────

def update_trajectory(
    trajectory: SessionTrajectory,
    top_emotion: str,
    confidence: float,
    cause_type: str | None = None,
) -> SessionTrajectory:
    """
    Ingest one new turn. Returns an updated SessionTrajectory.
    Does NOT mutate — works on a copy via model_copy(update=...).
    """
    valence      = get_valence(top_emotion)
    arousal      = get_arousal(top_emotion)
    turn_index   = trajectory.turn_count  # 0-based index of the incoming turn

    new_dominants  = trajectory.dominant_emotions + [top_emotion]
    new_valences   = trajectory.valence_series    + [valence]
    new_arousals   = trajectory.arousal_series    + [arousal.value]

    # ── Shift detection ───────────────────────────────────────────────────────
    new_shifts = list(trajectory.shift_events)
    if turn_index > 0:
        prev_emotion = trajectory.dominant_emotions[-1]
        prev_valence = trajectory.valence_series[-1]
        magnitude    = abs(valence - prev_valence)
        if magnitude >= SHIFT_MAGNITUDE_THRESHOLD:
            new_shifts.append(EmotionShiftEvent(
                turn_index   = turn_index,
                from_emotion = prev_emotion,
                to_emotion   = top_emotion,
                from_valence = prev_valence,
                to_valence   = valence,
                magnitude    = round(magnitude, 3),
            ))

    # ── Valence direction ─────────────────────────────────────────────────────
    direction = _compute_direction(new_valences)

    # ── Flags ─────────────────────────────────────────────────────────────────
    flag = _compute_flag(
        new_valences,
        new_arousals,
        new_shifts,
        direction,
        trajectory.current_arousal,
        arousal,
        confidence,
    )

    return trajectory.model_copy(update={
        "turn_count":       turn_index + 1,
        "dominant_emotions": new_dominants,
        "valence_series":   new_valences,
        "arousal_series":   new_arousals,
        "shift_events":     new_shifts,
        "valence_direction": direction.value,
        "current_arousal":  arousal.value,
        "current_flag":     flag.value,
    })


# ─── Direction ────────────────────────────────────────────────────────────────

def _compute_direction(valences: list[float]) -> ValenceDirection:
    if len(valences) < 2:
        return ValenceDirection.STABLE

    window = valences[-ESCALATION_WINDOW:]
    if len(window) < 2:
        return ValenceDirection.STABLE

    deltas = [window[i+1] - window[i] for i in range(len(window)-1)]
    mean_delta = sum(deltas) / len(deltas)

    if mean_delta <= -ESCALATION_SLOPE_THRESHOLD:
        return ValenceDirection.NEGATIVE
    if mean_delta >= ESCALATION_SLOPE_THRESHOLD:
        return ValenceDirection.POSITIVE

    # Check for oscillation
    sign_changes = sum(
        1 for i in range(len(deltas)-1)
        if (deltas[i] > 0) != (deltas[i+1] > 0)
    )
    if sign_changes >= len(deltas) - 1 and len(deltas) >= 2:
        return ValenceDirection.MIXED

    return ValenceDirection.STABLE


# ─── Flag computation ─────────────────────────────────────────────────────────

def _compute_flag(
    valences: list[float],
    arousals: list[str],
    shifts: list[EmotionShiftEvent],
    direction: ValenceDirection,
    prev_arousal: ArousalLevel,
    current_arousal: ArousalLevel,
    confidence: float,
) -> TrajectoryFlag:
    n = len(valences)

    # Arousal spike: jumped from low/medium to high in one turn
    if (arousal_to_int(prev_arousal) < arousal_to_int(ArousalLevel.HIGH)
            and current_arousal == ArousalLevel.HIGH
            and n > 1):
        return TrajectoryFlag.AROUSAL_SPIKE

    # Sustained negative: last N turns all below -0.4
    if n >= SUSTAINED_NEGATIVE_RUNS:
        last_n = valences[-SUSTAINED_NEGATIVE_RUNS:]
        if all(v < -0.4 for v in last_n):
            return TrajectoryFlag.SUSTAINED_NEGATIVE

    # Escalating: direction is negative over window
    if direction == ValenceDirection.NEGATIVE and n >= 2:
        return TrajectoryFlag.ESCALATING

    # De-escalating: direction is positive after a negative phase
    if direction == ValenceDirection.POSITIVE and n >= 2:
        if any(v < -0.3 for v in valences[:-1]):   # was negative before
            return TrajectoryFlag.DEESCALATING

    # Emotion shift just happened this turn
    if shifts and shifts[-1].turn_index == n - 1:
        return TrajectoryFlag.EMOTION_SHIFT

    # Suppression: low confidence + high arousal — user may be holding back
    if confidence < 0.45 and current_arousal == ArousalLevel.HIGH:
        return TrajectoryFlag.SUPPRESSION

    return TrajectoryFlag.NONE


# ─── Summary for LLM context ─────────────────────────────────────────────────

def format_trajectory_for_llm(trajectory: SessionTrajectory) -> str:
    """
    Produces a compact natural-language summary of the trajectory
    to append to causal engine prompts. Kept brief — the LLM context
    budget is finite.
    """
    if trajectory.turn_count == 0:
        return "Emotional trajectory: session just started, no prior turns."

    lines = [f"Emotional trajectory ({trajectory.turn_count} turns):"]

    # Valence summary
    if trajectory.valence_series:
        v_start = trajectory.valence_series[0]
        v_now   = trajectory.valence_series[-1]
        lines.append(
            f"  Valence: started at {v_start:+.2f}, currently {v_now:+.2f} "
            f"({trajectory.valence_direction})"
        )

    # Emotion arc (last 4 only)
    recent = trajectory.dominant_emotions[-4:]
    lines.append(f"  Recent emotion arc: {' → '.join(recent)}")

    # Arousal
    lines.append(f"  Current arousal: {trajectory.current_arousal}")

    # Flag
    if trajectory.current_flag != TrajectoryFlag.NONE.value:
        lines.append(f"  ⚠ Trajectory flag: {trajectory.current_flag}")

    # Shift events
    for shift in trajectory.shift_events[-2:]:  # last 2 only
        lines.append(
            f"  Shift at turn {shift.turn_index}: "
            f"{shift.from_emotion} → {shift.to_emotion} "
            f"(Δvalence {shift.magnitude:+.2f})"
        )

    # Cross-session baseline
    if trajectory.cross_session_baseline is not None:
        delta = (trajectory.valence_series[-1] - trajectory.cross_session_baseline
                 if trajectory.valence_series else 0.0)
        direction = "above" if delta >= 0 else "below"
        lines.append(
            f"  Cross-session baseline: {trajectory.cross_session_baseline:+.2f} "
            f"(current is {abs(delta):.2f} {direction} baseline)"
        )

    return "\n".join(lines)