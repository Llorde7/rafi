from contracts.classifier_contract import ClassifierInput, ClassifierOutput
from contracts.causal_contract import CausalInput, CausalOutput, HistoryTurn
from contracts.pipeline_envelope import PipelineEnvelope
from contracts.trajectory_contract import SessionTrajectory, TrajectoryFlag

from classifier import classify as _classify_raw
from causal_engine import analyse as _analyse_raw
from trajectory_engine import update_trajectory, format_trajectory_for_llm


# ─── Mappers ──────────────────────────────────────────────────────────────────

def _map_to_classifier_input(
    text: str,
    session_id: str | None,
    user_id: str | None
) -> ClassifierInput:
    return ClassifierInput(
        text=text,
        session_id=session_id,
        user_id=user_id
    )


def _map_classifier_to_causal(
    classifier_output: ClassifierOutput,
    session_history: list[HistoryTurn],
    trajectory: SessionTrajectory,
) -> CausalInput:
    return CausalInput(
        text=classifier_output.text,
        top_emotions=[e.model_dump() for e in classifier_output.top_3],
        reasoning=classifier_output.reasoning,
        session_history=session_history,
        trajectory_context=format_trajectory_for_llm(trajectory),
    )


def _map_turn_to_history(
    text: str,
    classifier_output: ClassifierOutput,
    causal_output: CausalOutput
) -> HistoryTurn:
    top = classifier_output.top_3[0] if classifier_output.top_3 else None
    return HistoryTurn(
        text=text,
        top_emotion=top.emotion if top else "neutral",
        confidence=top.confidence if top else 0.0,
        cause_type=causal_output.cause_type.value,
        temporal_pattern=causal_output.temporal_pattern
    )


# ─── Agent wrappers ───────────────────────────────────────────────────────────

def _run_classifier(input: ClassifierInput, history: list[dict]) -> ClassifierOutput:
    raw = _classify_raw(input.text, history)
    return ClassifierOutput(
        text=input.text,
        translation=raw.get("translation"),
        top_3=raw.get("top_3", []),
        reasoning=raw.get("reasoning", "")
    )


def _run_causal(input: CausalInput) -> CausalOutput:
    return _analyse_raw(input)


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline(
    text: str,
    session_id: str | None,
    user_id: str | None,
    classifier_history: list[dict],
    causal_history: list[HistoryTurn],
    trajectory: SessionTrajectory,
) -> PipelineEnvelope:
    """
    Runs all agents in sequence.
    Trajectory is passed in (loaded from Redis by main.py).
    Returns envelope. Caller updates and persists trajectory.
    """

    envelope = PipelineEnvelope(
        session_id=session_id,
        user_id=user_id,
        raw_text=text
    )

    # stage 1 — classifier
    classifier_input = _map_to_classifier_input(text, session_id, user_id)
    classifier_output = _run_classifier(classifier_input, classifier_history)
    envelope.classifier_output = classifier_output

    # stage 2 — causal analysis (receives trajectory context)
    causal_input = _map_classifier_to_causal(classifier_output, causal_history, trajectory)
    causal_output = _run_causal(causal_input)
    envelope.causal_output = causal_output

    return envelope


def build_history_turn(
    text: str,
    envelope: PipelineEnvelope
) -> HistoryTurn:
    """Called by main.py after pipeline completes."""
    return _map_turn_to_history(
        text,
        envelope.classifier_output,
        envelope.causal_output
    )


def advance_trajectory(
    trajectory: SessionTrajectory,
    envelope: PipelineEnvelope,
) -> SessionTrajectory:
    """
    Update trajectory from completed pipeline envelope.
    Called by main.py after pipeline completes, before saving to Redis.
    """
    top = (envelope.classifier_output.top_3[0]
           if envelope.classifier_output.top_3 else None)
    top_emotion = top.emotion if top else "neutral"
    confidence  = top.confidence if top else 0.0
    cause_type  = (envelope.causal_output.cause_type.value
                   if envelope.causal_output else None)

    return update_trajectory(trajectory, top_emotion, confidence, cause_type)


def get_escalation_flag(trajectory: SessionTrajectory) -> TrajectoryFlag | None:
    """
    Returns the current trajectory flag if it requires orchestration attention,
    otherwise None. Use this to decide whether to escalate to MIND-SAFE or
    change planner strategy.
    """
    actionable = {
        TrajectoryFlag.ESCALATING,
        TrajectoryFlag.SUSTAINED_NEGATIVE,
        TrajectoryFlag.AROUSAL_SPIKE,
        TrajectoryFlag.SUPPRESSION,
    }
    flag = TrajectoryFlag(trajectory.current_flag)
    return flag if flag in actionable else None