from contracts.classifier_contract import ClassifierInput, ClassifierOutput
from contracts.causal_contract import CausalInput, CausalOutput, HistoryTurn
from contracts.planner_contract import PlannerInput, PlannerOutput
from contracts.pipeline_envelope import PipelineEnvelope
from contracts.trajectory_contract import SessionTrajectory, TrajectoryFlag

from classifier import classify as _classify_raw
from causal_engine import analyse as _analyse_raw
from planner_engine import plan_async as _plan_raw
from trajectory_engine import update_trajectory, format_trajectory_for_llm, _weighted_valence


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
        top_emotions=[e.model_dump(mode="json") for e in classifier_output.top_3],
        reasoning=classifier_output.reasoning,
        session_history=session_history,
        trajectory_context=format_trajectory_for_llm(trajectory),
    )


def _map_to_planner_input(
    classifier_output: ClassifierOutput,
    causal_output: CausalOutput,
    trajectory: SessionTrajectory,
) -> PlannerInput:
    top = classifier_output.top_3[0] if classifier_output.top_3 else None

    # current_valence: weighted across all 3 scores — same as trajectory engine
    current_valence = _weighted_valence(classifier_output.top_3)

    return PlannerInput(
        # ── Classifier ────────────────────────────────────────────────────────
        text=classifier_output.text,
        top_emotion=top.emotion.value if top else "neutral",
        emotion_confidence=top.confidence if top else 0.0,
        top_3_emotions=[e.model_dump(mode="json") for e in classifier_output.top_3],

        # ── Causal ───────────────────────────────────────────────────────────
        global_cause=causal_output.global_cause,
        causal_chain=causal_output.causal_chain,
        cause_type=causal_output.cause_type.value,
        causal_confidence_score=causal_output.confidence_score,
        causal_confidence_category=causal_output.confidence_category.value,
        causal_planner_instruction=causal_output.planner_instruction.value,
        clarifying_question=causal_output.clarifying_question,

        # ── Trajectory ───────────────────────────────────────────────────────
        trajectory_flag=trajectory.current_flag.value,
        valence_direction=trajectory.valence_direction.value,
        current_arousal=trajectory.current_arousal.value,
        current_valence=current_valence,
        shift_events=[s.model_dump() for s in trajectory.shift_events[-3:]],
        turn_count=trajectory.turn_count,
        cross_session_baseline=trajectory.cross_session_baseline,
    )


def _map_turn_to_history(
    text: str,
    classifier_output: ClassifierOutput,
    causal_output: CausalOutput
) -> HistoryTurn:
    top = classifier_output.top_3[0] if classifier_output.top_3 else None
    return HistoryTurn(
        text=text,
        top_emotion=top.emotion.value if top else "neutral",
        confidence=top.confidence if top else 0.0,
        cause_type=causal_output.cause_type.value,
        temporal_pattern=causal_output.temporal_pattern
    )


# ─── Agent wrappers ───────────────────────────────────────────────────────────

def _run_classifier(input: ClassifierInput, history: list[dict]) -> ClassifierOutput:
    return _classify_raw(input.text, history)


def _run_causal(input: CausalInput) -> CausalOutput:
    return _analyse_raw(input)


async def _run_planner(input: PlannerInput) -> PlannerOutput:
    return await _plan_raw(input)


# ─── Pipeline ─────────────────────────────────────────────────────────────────

async def run_pipeline(
    text: str,
    session_id: str | None,
    user_id: str | None,
    classifier_history: list[dict],
    causal_history: list[HistoryTurn],
    trajectory: SessionTrajectory,
) -> PipelineEnvelope:
    """
    Stage 1 — Classifier
    Stage 2 — Causal Analysis  (receives trajectory context)
    Stage 3 — Strategic Planner (receives classifier + causal + trajectory)
    """
    envelope = PipelineEnvelope(
        session_id=session_id,
        user_id=user_id,
        raw_text=text
    )

    # Stage 1: Classifier
    classifier_input  = _map_to_classifier_input(text, session_id, user_id)
    classifier_output = _run_classifier(classifier_input, classifier_history)
    envelope.classifier_output = classifier_output

    # Stage 2: Causal Analysis
    causal_input  = _map_classifier_to_causal(classifier_output, causal_history, trajectory)
    causal_output = _run_causal(causal_input)
    envelope.causal_output = causal_output

    # Stage 3: Strategic Planner (async — may trigger RAG)
    planner_input  = _map_to_planner_input(classifier_output, causal_output, trajectory)
    planner_output = await _run_planner(planner_input)
    envelope.planner_output = planner_output

    return envelope


def build_history_turn(
    text: str,
    envelope: PipelineEnvelope
) -> HistoryTurn:
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
    Uses the full top_3 emotion scores — matching the actual trajectory_engine
    signature which takes list[EmotionScore] for weighted valence/arousal.
    """
    emotion_scores = (
        envelope.classifier_output.top_3
        if envelope.classifier_output else []
    )
    return update_trajectory(trajectory, emotion_scores)


def get_escalation_flag(trajectory: SessionTrajectory) -> TrajectoryFlag | None:
    actionable = {
        TrajectoryFlag.ESCALATING,
        TrajectoryFlag.SUSTAINED_NEGATIVE,
        TrajectoryFlag.AROUSAL_SPIKE,
        TrajectoryFlag.SUPPRESSION,
    }
    flag = TrajectoryFlag(trajectory.current_flag)
    return flag if flag in actionable else None