import logging
from time import perf_counter

from contracts.classifier_contract import ClassifierInput, ClassifierOutput
from contracts.causal_contract import CausalInput, CausalOutput, HistoryTurn
from contracts.planner_contract import PlannerInput, PlannerOutput
from contracts.trace_contract import TraceInput, TraceOutput, TraceTurn, TonePreference, DetectedLanguage
from contracts.pipeline_envelope import PipelineEnvelope
from contracts.trajectory_contract import SessionTrajectory, TrajectoryFlag
from contracts.session_framework_contract import SessionFramework, update_session_framework

from classifier import classify as _classify_raw
from causal_engine import analyse as _analyse_raw
from planner_engine import plan_async as _plan_raw
from trace_engine import generate_async as _trace_raw
from trajectory_engine import update_trajectory, format_trajectory_for_llm, _weighted_valence


logger = logging.getLogger(__name__)


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
    session_framework: SessionFramework,
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

        # ── Session framework ───────────────────────────────────────────────
        session_framework=session_framework.framework.value,
        session_framework_is_set=session_framework.is_set,
        session_framework_locked=(
            trajectory.turn_count < session_framework.locked_until_turn
        ),
        session_framework_change_count=session_framework.change_count,
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


def _detect_language(text: str) -> DetectedLanguage:
    """
    Lightweight Swahili detection via common function words and markers.
    Classifier already handles both languages for emotion — this is for
    TRACE response language only. Swahili is opted-in: defaults to English
    when ambiguous. A proper langdetect library can replace this later.
    """
    SWAHILI_MARKERS = {
        "ni", "na", "ya", "wa", "kwa", "pia", "lakini", "sana", "kabisa",
        "sijui", "ninajua", "sijisikii", "naomba", "asante", "tafadhali",
        "ninahisi", "ninafikiria", "leo", "jana", "kesho", "rafiki",
        "mimi", "wewe", "yeye", "sisi", "nyinyi", "wao", "hii", "hiyo",
        "ndiyo", "hapana", "bado", "tayari", "kweli", "pole", "samahani",
    }
    tokens = set(text.lower().split())
    matches = tokens & SWAHILI_MARKERS
    # Require at least 2 marker matches to reduce false positives
    return DetectedLanguage.SWAHILI if len(matches) >= 2 else DetectedLanguage.ENGLISH


def _map_to_trace_input(
    planner_output: PlannerOutput,
    raw_text: str,
    tone_preference: TonePreference,
    detected_language: DetectedLanguage,
    trace_history: list[TraceTurn],
    cross_session_summary: str | None,
) -> TraceInput:
    return TraceInput(
        response_directive=planner_output.response_directive,
        framework=planner_output.framework.value,
        strategy=planner_output.strategy.value,
        clarifying_question=planner_output.clarifying_question,
        kb_context=planner_output.kb_context,
        escalate_to_safety=planner_output.escalate_to_safety,
        student_text=raw_text,
        tone_preference=tone_preference,
        detected_language=detected_language,
        trace_history=trace_history,
        cross_session_summary=cross_session_summary,
    )


# ─── Pipeline ─────────────────────────────────────────────────────────────────

async def run_pipeline(
    text: str,
    session_id: str | None,
    user_id: str | None,
    classifier_history: list[dict],
    causal_history: list[HistoryTurn],
    trajectory: SessionTrajectory,
    session_framework: SessionFramework,
    tone_preference: TonePreference = TonePreference.FRIENDLY,
    trace_history: list[TraceTurn] = [],
    cross_session_summary: str | None = None,
) -> tuple[PipelineEnvelope, SessionTrajectory, SessionFramework]:
    """
    Stage 1 — Classifier
    Stage 2 — Causal Analysis  (receives trajectory context)
    Stage 3 — Strategic Planner (receives classifier + causal + trajectory + session framework)
    Stage 4 — TRACE (skipped if escalate_to_safety=True)

    Returns (envelope, updated_trajectory, updated_session_framework).
    """
    # Detect language on raw text before any processing
    detected_language = _detect_language(text)

    envelope = PipelineEnvelope(
        session_id=session_id,
        user_id=user_id,
        raw_text=text,
        tone_preference=tone_preference,
        detected_language=detected_language,
    )
    pipeline_started = perf_counter()

    # Stage 1: Classifier
    classifier_input  = _map_to_classifier_input(text, session_id, user_id)
    classifier_started = perf_counter()
    classifier_output = _run_classifier(classifier_input, classifier_history)
    logger.info(
        "Pipeline timing | session_id=%s stage=classifier duration_ms=%.1f",
        session_id,
        (perf_counter() - classifier_started) * 1000,
    )
    envelope.classifier_output = classifier_output

    # Stage 2: Causal Analysis
    causal_input  = _map_classifier_to_causal(classifier_output, causal_history, trajectory)
    causal_started = perf_counter()
    causal_output = _run_causal(causal_input)
    logger.info(
        "Pipeline timing | session_id=%s stage=causal duration_ms=%.1f",
        session_id,
        (perf_counter() - causal_started) * 1000,
    )
    envelope.causal_output = causal_output

    # Stage 3: Strategic Planner (async — may trigger RAG)
    planner_input  = _map_to_planner_input(
        classifier_output, causal_output, trajectory, session_framework
    )
    planner_started = perf_counter()
    planner_output = await _run_planner(planner_input)
    logger.info(
        "Pipeline timing | session_id=%s stage=planner duration_ms=%.1f",
        session_id,
        (perf_counter() - planner_started) * 1000,
    )
    envelope.planner_output = planner_output

    # Stage 4: TRACE — skipped entirely if escalate_to_safety is True
    if not planner_output.escalate_to_safety:
        trace_input = _map_to_trace_input(
            planner_output=planner_output,
            raw_text=text,
            tone_preference=tone_preference,
            detected_language=detected_language,
            trace_history=trace_history,
            cross_session_summary=cross_session_summary,
        )
        trace_started = perf_counter()
        trace_output = await _trace_raw(trace_input)
        logger.info(
            "Pipeline timing | session_id=%s stage=trace duration_ms=%.1f",
            session_id,
            (perf_counter() - trace_started) * 1000,
        )
        envelope.trace_output = trace_output
    else:
        logger.info(
            "Pipeline timing | session_id=%s stage=trace status=skipped reason=escalate_to_safety",
            session_id,
        )

    logger.info(
        "Pipeline timing | session_id=%s stage=total duration_ms=%.1f",
        session_id,
        (perf_counter() - pipeline_started) * 1000,
    )

    # Update session framework + trajectory based on this completed envelope
    updated_trajectory = advance_trajectory(trajectory, envelope)
    envelope.escalation_flag = get_escalation_flag(updated_trajectory)
    updated_sf = update_session_framework(
        sf=session_framework,
        turn_count=updated_trajectory.turn_count,
        trajectory_flag=updated_trajectory.current_flag.value,
        cause_type=causal_output.cause_type.value,
        causal_confidence_category=causal_output.confidence_category.value,
    )
    if (
        updated_sf.framework != session_framework.framework
        or updated_sf.is_set != session_framework.is_set
    ):
        logger.info(
            "Session framework | session_id=%s %s → %s (change_count=%d)",
            session_id,
            session_framework.framework.value,
            updated_sf.framework.value,
            updated_sf.change_count,
        )

    return envelope, updated_trajectory, updated_sf


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