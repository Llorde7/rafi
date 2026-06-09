import asyncio
from typing import Optional

from contracts.situation_contract import SituationType
from contracts.intent_contract import IntentState, detect_intent
from contracts.causal_contract import CausalInput, HistoryTurn
from contracts.planner_contract import PlannerInput
from contracts.trace_contract import TraceInput, TonePreference, DetectedLanguage
from contracts.session_framework_contract import SessionFramework
from contracts.pipeline_envelope import PipelineEnvelope
from classifier import classify, Stage1Output
from causal_engine import analyse
from planner_engine import plan
from trace_engine import generate

# ---------------------------------------------------------------------------
# Language detection — heuristic, keyword-based
# Upgrade to langdetect or lingua in a future session
# ---------------------------------------------------------------------------

SWAHILI_KEYWORDS = {
    "mimi", "wewe", "yeye", "sisi", "ninajua", "sijui", "niko", "uko",
    "hapa", "pale", "sana", "kidogo", "lakini", "kwa", "na", "ya",
    "ni", "au", "bado", "tayari", "tena", "karibu", "mbali", "sawa",
    "pole", "asante", "habari", "nzuri", "vibaya", "shida", "msaada",
    "nisaidie", "nifanye", "nataka", "sifahamu", "naomba",
}

def _detect_language(text: str) -> DetectedLanguage:
    words = set(text.lower().split())
    matches = words & SWAHILI_KEYWORDS
    if len(matches) >= 2:
        # Check for code-switching (mix of Swahili and English)
        english_words = {"i", "the", "is", "are", "was", "have", "my", "me", "you", "it"}
        if words & english_words:
            return DetectedLanguage.mixed
        return DetectedLanguage.sw
    return DetectedLanguage.en


# ---------------------------------------------------------------------------
# Mapper: Stage1Output + CausalOutput + intent → TraceInput
# ---------------------------------------------------------------------------

def _map_to_trace_input(
    text: str,
    detected_language: DetectedLanguage,
    tone_preference: TonePreference,
    planner_output,
    causal_output,
    intent_state: IntentState,
    trace_history: list,
) -> TraceInput:
    return TraceInput(
        text=text,
        detected_language=detected_language,
        tone_preference=tone_preference,
        technique_cluster=planner_output.technique_cluster,
        executor_instruction=planner_output.technique_cluster.executor_instruction,
        global_cause=causal_output.global_cause,
        trigger_spans=[span.span for span in causal_output.trigger_spans],
        clarifying_question_from_cae=causal_output.clarifying_question,
        intent_state=intent_state.value,
        trace_history=trace_history,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(
    text: str,
    session_id: str,
    session_context: dict,
    # session_context keys:
    #   classifier_history: list[dict]   — last 6 Stage1Output dicts
    #   causal_history: list[dict]       — last 6 HistoryTurn dicts
    #   trajectory: dict                 — SessionTrajectory dict
    #   trace_history: list              — last 6 TraceTurn objects
    #   session_framework: dict          — SessionFramework dict
    #   tone: str                        — "friendly" | "clinical"
    #   prior_intent: str                — last turn's intent state
    #   session_history_summary: str     — compressed prior turns
):
    from trajectory_engine import update_trajectory
    from contracts.session_framework_contract import update_session_framework
    from contracts.classifier_contract import ClassifierInput

    # ── Pre-pipeline ────────────────────────────────────────────────────────
    detected_language = _detect_language(text)
    tone_preference   = TonePreference(session_context.get("tone", "friendly"))

    # ── Stage 1 — Fused: emotion classification + situation type (1 LLM call) ─
    stage1_output: Stage1Output = await classify(
        ClassifierInput(text=text, session_id=session_id),
        session_context.get("classifier_history", []),
    )

    # ── Stage 2 — Trajectory update (deterministic) ─────────────────────────
    from contracts.trajectory_contract import SessionTrajectory

    # Rehydrate trajectory if it came in as a dict
    traj_data = session_context.get("trajectory")
    if isinstance(traj_data, dict):
        trajectory = SessionTrajectory(**traj_data)
    else:
        trajectory = traj_data if traj_data is not None else SessionTrajectory(session_id=session_id)

    trajectory = update_trajectory(
        trajectory,
        stage1_output.top_3,
    )

    # ── Stage 3 — Fused: causal analysis + cognitive pattern + behavioral risk ─
    causal_history = [
        HistoryTurn(**h) for h in session_context.get("causal_history", [])
    ]
    causal_output = await analyse(
        CausalInput(
            text=text,
            top_emotions=[e.model_dump() for e in stage1_output.top_3],
            reasoning=stage1_output.reasoning,
            situation_type=stage1_output.situation_type.value,
            session_history=causal_history,
        )
    )

    # ── Stage 4 — Intent detection (deterministic) ───────────────────────────
    prior_intent_str  = session_context.get("prior_intent", IntentState.processing.value)
    prior_intent      = IntentState(prior_intent_str)
    intent_state      = detect_intent(
        text=text,
        trajectory_flag=trajectory.current_flag,
        prior_intent=prior_intent,
        turn_count=trajectory.turn_count,
    )

    # ── Stage 5 — Session framework update (deterministic) ───────────────────
    # Rehydrate session framework if it came in as a dict
    sf_data = session_context.get("session_framework")
    if isinstance(sf_data, dict):
        prior_sf = SessionFramework(**sf_data)
    elif isinstance(sf_data, SessionFramework):
        prior_sf = sf_data
    else:
        prior_sf = SessionFramework()

    session_framework: SessionFramework = update_session_framework(
        sf=prior_sf,
        turn_count=trajectory.turn_count,
        trajectory_flag=trajectory.current_flag.value,
        cause_type=causal_output.cause_type.value,
        causal_confidence_category=causal_output.confidence_category.value,
    )

    # ── Stage 6 — Micro-intervention planner (1 LLM call) ────────────────────
    planner_output = await plan(
        PlannerInput(
            text=text,
            situation_type=stage1_output.situation_type.value,
            situation_summary=stage1_output.situation_summary,
            has_concrete_deadline=stage1_output.has_concrete_deadline,
            has_external_referents=stage1_output.has_external_referents,
            cognitive_pattern=causal_output.cognitive_pattern.value,
            behavioral_risk=causal_output.behavioral_risk.value,
            intent_state=intent_state.value,
            top_emotion=stage1_output.top_3[0].emotion,
            emotion_confidence=stage1_output.top_3[0].confidence,
            cause_type=causal_output.cause_type.value,
            confidence_category=causal_output.confidence_category.value,
            causal_chain=causal_output.causal_chain,
            temporal_pattern=causal_output.temporal_pattern,
            planner_instruction=causal_output.planner_instruction.value,
            clarifying_question=causal_output.clarifying_question,
            session_framework=session_framework.framework,
            session_history_summary=session_context.get("session_history_summary", ""),
        ),
        arousal=trajectory.current_arousal,
        valence=trajectory.valence_series[-1] if trajectory.valence_series else -0.3,
    )

    # ── Stage 7 — RAG (conditional) ──────────────────────────────────────────
    if _should_retrieve(planner_output, stage1_output):
        try:
            from rag.rag_pipeline import retrieve
            rag_result = await retrieve(text, planner_output)
            planner_output = _inject_rag_context(planner_output, rag_result)
        except Exception as e:
            print(f"[RAG] Retrieval failed: {e}")

    # ── Stage 8 — TRACE (1 LLM call, skipped if escalate_to_safety) ──────────
    trace_output = None
    if not planner_output.escalate_to_safety:
        trace_input = _map_to_trace_input(
            text=text,
            detected_language=detected_language,
            tone_preference=tone_preference,
            planner_output=planner_output,
            causal_output=causal_output,
            intent_state=intent_state,
            trace_history=session_context.get("trace_history", []),
        )
        trace_output = await generate(trace_input)

    # ── Assemble envelope ────────────────────────────────────────────────────
    envelope = PipelineEnvelope(
        text=text,
        detected_language=detected_language.value,
        tone_preference=tone_preference.value,
        classifier_output=stage1_output,
        situation_type=stage1_output.situation_type.value,
        situation_summary=stage1_output.situation_summary,
        causal_output=causal_output,
        cognitive_pattern=causal_output.cognitive_pattern.value,
        behavioral_risk=causal_output.behavioral_risk.value,
        intent_state=intent_state.value,
        planner_output=planner_output,
        trace_output=trace_output,
    )

    # Returns 4-tuple — unpack in main.py
    return envelope, trajectory, session_framework, intent_state


# ---------------------------------------------------------------------------
# RAG helpers
# ---------------------------------------------------------------------------

def _should_retrieve(planner_output, stage1_output: Stage1Output) -> bool:
    cluster = planner_output.technique_cluster
    technique_names = {t.name for t in cluster.techniques}

    # Retrieve if psychoeducation is in the cluster
    if any(t.modality == "psychoeducation" for t in cluster.techniques):
        return True

    # Retrieve if action_seeking + has_external_referents (factual query likely)
    if stage1_output.has_external_referents and stage1_output.has_concrete_deadline:
        return True

    return False


def _inject_rag_context(planner_output, rag_result) -> object:
    if rag_result and rag_result.chunks:
        summary = " ".join(c.content[:200] for c in rag_result.chunks[:2])
        planner_output.technique_cluster.rag_context = summary
        planner_output.technique_cluster.kb_sources  = [
            c.source for c in rag_result.chunks[:2]
        ]
    return planner_output