from contracts.classifier_contract import ClassifierInput, ClassifierOutput
from contracts.causal_contract import CausalInput, CausalOutput, HistoryTurn
from contracts.pipeline_envelope import PipelineEnvelope

from classifier import classify as _classify_raw
from causal_engine import analyse as _analyse_raw


# ─── Mappers ──────────────────────────────────────────────────────────────────
# Thin translation layers between agent output and next agent input.
# No logic. No LLM calls. Shape translation only.

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
    session_history: list[HistoryTurn]
) -> CausalInput:
    return CausalInput(
        text=classifier_output.text,
        top_emotions=[e.model_dump() for e in classifier_output.top_3],
        reasoning=classifier_output.reasoning,
        session_history=session_history
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
) -> PipelineEnvelope:
    """
    Runs all agents in sequence.
    Each agent receives only its own typed input.
    Returns a PipelineEnvelope with all outputs stacked.
    Caller (main.py) is responsible for persisting the envelope and updating history.
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

    # stage 2 — causal analysis
    causal_input = _map_classifier_to_causal(classifier_output, causal_history)
    causal_output = _run_causal(causal_input)
    envelope.causal_output = causal_output

    # stage N — planner slot
    # planner_input = _map_causal_to_planner(causal_output)
    # planner_output = _run_planner(planner_input)
    # envelope.planner_output = planner_output

    return envelope


def build_history_turn(
    text: str,
    envelope: PipelineEnvelope
) -> HistoryTurn:
    """
    Called by main.py after pipeline completes.
    Extracts the HistoryTurn from the envelope for causal context on next turn.
    """
    return _map_turn_to_history(
        text,
        envelope.classifier_output,
        envelope.causal_output
    )
