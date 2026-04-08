"""
planner_engine.py
─────────────────
Strategic Planner Agent.

Receives PlannerInput (classifier + causal + trajectory) and selects:
  1. A therapeutic framework  (CBT / MI / solution-focused / person-centred / none)
  2. A response strategy      (validate / reflect / reframe / probe / ...)
  3. A structured response_directive for TRACE

The causal engine's planner_instruction is advisory — the planner reads
it but is not bound by it.

The clarifying question from the causal engine is inherited by default.
The planner may override it if it has a more targeted one.
"""

import os
import json
import logging
from time import perf_counter
from groq import Groq
from dotenv import load_dotenv

from contracts.planner_contract import (
    PlannerInput, PlannerOutput,
    TherapeuticFramework, ResponseStrategy, PlannerConfidence,
)

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.3-70b-versatile"
logger = logging.getLogger(__name__)


# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Strategic Planner Agent in a mental health support pipeline for university students in Kenya.

You receive:
- The student's message and classified emotions (from a GoEmotions classifier)
- A causal analysis (what is driving the emotion and how confident the causal agent is)
- A session trajectory (how the student's emotional state has moved, and cross-session context)

You select a therapeutic framework and a response strategy. Your output feeds TRACE, a response generator — not a human directly.

YOUR FRAMEWORKS:
- cbt: Use when there is evidence of distorted thinking, catastrophising, all-or-nothing beliefs, or self-critical cognitions. Best for cause_type=cognitive_distortion.
- mi: Use when the student is ambivalent, avoiding change, or expressing desire without action. Best for avoidance_behaviour or identity_threat.
- solution_focused: Use when the student has prior coping capacity or trajectory is de-escalating. Surfaces strengths, not deficits.
- person_centred: Use when the student needs unconditional positive regard — grief, loss, shame, or when causal confidence is low. No technique. Presence first.
- none: Use when trajectory_flag is escalating/sustained_negative/arousal_spike AND arousal is high. Active listening only. Do not attempt technique on a dysregulated student.

YOUR STRATEGIES:
- validate: Acknowledge and normalise the emotion explicitly. Best for grief, fear, shame.
- reflect: Mirror content back without interpretation. Use early in session or when cause is ambiguous.
- reframe: Offer an alternative cognitive lens. Only after validation. Only when causal confidence >= 0.65.
- probe: Ask a targeted follow-up. Use when causal confidence is partial/insufficient.
- psychoeducate: Brief normalising information + relevant university support resources. Use when the student needs factual context about what support exists.
- active_listen: Stay present. No directive. Use when hold is received AND you agree, or arousal is very high.
- motivate: Elicit change talk. MI only.
- solution_elicit: Surface existing strengths. Solution-focused only.

PREFERRED PAIRINGS:
- cbt + reframe or psychoeducate
- mi + motivate or probe
- solution_focused + solution_elicit or validate
- person_centred + validate or reflect
- none + active_listen

CAUSAL PLANNER INSTRUCTION (advisory only — you may override with justification):
- proceed: Causal engine is confident. Any framework is available.
- ask_first: Causal engine is uncertain. Prefer probe or reflect. Avoid reframe.
- hold: Contradictory signals detected. Treat as advice. You can still select a technique if trajectory gives you a clearer picture — justify any override in rationale.

TRAJECTORY SIGNALS:
- escalating / sustained_negative: Reduce ambition. No reframe. Prefer validate + person_centred or none.
- arousal_spike: High arousal this turn. Active listening only. No CBT.
- de-escalating: Student stabilising. Solution_focused is now available.
- emotion_shift: Sharp transition occurred. Prefer reflect before committing to technique.
- suppression: Student may be underreporting distress. Probe gently.
- current_valence: A strongly negative current valence (< -0.6) should weight toward person_centred regardless of cause_type.
- cross_session_baseline: If current valence is significantly below baseline, treat as meaningful deterioration.

CLARIFYING QUESTION:
- Inherit the causal engine's clarifying_question by default (set clarifying_question_overridden=false).
- Override ONLY if your chosen strategy requires a more specific question.
- A good override is concrete: "You mentioned stopping your classes — was that this week or longer?" not "Can you tell me more?"
- If strategy is not probe and causal question is adequate, always inherit.

ESCALATION:
- Set escalate_to_safety=true if: sustained_negative AND arousal is high, OR if global_cause contains themes of hopelessness, self-harm, or harm to others.
- escalation_reason must be a brief clinical note, not a restatement of the flag.

RESPONSE DIRECTIVE:
- One to two sentences. Tells TRACE exactly what to do.
- Name the framework and strategy explicitly.
- Do NOT write the actual response — only the directive.
- Example: "Use person-centred validation to acknowledge the student's grief without interpretation. Reflect the sense of loss before asking anything."

OUTPUT SCHEMA (valid JSON only — no markdown, no preamble):
{
  "framework": "cbt|mi|solution_focused|person_centred|none",
  "strategy": "validate|reflect|reframe|probe|psychoeducate|active_listen|motivate|solution_elicit",
  "planner_confidence": "high|medium|low",
  "rationale": "<2-3 sentences explaining framework and strategy selection>",
  "clarifying_question": "<specific question or null>",
  "clarifying_question_overridden": true|false,
  "response_directive": "<1-2 sentence directive for TRACE>",
  "escalate_to_safety": true|false,
  "escalation_reason": "<brief clinical note or null>"
}"""


# ─── Input formatter ──────────────────────────────────────────────────────────

def _format_input(inp: PlannerInput) -> str:
    shift_str = ""
    if inp.shift_events:
        parts = [
            f"{s['from_emotion']} → {s['to_emotion']} (Δ{s['magnitude']:.2f})"
            for s in inp.shift_events[-2:]
        ]
        shift_str = f"\nRecent emotion shifts: {'; '.join(parts)}"

    baseline_str = ""
    if inp.cross_session_baseline is not None:
        delta = inp.current_valence - inp.cross_session_baseline
        rel   = "above" if delta >= 0 else "below"
        baseline_str = (
            f"\nCross-session baseline: {inp.cross_session_baseline:+.2f} "
            f"(current {abs(delta):.2f} {rel} baseline)"
        )

    return f"""Student message: "{inp.text}"

--- Emotion Classification ---
Top emotion: {inp.top_emotion} ({inp.emotion_confidence:.2f})
Top 3: {json.dumps(inp.top_3_emotions)}

--- Causal Analysis ---
Cause type: {inp.cause_type}
Causal confidence: {inp.causal_confidence_score:.2f} ({inp.causal_confidence_category})
Global cause: {inp.global_cause}
Causal chain: {json.dumps(inp.causal_chain)}
Causal planner instruction (advisory): {inp.causal_planner_instruction}
Inherited clarifying question: {inp.clarifying_question or "none"}

--- Session Trajectory ({inp.turn_count} turns) ---
Trajectory flag: {inp.trajectory_flag}
Valence direction: {inp.valence_direction}
Current valence: {inp.current_valence:+.2f}
Current arousal: {inp.current_arousal}{shift_str}{baseline_str}

Make your strategic planning decision."""


# ─── Fallback ─────────────────────────────────────────────────────────────────

def _safe_fallback(inp: PlannerInput, error: str) -> PlannerOutput:
    return PlannerOutput(
        framework=TherapeuticFramework.PERSON_CENTRED,
        strategy=ResponseStrategy.VALIDATE,
        planner_confidence=PlannerConfidence.LOW,
        rationale="Fallback due to planner error. Defaulting to person-centred validation.",
        clarifying_question=inp.clarifying_question,
        clarifying_question_overridden=False,
        response_directive=(
            "Use person-centred validation. Acknowledge what the student has shared "
            "warmly and without judgement. Do not probe or reframe."
        ),
        escalate_to_safety=False,
        error=error,
    )


# ─── Sync planner (no RAG) ────────────────────────────────────────────────────

def plan(inp: PlannerInput) -> PlannerOutput:
    """
    Sync LLM call. Returns PlannerOutput without RAG enrichment.
    Available for unit testing without Qdrant.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _format_input(inp)},
            ],
            temperature=0.15,
            max_tokens=600,
        )

        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        start, end = raw.index("{"), raw.rindex("}") + 1
        data = json.loads(raw[start:end])

        # ── Enum validation ───────────────────────────────────────────────────
        try:
            framework = TherapeuticFramework(data["framework"])
        except (ValueError, KeyError):
            framework = TherapeuticFramework.PERSON_CENTRED

        try:
            strategy = ResponseStrategy(data["strategy"])
        except (ValueError, KeyError):
            strategy = ResponseStrategy.VALIDATE

        try:
            confidence = PlannerConfidence(data["planner_confidence"])
        except (ValueError, KeyError):
            confidence = PlannerConfidence.LOW

        # ── Clarifying question ───────────────────────────────────────────────
        llm_question = data.get("clarifying_question")
        inherited    = inp.clarifying_question
        overridden   = bool(
            data.get("clarifying_question_overridden", False)
            or (llm_question and llm_question != inherited)
        )
        final_question = llm_question or inherited

        # ── Hard escalation guard ─────────────────────────────────────────────
        # Force escalation if sustained negative + high arousal regardless of LLM
        force_escalate = (
            inp.trajectory_flag in ("sustained_negative", "escalating")
            and inp.current_arousal == "high"
        )
        escalate = data.get("escalate_to_safety", False) or force_escalate
        escalation_reason = data.get("escalation_reason")
        if force_escalate and not escalation_reason:
            escalation_reason = (
                f"Auto-escalated: trajectory_flag={inp.trajectory_flag}, "
                f"arousal={inp.current_arousal}, valence={inp.current_valence:+.2f}"
            )

        # ── Downgrade strategy if escalating ─────────────────────────────────
        if escalate and strategy not in (
            ResponseStrategy.ACTIVE_LISTEN, ResponseStrategy.VALIDATE
        ):
            strategy  = ResponseStrategy.ACTIVE_LISTEN
            framework = TherapeuticFramework.NONE

        return PlannerOutput(
            framework=framework,
            strategy=strategy,
            planner_confidence=confidence,
            rationale=data.get("rationale", ""),
            clarifying_question=final_question,
            clarifying_question_overridden=overridden,
            response_directive=data.get("response_directive", ""),
            escalate_to_safety=escalate,
            escalation_reason=escalation_reason,
        )

    except Exception as e:
        return _safe_fallback(inp, str(e))


# ─── Async planner with RAG ───────────────────────────────────────────────────

async def plan_async(inp: PlannerInput) -> PlannerOutput:
    """
    Async entry point. Runs sync plan() then triggers RAG if
    strategy=psychoeducate. Called by the orchestrator.
    """
    from rag.rag_pipeline import run_rag

    planning_started = perf_counter()
    output = plan(inp)
    logger.info(
        "Planner timing | stage=plan strategy=%s duration_ms=%.1f",
        output.strategy.value,
        (perf_counter() - planning_started) * 1000,
    )

    if (
        output.strategy == ResponseStrategy.PSYCHOEDUCATE
        and not output.error
        and not output.escalate_to_safety
    ):
        rag_started = perf_counter()
        rag_result = await run_rag(inp)
        logger.info(
            "Planner timing | stage=rag duration_ms=%.1f retrieval_successful=%s summary_present=%s",
            (perf_counter() - rag_started) * 1000,
            rag_result.retrieval_successful,
            bool(rag_result.summary),
        )

        if rag_result.summary:
            enriched_directive = (
                f"{output.response_directive} "
                f"Incorporate the following university support information "
                f"naturally into the response: {rag_result.summary}"
            )
            output = output.model_copy(update={
                "response_directive":     enriched_directive,
                "kb_context":             rag_result.summary,
                "kb_sources":             rag_result.sources,
                "kb_retrieval_attempted": True,
            })
        else:
            output = output.model_copy(update={
                "kb_retrieval_attempted": True,
            })

    logger.info(
        "Planner timing | stage=total duration_ms=%.1f final_strategy=%s",
        (perf_counter() - planning_started) * 1000,
        output.strategy.value,
    )
    return output
