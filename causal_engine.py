import json
import os
from typing import Optional
from groq import AsyncGroq
from contracts.causal_contract import (
    CausalInput, CausalOutput, TriggerSpan,
    ConfidenceCategory, CauseType, PlannerInstruction,
    CognitivePattern, BehavioralRisk,
)

MODEL  = "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# Lazy client — reads GROQ_API_KEY from env at first call, after load_dotenv()
# ---------------------------------------------------------------------------

_client: AsyncGroq | None = None

def _get_client() -> AsyncGroq:
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    return _client


SYSTEM_PROMPT = """You are a clinical psychologist specialising in university student mental health in Kenya.

You receive a student's message, their top emotions, and recent session history.
You must produce a causal analysis with cognitive pattern and behavioral risk assessment in a single JSON response.

═══════════════════════════════════════════
PART 1 — CAUSAL ANALYSIS
═══════════════════════════════════════════

confidence_score: 0.0–1.0 — your confidence in identifying the root cause
confidence_category:
  - confident (≥0.75): clear cause, proceed with intervention
  - partial (0.45–0.74): probable cause, ask a clarifying question
  - insufficient (<0.45): too little information
  - contradictory: signals conflict — hold

trigger_spans: verbatim substrings of the student's message that carry emotional weight.
  CRITICAL: every span MUST be a verbatim substring of the student's message.
  Do not paraphrase. Do not hallucinate. If uncertain, omit.

global_cause: the psychological construct driving the distress. NOT an emotion restatement.
  WRONG: "The student feels sad."
  CORRECT: "The student is applying all-or-nothing thinking to academic performance."

causal_chain: 2–4 grounded steps from trigger to current emotional state.

temporal_pattern: a recurring theme across the session — null if fewer than 3 turns.

cause_type — one of:
  cognitive_distortion: distorted belief, catastrophising, black-and-white thinking
  avoidance_behaviour: avoidance, suppression, incomplete action, withdrawal,
    escape planning, proposing a temporary exit with intention to return.
    "I want to quit and come back later" → avoidance_behaviour
    "I'll never be able to handle this" → cognitive_distortion
    "I'm just not cut out for this"     → identity_threat
  unresolved_loss: grief, absence, attachment disruption
  somatic_response: physical sensation carrying emotional signal
  interpersonal_conflict: relational tension, betrayal, neglect
  identity_threat: self-concept under threat, shame, role loss
  ambiguous: cannot determine from available text

clarifying_question: null ONLY when confidence_category = confident. Otherwise required.

planner_instruction:
  proceed    → confident
  ask_first  → partial or insufficient
  hold       → contradictory

═══════════════════════════════════════════
PART 2 — COGNITIVE PATTERN
═══════════════════════════════════════════

Identify the specific thinking error maintaining the student's distress.
This is more precise than cause_type — it names the cognitive mechanism.

cognitive_pattern — one of:
  catastrophising: treating the worst-case outcome as certain
  all_or_nothing: incomplete or imperfect = total failure (not partial progress)
  mind_reading: assuming others' negative judgment without evidence
  personalisation: attributing external events or others' behaviour to self
  should_statements: rigid internal rules generating shame or pressure ("I should be able to...")
  emotional_reasoning: treating a feeling as proof of a fact ("I feel stupid therefore I am")
  avoidance_rationalisation: constructing logical justification for avoidance behaviour
  none_identified: no clear cognitive distortion present

═══════════════════════════════════════════
PART 3 — BEHAVIORAL RISK
═══════════════════════════════════════════

Identify what the student is most likely to DO if nothing changes — their behavioural trajectory.

behavioral_risk — one of:
  shutdown_paralysis: stop acting entirely, freeze on the problem, unable to start
  avoidance: disengage from the stressor, delay, put it off further
  rumination_loop: continue processing emotionally without moving toward action
  impulsive_exit: make an abrupt exit decision (quit course, withdraw, leave situation)
  none_identified: no clear behavioural risk present

═══════════════════════════════════════════
OUTPUT FORMAT — STRICT
═══════════════════════════════════════════

Single JSON object. No preamble. No markdown fences.

{
  "confidence_score": <float>,
  "confidence_category": "<confident|partial|insufficient|contradictory>",
  "trigger_spans": [
    {"span": "<verbatim substring>", "emotion": "<label>", "weight": <float>}
  ],
  "global_cause": "<psychological construct — not emotion restatement>",
  "causal_chain": ["<step 1>", "<step 2>"],
  "temporal_pattern": "<string or null>",
  "cause_type": "<one of 7>",
  "cognitive_pattern": "<one of 8>",
  "behavioral_risk": "<one of 5>",
  "clarifying_question": "<string or null>",
  "planner_instruction": "<proceed|ask_first|hold>"
}
"""


def _validate_spans(spans: list[dict], text: str) -> list[TriggerSpan]:
    """Drop any span that is not a verbatim substring of the student's text."""
    valid = []
    for s in spans:
        if s.get("span", "") in text:
            valid.append(TriggerSpan(**s))
    return valid


def _build_context_block(history: list[dict]) -> str:
    if not history:
        return ""
    lines = ["RECENT SESSION HISTORY (compressed):"]
    for h in history[-4:]:
        lines.append(
            f"Turn {h.get('turn_index', '?')}: {h.get('text', '')} "
            f"[emotion: {h.get('top_emotion', '?')}, cause: {h.get('cause_type', '?')}]"
        )
    return "\n".join(lines)


async def analyse(inp: CausalInput) -> CausalOutput:
    client = _get_client()
    context = _build_context_block(
        [h.model_dump() for h in inp.session_history]
    )

    emotions_str = ", ".join(
        f"{e['emotion']} ({e['confidence']:.2f})" for e in inp.top_emotions
    )

    user_content = (
        f"{context}\n\n" if context else ""
    ) + (
        f"CURRENT MESSAGE: {inp.text}\n"
        f"TOP EMOTIONS: {emotions_str}\n"
        f"EMOTION REASONING: {inp.reasoning}\n"
        f"SITUATION TYPE: {inp.situation_type}"
    )

    response = await client.chat.completions.create(
        model=MODEL,
        temperature=0.1,
        max_tokens=700,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    data = json.loads(raw.strip())

    # Required field validation — fail loudly rather than silently
    required = [
        "confidence_score", "confidence_category", "trigger_spans",
        "global_cause", "causal_chain", "cause_type",
        "cognitive_pattern", "behavioral_risk", "planner_instruction",
    ]
    missing = [f for f in required if f not in data]
    if missing:
        raise ValueError(f"CausalEngine response missing required fields: {missing}")

    return CausalOutput(
        confidence_score=float(data["confidence_score"]),
        confidence_category=ConfidenceCategory(data["confidence_category"]),
        trigger_spans=_validate_spans(data["trigger_spans"], inp.text),
        global_cause=data["global_cause"],
        causal_chain=data["causal_chain"],
        temporal_pattern=data.get("temporal_pattern"),
        cause_type=CauseType(data["cause_type"]),
        cognitive_pattern=CognitivePattern(data["cognitive_pattern"]),
        behavioral_risk=BehavioralRisk(data["behavioral_risk"]),
        clarifying_question=data.get("clarifying_question"),
        planner_instruction=PlannerInstruction(data["planner_instruction"]),
    )