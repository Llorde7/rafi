import json
import logging
import os

from groq import Groq
from dotenv import load_dotenv

from contracts.causal_contract import (
    CausalInput, CausalOutput, TriggerSpan,
    ConfidenceCategory, CauseType, PlannerInstruction
)

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a causal analysis engine for a mental health chatbot pipeline.

You receive a user message, its classified emotions, and session history.
Your output feeds a strategic response planner — not a human.

YOUR RULES:
1. Never assume. If the text is ambiguous, say so explicitly.
2. If you are not confident, produce a clarifying_question the planner will ask the user.
3. Trigger spans must be verbatim substrings of the user message. Never paraphrase.
4. Causal chain steps must be grounded in the text or history. No speculation.
5. Global cause names a psychological construct — not a restatement of the emotion.
6. Return ONLY valid JSON matching the schema exactly. No markdown, no preamble.

CONFIDENCE SCORING:
- confident (>=0.75): clear causal evidence in text and/or history
- partial (0.45-0.74): plausible but ambiguous between two readings
- insufficient (<0.45): text too sparse or session too early to diagnose
- contradictory: signals in text actively conflict with each other or with history
  (contradictory overrides the score — flag it regardless of numeric confidence)

CAUSE TYPES (tag the psychological territory — do not select a therapeutic technique):
- cognitive_distortion: distorted belief, catastrophising, all-or-nothing thinking, catastrophic predictions about outcomes
- avoidance_behaviour: avoidance, suppression, incomplete action, withdrawal, escape planning, proposing a temporary exit from a situation with intention to return — the student is moving away from the stressor rather than through it
- unresolved_loss: grief, absence, attachment disruption
- somatic_response: physical sensation carrying emotional signal
- interpersonal_conflict: relational tension, betrayal, neglect, boundary violation
- identity_threat: self-concept under threat, shame, inadequacy, role loss
- ambiguous: cannot determine from available text

AVOIDANCE vs COGNITIVE DISTORTION — common confusion:
- "I want to quit and come back later" = avoidance_behaviour (escape plan with return intention)
- "I'll never be able to handle this" = cognitive_distortion (distorted belief about capacity)
- "I'm just not cut out for this" = identity_threat (self-concept under threat)
- When a student proposes a concrete behavioural exit (quitting, withdrawing, taking a break), that is avoidance_behaviour regardless of the emotions present.

PLANNER INSTRUCTION:
- "proceed": confident — planner may select technique
- "ask_first": partial or insufficient — planner must ask clarifying_question first
- "hold": contradictory — planner must not select technique, use active listening

CLARIFYING QUESTION:
- Must be specific, not generic. "You mentioned X — did you mean Y or Z?" not "tell me more"
- Null ONLY when confident
- Must target the specific ambiguity or gap in evidence

OUTPUT SCHEMA:
{
  "confidence_score": <float 0.0-1.0>,
  "confidence_category": "confident|partial|insufficient|contradictory",
  "trigger_spans": [{"span": "<verbatim>", "emotion": "<label>", "weight": <float>}],
  "global_cause": "<one sentence psychological construct>",
  "causal_chain": ["<step 1>", "<step 2>", "<step 3 optional>", "<step 4 optional>"],
  "temporal_pattern": "<pattern name>" | null,
  "cause_type": "<cause type>",
  "clarifying_question": "<specific question>" | null,
  "planner_instruction": "proceed|ask_first|hold"
}

EXAMPLES:

Message: "I keep going to call him and then I remember"
Emotions: grief (0.65), sadness (0.25), realization (0.10)
History: []
Output: {"confidence_score":0.82,"confidence_category":"confident","trigger_spans":[{"span":"keep going to call him","emotion":"grief","weight":0.75},{"span":"then I remember","emotion":"realization","weight":0.60}],"global_cause":"Habitual attachment behaviour is being interrupted by conscious awareness of loss, producing acute grief.","causal_chain":["Repeated calling pattern formed during relationship","Person is now absent","Reflex fires before conscious memory catches up","Collision between habit and reality produces grief"],"temporal_pattern":null,"cause_type":"unresolved_loss","clarifying_question":null,"planner_instruction":"proceed"}

Message: "I'm fine"
Emotions: neutral (0.70), nervousness (0.20), disapproval (0.10)
History: []
Output: {"confidence_score":0.28,"confidence_category":"insufficient","trigger_spans":[],"global_cause":"Insufficient evidence to identify underlying cause from current message.","causal_chain":[],"temporal_pattern":null,"cause_type":"ambiguous","clarifying_question":"When you say you are fine — is that how things genuinely feel, or is that what feels easiest to say right now?","planner_instruction":"ask_first"}

Message: "I wanted to argue but I just sat there — though honestly part of me agreed with them"
Emotions: disappointment (0.45), anger (0.30), approval (0.25)
History: [{"text":"My manager gave my project away","top_emotion":"anger"}]
Output: {"confidence_score":0.51,"confidence_category":"contradictory","trigger_spans":[{"span":"wanted to argue","emotion":"anger","weight":0.65},{"span":"part of me agreed with them","emotion":"approval","weight":0.58},{"span":"just sat there","emotion":"avoidance_behaviour","weight":0.70}],"global_cause":"Simultaneous anger at the outcome and partial identification with the decision creates unresolvable internal conflict.","causal_chain":["Project removed without consultation produces anger","User simultaneously recognises some legitimacy in the decision","Conflicting impulses produce paralysis and silence"],"temporal_pattern":"passive suppression under authority — sitting still instead of advocating across two sessions","cause_type":"interpersonal_conflict","clarifying_question":"You mentioned part of you agreed with them — what part of their reasoning felt valid to you?","planner_instruction":"hold"}"""


def _format_history(history: list) -> str:
    if not history:
        return "Session history: none"
    lines = ["Session history (oldest first):"]
    for h in history:
        pattern = f" | pattern: {h.temporal_pattern}" if h.temporal_pattern else ""
        cause = f" | cause_type: {h.cause_type}" if h.cause_type else ""
        lines.append(
            f'- "{h.text[:70]}" → {h.top_emotion} ({h.confidence:.2f}){cause}{pattern}'
        )
    return "\n".join(lines)


def _derive_planner_instruction(
    category: ConfidenceCategory,
    clarifying_question: str | None
) -> PlannerInstruction:
    if category == ConfidenceCategory.CONTRADICTORY:
        return PlannerInstruction.HOLD
    if category in (ConfidenceCategory.PARTIAL, ConfidenceCategory.INSUFFICIENT):
        return PlannerInstruction.ASK_FIRST
    return PlannerInstruction.PROCEED


def _call_model(messages: list[dict], max_tokens: int) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def _extract_json_object(raw: str) -> dict:
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    start, end = cleaned.index("{"), cleaned.rindex("}") + 1
    return json.loads(cleaned[start:end])


def _build_repair_messages(messages: list[dict], raw: str, error: Exception) -> list[dict]:
    return messages + [
        {"role": "assistant", "content": raw},
        {
            "role": "user",
            "content": (
                "Repair the previous response. Return only valid JSON matching the causal "
                "analysis schema exactly. Keep trigger spans verbatim substrings from the "
                f'user message. Validation error: {error}'
            ),
        },
    ]


def _validate_causal_payload(input: CausalInput, data: dict) -> CausalOutput:
    required_fields = [
        "confidence_score",
        "confidence_category",
        "global_cause",
        "cause_type",
        "planner_instruction",
    ]
    missing = [field for field in required_fields if field not in data]
    if missing:
        raise ValueError(f"LLM response missing required fields: {missing}")

    validated_spans = [
        TriggerSpan(**span)
        for span in data.get("trigger_spans", [])
        if span.get("span") and span["span"] in input.text
    ]

    try:
        category = ConfidenceCategory(data["confidence_category"])
    except ValueError:
        category = ConfidenceCategory.INSUFFICIENT

    try:
        cause_type = CauseType(data["cause_type"])
    except ValueError:
        cause_type = CauseType.AMBIGUOUS

    clarifying_question = data.get("clarifying_question")
    if category == ConfidenceCategory.CONFIDENT:
        clarifying_question = None
    if category != ConfidenceCategory.CONFIDENT and not clarifying_question:
        clarifying_question = "Could you tell me a bit more about what's been happening for you?"

    planner_instruction = _derive_planner_instruction(category, clarifying_question)

    return CausalOutput(
        confidence_score=round(float(data.get("confidence_score", 0.5)), 3),
        confidence_category=category,
        trigger_spans=validated_spans,
        global_cause=data.get("global_cause", ""),
        causal_chain=data.get("causal_chain", []),
        temporal_pattern=data.get("temporal_pattern"),
        cause_type=cause_type,
        clarifying_question=clarifying_question,
        planner_instruction=planner_instruction,
    )


def analyse(input: CausalInput) -> CausalOutput:
    """
    Receives CausalInput contract.
    Returns CausalOutput contract.
    No knowledge of classifier internals.
    """
    emotions_str = json.dumps(input.top_emotions)
    history_str = _format_history(input.session_history)
    trajectory_str = input.trajectory_context or "Emotional trajectory: not available."

    user_content = f"""User message: "{input.text}"
Classified emotions: {emotions_str}
Classification reasoning: {input.reasoning}

{history_str}

{trajectory_str}

Perform causal analysis."""

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        raw = _call_model(messages, max_tokens=500)
        try:
            return _validate_causal_payload(input, _extract_json_object(raw))
        except Exception as exc:
            logger.warning("Causal parse/validation failed on first attempt: %s", exc)
            repair_messages = _build_repair_messages(messages, raw, exc)
            repaired_raw = _call_model(repair_messages, max_tokens=550)
            return _validate_causal_payload(input, _extract_json_object(repaired_raw))

    except Exception as e:
        logger.exception("Causal analysis failed after retry/repair: %s", e)
        return CausalOutput(
            confidence_score=0.0,
            confidence_category=ConfidenceCategory.INSUFFICIENT,
            trigger_spans=[],
            global_cause="Causal analysis failed.",
            causal_chain=[],
            temporal_pattern=None,
            cause_type=CauseType.AMBIGUOUS,
            clarifying_question="Could you tell me a bit more about what's been going on for you?",
            planner_instruction=PlannerInstruction.ASK_FIRST,
        )
