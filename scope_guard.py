"""
scope_guard.py — EmpathAI scope guard agent.

Sits between MIND-SAFE Layer 1 (crisis gate) and the full orchestrator pipeline.
Returns a binary in_scope decision with category and redirect message.

Pipeline position:
    Layer 1 crisis gate → scope_guard → [blocked: return redirect] → [pass: orchestrator]

Model: llama-3.1-8b-instant
- Binary classification task — 8b is sufficient, 70b would be wasteful here
- Temperature 0.0 — deterministic, no creativity needed
- max_tokens 200 — category + reasoning only, no prose generation
"""

import json
import os
import logging
from llm import get_llm_client
from contracts.scope_guard_contract import (
    ScopeGuardInput,
    ScopeGuardOutput,
    ScopeCategory,
    IN_SCOPE_CATEGORIES,
)

_client, MODEL = get_llm_client()
_SCOPE_MODEL = os.getenv("GROQ_SCOPE_MODEL", "llama-3.1-8b-instant")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a scope classifier for a student mental health support chatbot deployed at Kenyan universities.

Your task: decide whether a student's message is within scope for a mental health counselling service.

## In-scope categories (pipeline should proceed)
- emotional_distress: grief, anxiety, low mood, shame, fear, overwhelm, hopelessness
- academic_distress: exam pressure, fear of failure, burnout, imposter syndrome — the ACADEMIC topic is the surface; the EMOTIONAL weight is the referent
- relational_distress: family tension, friendship conflict, romantic difficulty, loneliness, betrayal
- identity_distress: belonging, self-worth, purpose, cultural identity, gender, faith
- wellbeing: sleep, motivation, appetite changes, general coping, "I haven't been okay"
- ambiguous_in_scope: you are uncertain but the message carries emotional weight — default to this when in doubt

## Out-of-scope categories (return redirect)
- academic_task: write my essay, summarise this article, solve this equation, proofread my work
- factual_query: what is X, when did Y happen, how does Z work, general knowledge
- general_conversation: pure greetings with no distress signal ("hey", "what's up", "how are you" — note: "how are you" FROM the student about themselves may be in scope)
- entertainment: jokes, games, stories, roleplay
- technical_request: code help, app troubleshooting, device issues

## Critical rules — read carefully

RULE 1 — BE GENEROUS, NOT RESTRICTIVE.
A student saying "I can't focus on my essay" is about academic distress, not an academic task request. The essay is the surface; the inability to cope is the referent. Default to in-scope when there is any emotional weight in the message.

RULE 2 — TOPIC ≠ TASK.
Mentioning an academic topic does not make a message an academic task. "I'm falling behind and I don't know what to do" is emotional_distress even though it's about academics.

RULE 3 — ONLY BLOCK CLEAR TASK REQUESTS.
Block only messages that are unambiguously requesting you to perform a task you cannot do: write, solve, explain factually, fix technically. "Can you help me?" alone is never out-of-scope.

RULE 4 — SHORT MESSAGES DEFAULT IN-SCOPE.
Very short messages ("I don't know", "I'm tired", "nothing") carry no sufficient signal to block. Classify as ambiguous_in_scope and let the pipeline handle them.

RULE 5 — KENYAN CONTEXT.
Messages may include Swahili, Sheng, or code-switching. Apply the same logic regardless of language. "Sijui nifanye nini" (I don't know what to do) is emotional_distress.

## Output format — JSON only, no other text

{
  "category": "<one of the category values above>",
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<one sentence explaining the decision>"
}"""


# ---------------------------------------------------------------------------
# Few-shot examples (injected as user/assistant turns)
# ---------------------------------------------------------------------------

_FEW_SHOT: list[dict] = [
    {
        "role": "user",
        "content": "Can you write the introduction to my psychology essay on attachment theory?"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "category": "academic_task",
            "confidence": 0.97,
            "reasoning": "Explicit request to produce academic writing — task request, not distress."
        })
    },
    {
        "role": "user",
        "content": "I keep freezing when I sit down to study. I don't know what's wrong with me."
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "category": "academic_distress",
            "confidence": 0.93,
            "reasoning": "Academic surface but the student is expressing self-doubt and functional impairment."
        })
    },
    {
        "role": "user",
        "content": "What year did Kenya gain independence?"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "category": "factual_query",
            "confidence": 0.99,
            "reasoning": "Pure factual question with no emotional content."
        })
    },
    {
        "role": "user",
        "content": "My mum keeps calling and I don't want to pick up anymore."
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "category": "relational_distress",
            "confidence": 0.91,
            "reasoning": "Avoidance of parental contact — relational tension with emotional weight."
        })
    },
    {
        "role": "user",
        "content": "Sijui"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "category": "ambiguous_in_scope",
            "confidence": 0.60,
            "reasoning": "Short Swahili message ('I don't know') — insufficient signal to block, may carry unexpressed distress."
        })
    },
    {
        "role": "user",
        "content": "Tell me a joke"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "category": "entertainment",
            "confidence": 0.98,
            "reasoning": "No emotional content — explicit entertainment request."
        })
    },
    {
        "role": "user",
        "content": "I'm fine I just wanted to ask if you can fix this Python error for me"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "category": "technical_request",
            "confidence": 0.95,
            "reasoning": "Technical task request; 'I'm fine' is a social opener, not a distress signal."
        })
    },
    {
        "role": "user",
        "content": "I feel like I don't belong here. Like everyone else knows what they're doing."
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "category": "identity_distress",
            "confidence": 0.95,
            "reasoning": "Belonging and comparative self-worth — identity distress with imposter syndrome pattern."
        })
    },
]


# ---------------------------------------------------------------------------
# Main classification function
# ---------------------------------------------------------------------------

async def run_scope_guard(input: ScopeGuardInput) -> ScopeGuardOutput:
    """
    Classify a student message as in-scope or out-of-scope.
    Always returns is_in_scope=True — out-of-scope blocking has been removed.
    Still classifies and logs category for analytics/reference.
    """
    try:
        response_text = await _client.chat_completion(
            model=_SCOPE_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                *_FEW_SHOT,
                {"role": "user", "content": input.text.strip()}
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = response_text

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as parse_error:
            logger.error(f"Scope guard JSON parse error for text='{input.text[:50]}...': {parse_error}. Raw: {raw[:200]}")
            return ScopeGuardOutput(
                is_in_scope=True,
                category=ScopeCategory.AMBIGUOUS_IN_SCOPE,
                confidence=0.5,
                reasoning=f"JSON parse error — defaulting in-scope. Error: {str(parse_error)[:80]}",
            )

        category = ScopeCategory(parsed["category"])
        confidence = float(parsed.get("confidence", 0.7))
        reasoning = parsed.get("reasoning", "")

        logger.info(
            f"Scope guard | text='{input.text[:50]}...' | category={category.value} | "
            f"confidence={confidence} | in_scope={category in IN_SCOPE_CATEGORIES} | "
            f"reasoning={reasoning}"
        )

    except Exception as e:
        return ScopeGuardOutput(
            is_in_scope=True,
            category=ScopeCategory.AMBIGUOUS_IN_SCOPE,
            confidence=0.5,
            reasoning=f"Scope guard error — defaulting in-scope. Error: {str(e)[:80]}",
        )

    is_in_scope = category in IN_SCOPE_CATEGORIES

    DISTRESS_KEYWORDS = {
        "overwhelm", "overwhelmed", "anxious", "anxiety", "depressed", "depression",
        "sad", "sadness", "stressed", "stress", "worried", "worry", "fear", "scared",
        "hopeless", "hopelessness", "suicidal", "suicide", "panic", "panic attack",
        "lonely", "loneliness", "grief", "grieving", "trauma", "traumatic", "abuse",
        "hurt", "pain", "suffering", "struggling", "struggle", "coping", "cope",
        "exhausted", "exhaustion", "burnout", "tired", "fatigue", "mental health",
        "therapy", "counselling", "counseling", "help me", "need help", "can't cope",
        "falling apart", "breaking down", "breakdown", "emotional", "feelings",
        "unhappy", "miserable", "devastated", "devastating", "heartbroken"
    }

    text_lower = input.text.lower()
    has_distress_keywords = any(keyword in text_lower for keyword in DISTRESS_KEYWORDS)

    if has_distress_keywords and not is_in_scope:
        category = ScopeCategory.EMOTIONAL_DISTRESS
        reasoning = f"Overridden model decision: contains distress keywords. Original: {reasoning}"
        logger.info(
            f"Scope guard override | text='{input.text[:50]}...' | "
            f"original_category={parsed['category']} | overridden to emotional_distress"
        )

    # Out-of-scope functionality has been removed — always proceed with pipeline.
    return ScopeGuardOutput(
        is_in_scope=True,
        category=category,
        confidence=confidence,
        reasoning=reasoning,
    )