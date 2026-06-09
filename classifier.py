import json
import os
from typing import Optional
from pydantic import BaseModel
from contracts.classifier_contract import ClassifierInput, EmotionScore, GoEmotionLabel
from contracts.situation_contract import SituationType, SituationOutput
from llm._provider import get_async_chat_client, get_default_model
from llm.usage import instrumented_create

# Model is provider-aware: OpenRouter -> meta-llama/llama-3.3-70b-instruct:free
#                            Groq      -> llama-3.3-70b-versatile
MODEL = get_default_model()


# ---------------------------------------------------------------------------
# Lazy client — provider is selected by LLM_PROVIDER env var
#   openrouter (default) | groq | openai | gemini
# ---------------------------------------------------------------------------

def _get_client():
    return get_async_chat_client()


# ---------------------------------------------------------------------------
# Fused output contract — classifier + situation in one call
# ---------------------------------------------------------------------------

class Stage1Output(BaseModel):
    # Classifier fields
    text:         str
    translation:  Optional[str]
    top_3:        list[EmotionScore]
    reasoning:    str
    # Situation fields
    situation_type:          SituationType
    situation_summary:       str
    has_concrete_deadline:   bool
    has_external_referents:  bool
    situation_confidence:    float


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert clinical psychologist and emotion analyst specialising in university student mental health in Kenya.

You receive a student's message and recent session history.
You must produce two analyses in a single JSON response: emotion classification and situation assessment.

═══════════════════════════════════════════
PART 1 — EMOTION CLASSIFICATION
═══════════════════════════════════════════

Use the GoEmotions 28-label taxonomy:
admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity,
desire, disappointment, disapproval, disgust, embarrassment, excitement, fear,
gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief,
remorse, sadness, surprise, neutral

ONLY USE THE EXACT LOWERCASE LABELS ABOVE. Do not invent or return labels not in
this list. If none of the labels fits exactly, pick the nearest label from the
list (for example, map "frustration" -> "annoyance"). If truly uncertain,
return "neutral".

IMPLICIT EMOTION GUIDE
Implicit emotions are carried by behaviour and situation, not feeling words.
Classify the underlying emotional state — not the surface description.
Examples:
- "I keep going to call him and then I remember" → grief, sadness
- "I drove past the hospital and couldn't look" → grief, fear, avoidance
- "I just need to get through the next few days" → nervousness, determination

SWAHILI / SHENG / CODE-SWITCHING
If the message is not in English, translate it first, then classify the English meaning.
Populate the translation field with the English translation.

Output: top_3 emotions with confidence scores summing to 1.0. reasoning field: explain the classification in 1–2 sentences.

═══════════════════════════════════════════
PART 2 — SITUATION ASSESSMENT
═══════════════════════════════════════════

Classify the concrete circumstances of the student's situation — not their emotional state.

SITUATION TYPES:
- acute_task_crisis: imminent deadline, incomplete or failing deliverable, presentation/exam tomorrow
- convergent_overload: multiple concurrent demands competing for limited time or capacity
- interpersonal_event: a specific named interaction or conflict with another person
- loss_or_absence: grief, separation, something or someone no longer present
- identity_pressure: self-worth, shame, role or capability under threat
- chronic_low_mood: persistent low state with no clear acute precipitating event
- ambiguous: message does not provide enough situational information

situation_summary: ONE sentence describing what is concretely happening.
  CORRECT: "Student has a project presentation tomorrow and the project is incomplete."
  WRONG: "Student is feeling anxious and overwhelmed."

has_concrete_deadline: true ONLY if a specific time pressure is named (tomorrow, this week, in 2 days, etc.)
has_external_referents: true if tasks, deliverables, people, places, or events are named
situation_confidence: your confidence in the situation classification, 0.0–1.0

═══════════════════════════════════════════
OUTPUT FORMAT — STRICT
═══════════════════════════════════════════

Respond with a single JSON object. No preamble. No markdown fences.

{
  "translation": null,
  "top_3": [
    {"emotion": "<label>", "confidence": <float>},
    {"emotion": "<label>", "confidence": <float>},
    {"emotion": "<label>", "confidence": <float>}
  ],
  "reasoning": "<1-2 sentences>",
  "situation_type": "<one of the 7 types>",
  "situation_summary": "<one sentence — concrete facts only>",
  "has_concrete_deadline": <true|false>,
  "has_external_referents": <true|false>,
  "situation_confidence": <float>
}
"""


def _build_context_block(history: list[dict]) -> str:
    if not history:
        return ""
    lines = ["RECENT SESSION HISTORY (compressed):"]
    for h in history[-4:]:
        lines.append(
            f"Turn {h.get('turn_index', '?')}: {h.get('text', '')} "
            f"[emotion: {h.get('top_emotion', '?')}, "
            f"situation: {h.get('situation_type', '?')}]"
        )
    return "\n".join(lines)


async def classify(
    inp: ClassifierInput,
    classifier_history: list[dict],
) -> Stage1Output:
    client = _get_client()
    context = _build_context_block(classifier_history)
    user_content = inp.text
    if context:
        user_content = f"{context}\n\nCURRENT MESSAGE:\n{inp.text}"

    response = await instrumented_create(
        stage="classifier",
        client=client,
        model=MODEL,
        temperature=0.1,
        max_tokens=600,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    data = json.loads(raw.strip())
    # Normalize and force emotions to the canonical 28-label GoEmotions set.
    allowed_labels = [e.value for e in GoEmotionLabel]

    NORMALIZATION_MAP = {
        "frustration": "annoyance",
        "frustrated": "annoyance",
        "stressed": "nervousness",
        "overwhelmed": "nervousness",
        "stress": "nervousness",
    }

    def normalize_label(label: str) -> str:
        if not isinstance(label, str):
            return "neutral"
        l = label.strip().lower()
        if l in allowed_labels:
            return l
        if l in NORMALIZATION_MAP:
            return NORMALIZATION_MAP[l]
        for a in allowed_labels:
            if a in l or l in a:
                return a
        return "neutral"

    # Map incoming labels and aggregate confidences for duplicates
    agg: dict[str, float] = {}
    for item in data.get("top_3", []):
        em = normalize_label(item.get("emotion"))
        conf = float(item.get("confidence", 0.0))
        agg[em] = agg.get(em, 0.0) + conf

    # Build sorted top list and ensure exactly 3 entries
    items = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
    # Fill up with 'neutral' or other labels with 0.0 if needed
    if len(items) < 3:
        for a in allowed_labels:
            if a not in agg:
                items.append((a, 0.0))
            if len(items) >= 3:
                break

    top3 = items[:3]

    # Renormalize confidences to sum to 1.0
    total = sum(conf for _, conf in top3)
    if total <= 0.0:
        # If model returned zero/confidences collapsed, distribute uniformly
        normed = [(label, 1.0 / 3.0) for label, _ in top3]
    else:
        normed = [(label, conf / total) for label, conf in top3]

    top_3_normalized = [{"emotion": label, "confidence": conf} for label, conf in normed]

    return Stage1Output(
        text=inp.text,
        translation=data.get("translation"),
        top_3=[EmotionScore(**e) for e in top_3_normalized],
        reasoning=data.get("reasoning", ""),
        situation_type=SituationType(data["situation_type"]),
        situation_summary=data["situation_summary"],
        has_concrete_deadline=bool(data["has_concrete_deadline"]),
        has_external_referents=bool(data["has_external_referents"]),
        situation_confidence=float(data["situation_confidence"]),
    )