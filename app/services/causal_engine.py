import os, json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a causal analysis engine for a mental health chatbot.

You receive:
- The user's current message
- The top 3 classified emotions with confidence scores and reasoning
- Session history (prior turns with their emotions and causal analyses)

Your job is dual-granularity causal analysis:

FINE-GRAINED — trigger span extraction:
Identify the exact phrases or words in the user's message that triggered each emotion.
A trigger span is a verbatim substring of the user's message.
Each span must have a causal weight (0.01–0.99) indicating how strongly it drove the emotion.
Multiple spans can trigger the same emotion.

COARSE-GRAINED — global cause summary:
One sentence explaining the underlying psychological cause of the dominant emotion,
going beyond the surface words to the deeper reason (loss, fear of rejection, burnout, etc).

CAUSAL CHAIN:
A sequence of 2–4 steps showing how one thing led to another led to the emotion.
Format: ["event/thought A", "which caused B", "which produced emotion C"]
Ground each step in the session history or the current message — do not speculate.
If insufficient history exists for a chain, return a 2-step chain from the current message only.

TEMPORAL PATTERN:
Look across the session history for recurring triggers — the same theme, word type,
or situation appearing in multiple turns.
If a pattern exists, name it concisely (e.g. "avoidance of help-seeking", "loss reminders",
"performance anxiety"). If no pattern yet (fewer than 3 turns), return null.

RULES:
1. Trigger spans must be verbatim substrings of the user message. Never paraphrase.
2. Causal chain steps must be grounded — no speculation beyond what the text implies.
3. Global cause must name a psychological construct, not just restate the emotion.
4. Return ONLY valid JSON. No markdown, no preamble.

OUTPUT SCHEMA:
{
  "trigger_spans": [
    {"span": "<verbatim substring>", "emotion": "<label>", "weight": <float>}
  ],
  "global_cause": "<one sentence psychological cause>",
  "causal_chain": ["<step 1>", "<step 2>", "<step 3 optional>", "<step 4 optional>"],
  "temporal_pattern": "<pattern name>" | null
}

EXAMPLES:

User message: "I keep going to call him and then I remember"
Classified: grief (0.65)
History: []
Output: {"trigger_spans":[{"span":"keep going to call him","emotion":"grief","weight":0.72},{"span":"then I remember","emotion":"grief","weight":0.58}],"global_cause":"Habitual attachment behaviour interrupted by cognitive reality of loss produces acute grief.","causal_chain":["Close relationship created habitual communication pattern","Person is now absent","Reflex fires before conscious memory catches up → grief"],"temporal_pattern":null}

User message: "I typed an email and deleted it three times"
Classified: nervousness (0.60)
History: [{"text":"I am too embarrassed to talk to the lecturer","top_emotion":"fear"}]
Output: {"trigger_spans":[{"span":"deleted it three times","emotion":"nervousness","weight":0.81},{"span":"typed an email","emotion":"nervousness","weight":0.34}],"global_cause":"Fear of negative evaluation creating approach-avoidance conflict manifesting as repeated incomplete action.","causal_chain":["Perceived authority gap with lecturer","Fear of confirming inadequacy","Approach-avoidance loop → repeated deletion → nervousness"],"temporal_pattern":"help-seeking avoidance — embarrassment and deletion both reflect reluctance to initiate contact with authority"}"""


def format_history(history: list[dict]) -> str:
    if not history:
        return ""
    lines = ["Session history (most recent last):"]
    for h in history[-6:]:
        top = h["top_3"][0]["emotion"] if h.get("top_3") else "neutral"
        conf = h["top_3"][0]["confidence"] if h.get("top_3") else 0.0
        chain = ""
        if h.get("causal_analysis") and h["causal_analysis"].get("causal_chain"):
            chain = " | chain: " + " → ".join(h["causal_analysis"]["causal_chain"][:2])
        pattern = ""
        if h.get("causal_analysis") and h["causal_analysis"].get("temporal_pattern"):
            pattern = f' | pattern: {h["causal_analysis"]["temporal_pattern"]}'
        lines.append(f'- "{h["text"][:70]}" → {top} ({conf:.2f}){chain}{pattern}')
    return "\n".join(lines)


def analyse(text: str, classification: dict, history: list[dict] = None) -> dict:
    history = history or []
    top_3_str = json.dumps(classification.get("top_3", []))
    reasoning = classification.get("reasoning", "")
    history_str = format_history(history)

    user_content = f"""User message: "{text}"

Classified emotions: {top_3_str}
Classification reasoning: {reasoning}

{history_str}

Perform dual-granularity causal analysis."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0.1,
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        start, end = raw.index("{"), raw.rindex("}") + 1
        result = json.loads(raw[start:end])

        # validate spans are actual substrings — drop hallucinated ones
        result["trigger_spans"] = [
            s for s in result.get("trigger_spans", [])
            if s.get("span") and s["span"] in text
        ]
        return result

    except Exception as e:
        return {
            "trigger_spans": [],
            "global_cause": f"Analysis failed: {e}",
            "causal_chain": [],
            "temporal_pattern": None,
            "error": str(e)
        }