import os, json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a fast emotion classifier using the GoEmotions 28-label taxonomy.

ALLOWED LABELS (exactly these 28):
admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity,
desire, disappointment, disapproval, disgust, embarrassment, excitement, fear,
gratitude, grief, joy, love, nervousness, neutral, optimism, pride, realization,
relief, remorse, sadness, surprise

RULES:
1. Top 3 emotions only. Never invent labels.
2. Confidence scores: 0.01–0.99, sum to exactly 1.0.
3. Classify implied behaviour, not literal words.
4. Return ONLY valid JSON. No markdown, no preamble.

IMPLICIT GUIDE:
repeated checking/rehearsing → nervousness/fear | two cups/cooking for two → grief
letting calls ring out → remorse | smiling at nothing → joy
credit taken for your work → anger | lights on all night → fear

EXAMPLES:
{"top_3":[{"emotion":"grief","confidence":0.65},{"emotion":"sadness","confidence":0.25},{"emotion":"neutral","confidence":0.10}],"reasoning":"Reflex of reaching for someone gone signals grief."}
{"top_3":[{"emotion":"nervousness","confidence":0.60},{"emotion":"fear","confidence":0.25},{"emotion":"confusion","confidence":0.15}],"reasoning":"Repeated composing and deleting signals hesitation from anxiety."}"""


def compress_history(history: list[dict]) -> str:
    if not history:
        return ""
    lines = []
    for h in history[-4:]:
        top = h["top_3"][0]["emotion"] if h.get("top_3") else "neutral"
        conf = h["top_3"][0]["confidence"] if h.get("top_3") else 0.0
        reasoning = h.get("reasoning", "")[:80]
        lines.append(f'- "{h["text"][:60]}" → {top} ({conf:.2f}): {reasoning}')
    return "Prior turns:\n" + "\n".join(lines)


def build_messages(text: str, history: list[dict]) -> list[dict]:
    context = compress_history(history)
    user_content = f'{context}\n\nClassify: "{text}"' if context else f'Classify: "{text}"'
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]


def classify(text: str, history: list[dict] = None) -> dict:
    messages = build_messages(text, history or [])
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=200,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    start, end = raw.index("{"), raw.rindex("}") + 1
    return json.loads(raw[start:end])
