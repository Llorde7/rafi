import os, json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

SYSTEM_PROMPT = """Return compact JSON for emotion classification using the GoEmotions 28-label taxonomy.

ALLOWED LABELS (exactly these 28):
admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity,
desire, disappointment, disapproval, disgust, embarrassment, excitement, fear,
gratitude, grief, joy, love, nervousness, neutral, optimism, pride, realization,
relief, remorse, sadness, surprise

RULES:
1. Return JSON only with keys: translation, top_3, reasoning.
2. Use exactly 3 emotions from the allowed list.
3. Confidence scores must sum to 1.0.
4. Keep reasoning under 18 words.
5. If the text is already English, set translation to an empty string."""

def compress_history(history: list[dict]) -> str:
    if not history:
        return ""
    lines = []
    for h in history[-2:]:
        top = h["top_3"][0]["emotion"] if h.get("top_3") else "neutral"
        conf = h["top_3"][0]["confidence"] if h.get("top_3") else 0.0
        lines.append(f'"{h["text"][:40]}" -> {top} {conf:.2f}')
    return "Context: " + " | ".join(lines)

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
        temperature=0,
        max_tokens=96,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    start, end = raw.index("{"), raw.rindex("}") + 1
    return json.loads(raw[start:end])
