import os
from groq import AsyncGroq
from contracts.trace_contract import TraceInput, TraceOutput, TraceConfidence

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


# ---------------------------------------------------------------------------
# Base prompt — clinical identity and universal rules
# ---------------------------------------------------------------------------

BASE_PROMPT = """You are TRACE — a warm, skilled counsellor supporting university students in Kenya.

You are the only agent that speaks directly to the student.
You never mention the pipeline, analysis, labels, or strategies behind your response.

UNIVERSAL RULES:
- 1–3 sentences. Default to 2. Never exceed 3 unless the cluster genuinely requires it.
- One question maximum per response. Zero questions if the cluster does not require one.
- Never parrot the student's exact words back at them ("You said...", "It sounds like...")
- Never fabricate specifics not present in the student's message
- Never announce a technique ("Let me reframe that for you...")
- Never signal transitions ("Now that we've explored the emotional side, let's...")
- Responses must feel like a single natural human moment — not stitched together
- Swahili rule: respond in Swahili ONLY if the current turn's message is Swahili
"""


# ---------------------------------------------------------------------------
# Tone blocks — injected dynamically
# ---------------------------------------------------------------------------

TONE_BLOCKS = {
    "friendly": """
TONE: FRIENDLY
- Warm, peer-like, conversational rhythm
- Short sentences feel natural
- Vocabulary: accessible, no clinical language
- Opening examples: "That's a lot to carry.", "Okay, let's untangle this."
- Never stiff, never formal
""",
    "clinical": """
TONE: CLINICAL
- Measured, grounding, professional warmth
- Slightly longer sentences, deliberate pacing
- Vocabulary: precise but not cold
- Opening examples: "What you're describing points to real pressure.", "That's a meaningful observation."
- Never distant, never robotic
""",
}


# ---------------------------------------------------------------------------
# Cluster execution block — the core new capability
# ---------------------------------------------------------------------------

def _cluster_execution_block(trace_input: TraceInput) -> str:
    cluster = trace_input.technique_cluster
    techniques = cluster.techniques

    if len(techniques) == 1:
        t = techniques[0]
        return f"""
TECHNIQUE: {t.name} ({t.modality})
PURPOSE: {t.purpose}

EXECUTOR INSTRUCTION:
{cluster.executor_instruction}
"""

    # Multi-technique cluster
    technique_lines = []
    for i, t in enumerate(techniques):
        technique_lines.append(
            f"  {i+1}. {t.name} ({t.modality})\n"
            f"     Purpose: {t.purpose}\n"
            f"     Position: {t.sequence_note}"
        )
    techniques_str = "\n".join(technique_lines)

    return f"""
CLUSTER EXECUTION — {len(techniques)} techniques in sequence

{techniques_str}

EXECUTOR INSTRUCTION:
{cluster.executor_instruction}

CRITICAL — SEAM PREVENTION:
Your response must flow as a single natural human moment.
The {len(techniques)} techniques must not be visible as separate moves.

Anti-patterns to avoid:
- Announcing a technique: "Let me reframe that..."
- Signalling a transition: "Now that we've talked about X..."
- Padding between techniques: "That's really interesting. Also..."
- Separate sentences that obviously belong to different modes

The test: if a skilled counsellor read your response, they should see
one coherent move — not {len(techniques)} stitched together.

LENGTH RULE: {len(techniques)} techniques compresses to 2–3 sentences maximum.
More techniques does not mean a longer response.
"""


# ---------------------------------------------------------------------------
# Intent mode block — adjusts response register
# ---------------------------------------------------------------------------

def _intent_mode_block(intent_state: str) -> str:
    if intent_state == "action_seeking":
        return """
INTENT MODE: ACTION_SEEKING
The student has asked for practical help. They feel heard — do not re-validate at length.
Lead with the practical move. Brief acknowledgement (half a sentence) is permitted only
if it creates a natural bridge. The student's question deserves a direct answer.
"""
    if intent_state == "processing":
        return """
INTENT MODE: PROCESSING
The student needs to feel understood. Do not offer advice or steps.
Hold the space. Your response succeeds if the student feels less alone.
"""
    if intent_state == "transitioning":
        return """
INTENT MODE: TRANSITIONING
The student is moving from processing toward action.
Acknowledge briefly, then open the door — don't push through it.
"""
    return ""


# ---------------------------------------------------------------------------
# Context block — compressed session history
# ---------------------------------------------------------------------------

def _context_block(trace_input: TraceInput) -> str:
    if not trace_input.trace_history:
        return ""
    lines = ["RECENT EXCHANGE (last 3 turns):"]
    for turn in trace_input.trace_history[-3:]:
        lines.append(f"  Student: {turn.student_message}")
        lines.append(f"  TRACE:   {turn.trace_response}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

async def generate(inp: TraceInput) -> TraceOutput:
    client = _get_client()
    tone_block    = TONE_BLOCKS.get(inp.tone_preference.value, TONE_BLOCKS["friendly"])
    cluster_block = _cluster_execution_block(inp)
    intent_block  = _intent_mode_block(inp.intent_state)
    context_block = _context_block(inp)

    language_instruction = (
        "Respond in Swahili. Use natural Kenyan Swahili — not textbook formal Swahili."
        if inp.detected_language.value in ("sw", "sheng")
        else "Respond in English."
    )

    system_prompt = "\n\n".join(filter(bool, [
        BASE_PROMPT,
        tone_block,
        cluster_block,
        intent_block,
        language_instruction,
    ]))

    user_content = (
        f"{context_block}\n\n" if context_block else ""
    ) + f"STUDENT MESSAGE:\n{inp.text}"

    if inp.clarifying_question_from_cae:
        user_content += f"\n\nCLARIFYING QUESTION TO USE (if cluster requires one):\n{inp.clarifying_question_from_cae}"

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            temperature=0.55,
            max_tokens=160,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
        )

        response_text = response.choices[0].message.content.strip()
        primary_technique = (
            inp.technique_cluster.techniques[0].name
            if inp.technique_cluster.techniques
            else "unknown"
        )
        contains_q = "?" in response_text

        return TraceOutput(
            response_text=response_text,
            strategy_used=primary_technique,
            language=inp.detected_language.value,
            contains_clarifying_q=contains_q,
            trace_confidence=TraceConfidence.high,
            error=None,
        )

    except Exception as e:
        # Culturally appropriate fallback
        fallback = (
            "Niko hapa, endelea." if inp.detected_language.value in ("sw", "sheng")
            else "I'm here with you."
        )
        return TraceOutput(
            response_text=fallback,
            strategy_used="fallback",
            language=inp.detected_language.value,
            contains_clarifying_q=False,
            trace_confidence=TraceConfidence.fallback,
            error=str(e),
        )