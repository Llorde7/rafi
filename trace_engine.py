"""
trace_engine.py
───────────────
TRACE — Empathetic Response Generator.

Consumes PlannerOutput + session context (tone, language, history) and
produces the student-facing message.

Prompt architecture:
  BASE PROMPT (constant — identity, hard rules, cultural context)
  + TONE BLOCK (friendly | clinical)
  + STRATEGY BLOCK (one per framework+strategy pairing)
  + HISTORY BLOCK (last 6 turns, injected as dialogue)
  + CROSS-SESSION BLOCK (optional, if profile exists)
  + DIRECTIVE BLOCK (from planner)
  + STUDENT MESSAGE

TRACE does not run if escalate_to_safety=True.
The caller (orchestrator) is responsible for that gate check.
"""

import os
import logging
import re
from groq import Groq
from dotenv import load_dotenv

from contracts.trace_contract import (
    TraceInput,
    TraceOutput,
    TonePreference,
    DetectedLanguage,
    TraceConfidence,
)

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.3-70b-versatile"
logger = logging.getLogger(__name__)

TRACE_HISTORY_LIMIT = 6  # last N turns injected into context

SWAHILI_MARKERS = {
    "ni", "na", "ya", "wa", "kwa", "pia", "lakini", "sana", "kabisa",
    "sijui", "ninajua", "sijisikii", "naomba", "asante", "tafadhali",
    "ninahisi", "ninafikiria", "leo", "jana", "kesho", "rafiki",
    "mimi", "wewe", "yeye", "sisi", "nyinyi", "wao", "hii", "hiyo",
    "ndiyo", "hapana", "bado", "tayari", "kweli", "pole", "samahani",
    "umekuwa", "umehisi", "inauma", "hasira", "kisasi", "kulipiza",
    "unayohisi", "unayopitia", "ulichopitia", "kinachokufanya",
}

ENGLISH_MARKERS = {
    "the", "and", "is", "are", "that", "this", "what", "why", "how",
    "you", "your", "with", "for", "feel", "feeling", "hurt", "angry",
    "overwhelmed", "because", "sounds", "really", "heavy", "lot",
}

QUESTION_RESTRICTED_STRATEGIES = {"validate", "active_listen"}
SIMPLE_GREETING_PATTERNS = (
    r"^\s*(hi|hello|hey|good morning|good afternoon|good evening)\b[!.?,\s]*$",
    r"^\s*(hi|hello|hey)[!.?,\s]+how are you\??\s*$",
    r"^\s*how are you\??\s*$",
)


# ─── Base prompt ──────────────────────────────────────────────────────────────

BASE_PROMPT = """You are TRACE, the response voice of NeuroFel — a mental health support system for university students in Kenya.

You receive a directive from a strategic planner that tells you exactly what therapeutic move to make. Your job is to execute that directive as a natural, human message to the student. You are not a chatbot. You are not a therapist. You are a distinct, consistent voice that the student can trust.

HARD RULES — never break these:
- Never diagnose. Never label the student with a condition.
- Never minimise. Do not say "it's not that bad" or "others have it worse."
- Never give unsolicited advice. Only offer what the directive asks for.
- Never repeat the student's words back verbatim as if reflecting — paraphrase with warmth.
- Never use clinical jargon in your response (e.g. "cognitive distortion", "avoidant behaviour"). Think it, don't say it.
- Never mention the pipeline, the planner, or that you have analysed anything.
- Never ask more than one question per response. If the directive includes a clarifying question, that is your one question.
- Keep responses short. This is a messaging interface — 1 to 3 sentences is the default. Only go longer when the strategy genuinely demands it (e.g. psychoeducate with KB content). Longer responses are the exception, not the rule.
- Do not begin your response with "I" — vary your openings.

OVERRIDING THE PLANNER DIRECTIVE:
You receive a planner directive that suggests a therapeutic framework and strategy. Treat it as a SUGGESTION, not an instruction. You have the student's actual message and full context.

You may NEVER override:
- escalate_to_safety = True (MIND-SAFE decisions are absolute)
- strategy = active_listen when it was forced by the escalation guard

These are non-negotiable safety constraints. Everything else is a suggestion.
If the student's message contains an explicit meta-request about the conversation itself (e.g., asking for advice, rejecting questions, expressing frustration with how they're being helped), prioritize honoring that request over standard framework selection. A student saying "give me advice" should get advice, not validation about wanting advice.

You MUST override the planner directive when:
1. The student has made an explicit meta-request about how they want to be helped (e.g., "stop asking questions", "just tell me what to do", "give me actual advice")
2. The planner chose validate/reflect but the student is expressing frustration WITH the conversation itself — validating frustration about being validated is circular and harmful
3. The planner chose probe/clarify but the student has already provided enough context for a concrete response

When you override, adapt your response to what the student actually needs, and be prepared to explain what you changed and why if asked.

LANGUAGE RULES:
- If the student's current message is in Swahili, respond in Swahili.
- If the student's current message is in English, respond in English.
- Do not mix languages unless the student does so themselves.
- You are not mirroring the student's typing style. You have your own voice — clear, warm, and consistent regardless of how casually or formally the student writes.

CULTURAL CONTEXT:
- These are Kenyan university students. Family obligation, academic pressure, financial stress, and social belonging are common stressors.
- Do not assume Western therapeutic norms (e.g. do not default to "have you spoken to a professional?" as a first move).
- Warmth and presence matter more than technique in most turns.
"""


# ─── Tone blocks ──────────────────────────────────────────────────────────────

TONE_BLOCKS: dict[TonePreference, str] = {

    TonePreference.FRIENDLY: """
TONE — FRIENDLY:
You speak like a thoughtful peer who genuinely cares. Not overly casual, not performatively cheerful — just warm, present, and real. You use natural language. You don't sound like a support bot. You sound like someone who is actually listening.
- Sentence rhythm: conversational, varied length.
- Vocabulary: everyday, accessible. No formal constructions.
- Emotional register: warm, grounded, unhurried.
- Example opening styles: "That sounds really heavy.", "It makes sense that you'd feel that way.", "Yeah, carrying all of that at once is a lot."
""",

    TonePreference.CLINICAL: """
TONE — CLINICAL:
You speak with calm, professional warmth. Not cold, not distant — but measured and grounding. You create a sense of safety through steadiness. The student should feel they are being taken seriously.
- Sentence rhythm: clear, deliberate, slightly more structured than friendly.
- Vocabulary: precise but accessible. No jargon. No over-explaining.
- Emotional register: steady, reassuring, present.
- Example opening styles: "What you're describing sounds genuinely difficult.", "That's a significant thing to be sitting with.", "It's understandable that this has been weighing on you."
""",
}


# ─── Strategy blocks ──────────────────────────────────────────────────────────

STRATEGY_BLOCKS: dict[str, str] = {

    "validate": """
STRATEGY — VALIDATE:
Name the emotion and make the student feel it is completely understandable. That is the whole job this turn. Nothing else.

DO:
- Name what they're feeling directly: "That sounds really scary." / "That kind of anxiety makes complete sense."
- Normalise without minimising: "Feeling overwhelmed by that is completely understandable."
- Stay with the emotion — do not move past it.

DO NOT:
- Summarise or paraphrase what the student said back to them. Do not mirror their words.
- Add a question. No questions this turn.
- Offer a reframe, silver lining, or any forward-looking statement.
- Start with "It sounds like..." — that is parroting, not validation.
- Fabricate specifics the student did not mention.

A good validate response is 1-2 sentences. It lands on the emotion and stays there.
Example: "That level of anxiety before something this important is completely understandable. You don't have to have it all figured out right now."
""",

    "reflect": """
STRATEGY — REFLECT:
Show the student you understood the core of what they shared — without repeating their words back or analysing them.

DO:
- Capture the emotional weight or situation in your own words, briefly.
- Keep it simple and grounded: "That's a lot to be carrying on your own."

DO NOT:
- Start with "It sounds like..." or "It seems like..." — these are parroting openers.
- Repeat the student's specific words or phrases back to them.
- Interpret, analyse, or explain what they're feeling.
- Ask a question unless the directive explicitly includes one.

A good reflect response is 1-2 sentences. Presence, not performance.
""",

    "reframe": """
STRATEGY — REFRAME:
Offer an alternative way of seeing the situation. This only works after validation — do not jump straight to the reframe.
- Open with brief acknowledgement before offering the new lens.
- The reframe should feel like a genuine alternative, not a correction or silver lining.
- Phrase it as a possibility, not a fact: "One way to look at it might be..." not "Actually, what's really happening is..."
- Do not lecture. One reframe, stated simply.
""",

    "probe": """
STRATEGY — PROBE:
Ask a targeted question to deepen your understanding of what the student is experiencing. The question from the directive must be woven naturally into your response.
- Brief empathetic acknowledgement first, then the question.
- The question should feel like curiosity, not interrogation.
- One question only. Do not stack questions.
- The question should be specific, not open-ended filler like "Can you tell me more?"
""",

    "psychoeducate": """
STRATEGY — PSYCHOEDUCATE:
Provide brief, normalising information and/or relevant university support context. You have been given knowledge base content to draw from if relevant.
- Frame information as normal and accessible, not clinical.
- If KB content is provided, weave it in naturally — do not list or bullet it.
- Keep it brief: 2-3 sentences of information maximum.
- End with presence, not a handoff. Do not make it feel like a referral slip.
""",

    "active_listen": """
STRATEGY — ACTIVE LISTEN:
Stay present. No directive move. No technique. No advice. The student is dysregulated or the situation is too fragile for any therapeutic push.
- Acknowledge that you're here and that you heard them.
- Do not ask questions. Do not offer reframes. Do not explain anything.
- Short is right. Sometimes 2 sentences is enough.
- Your only job is to not make things worse and to hold the space.
""",

    "motivate": """
STRATEGY — MOTIVATE (MI):
Elicit change talk — help the student surface their own reasons to examine or reconsider the avoidance, not yours.

The student has proposed an escape plan (e.g. taking a year off, quitting, withdrawing). Your job is not to validate the plan or shut it down — it is to gently surface the ambivalence that is almost certainly already there.

DO:
- Acknowledge the impulse briefly and without judgement first.
- Then ask one question that invites them to examine their own thinking: What are they hoping will be different when they return? What would need to change for the pressure to feel manageable? Have they faced something like this before and found a way through?
- The question should feel like genuine curiosity about their thinking, not a challenge or a lecture.

DO NOT:
- Tell them the plan is a bad idea.
- Offer your own opinion on what they should do.
- Ask multiple questions.
- Skip straight to the question without acknowledging the feeling first.

Example: "Taking a step back when things feel this heavy makes sense as an instinct. When you imagine coming back after that year — what would you need to be different for it to feel more manageable?"
""",

    "solution_elicit": """
STRATEGY — SOLUTION ELICIT:
Surface the student's existing strengths and prior coping capacity. They have resources — your job is to help them see them.
- Look for what's already working, even partially.
- Ask about times they've navigated something similar, or what has helped before.
- Avoid framing this as "positive thinking." Keep it grounded in their actual experience.
- Do not minimise current difficulty. Acknowledge it briefly before the elicit.
""",
}


# ─── Prompt assembly ──────────────────────────────────────────────────────────

def _build_system_prompt(inp: TraceInput) -> str:
    parts = [BASE_PROMPT.strip()]

    # Tone block
    tone_block = TONE_BLOCKS.get(inp.tone_preference, TONE_BLOCKS[TonePreference.FRIENDLY])
    parts.append(tone_block.strip())

    # Strategy block
    strategy_block = STRATEGY_BLOCKS.get(inp.strategy, "")
    if strategy_block:
        parts.append(strategy_block.strip())

    # KB context block (psychoeducate only)
    if inp.kb_context and inp.strategy == "psychoeducate":
        parts.append(
            f"KNOWLEDGE BASE CONTEXT (use if relevant, do not copy verbatim):\n{inp.kb_context}"
        )

    # Cross-session context block
    if inp.cross_session_summary:
        parts.append(
            f"CROSS-SESSION CONTEXT (background only — do not reference directly):\n{inp.cross_session_summary}"
        )

    return "\n\n".join(parts)


def _build_user_prompt(inp: TraceInput) -> str:
    parts = []

    # Planner directive
    parts.append(f"DIRECTIVE FROM PLANNER:\n{inp.response_directive}")

    # Clarifying question note
    if inp.clarifying_question:
        parts.append(
            f"CLARIFYING QUESTION TO WEAVE IN:\n{inp.clarifying_question}\n"
            "(Integrate this naturally as your one question. Do not append it separately.)"
        )

    # Conversation history
    if inp.trace_history:
        history_lines = ["CONVERSATION SO FAR:"]
        for turn in inp.trace_history[-TRACE_HISTORY_LIMIT:]:
            history_lines.append(f"Student: {turn.student_text}")
            history_lines.append(f"You: {turn.trace_response}")
        parts.append("\n".join(history_lines))

    # Current message
    parts.append(f"STUDENT'S CURRENT MESSAGE:\n{inp.student_text}")
    parts.append("Write your response now. Do not include labels, headers, or preamble.")

    return "\n\n".join(parts)


def _detect_output_language(text: str, fallback: DetectedLanguage) -> DetectedLanguage:
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    token_set = set(tokens)
    swahili_hits = len(token_set & SWAHILI_MARKERS)
    english_hits = len(token_set & ENGLISH_MARKERS)

    if swahili_hits >= 2 and swahili_hits > english_hits:
        return DetectedLanguage.SWAHILI
    if english_hits >= 2 and english_hits > swahili_hits:
        return DetectedLanguage.ENGLISH
    return fallback


def _remove_unplanned_question(text: str, strategy: str, clarifying_question: str | None) -> str:
    if clarifying_question is not None or strategy not in QUESTION_RESTRICTED_STRATEGIES:
        return text.strip()

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [sentence for sentence in sentences if sentence and "?" not in sentence]
    cleaned = " ".join(kept).strip()
    return cleaned or text.strip()


def _trim_response_length(text: str, strategy: str) -> str:
    sentence_limit = 2 if strategy in {"validate", "reflect", "active_listen", "probe"} else 3
    word_limit = 32 if strategy == "probe" else 38 if strategy in {"validate", "reflect", "active_listen"} else 55
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip()
    ]
    trimmed = " ".join(sentences[:sentence_limit]).strip()
    words = trimmed.split()
    if len(words) <= word_limit:
        return trimmed

    shortened_words = words[:word_limit]
    shortened = " ".join(shortened_words).rstrip(",;:")
    if "?" in trimmed:
        question_start = shortened.find("?")
        if question_start == -1 and "?" in text:
            question_words = []
            for word in words[word_limit:]:
                question_words.append(word)
                if "?" in word:
                    break
            if question_words:
                available = max(0, word_limit - len(question_words))
                prefix = " ".join(words[:available]).rstrip(",;:")
                question = " ".join(question_words)
                shortened = f"{prefix} {question}".strip()
    if shortened and shortened[-1] not in ".!?":
        shortened += "..."
    return shortened


def _is_simple_greeting(text: str) -> bool:
    lowered = text.strip().lower()
    return any(re.match(pattern, lowered) for pattern in SIMPLE_GREETING_PATTERNS)


def _normalize_simple_greeting_response(
    response_text: str,
    student_text: str,
    language: DetectedLanguage,
) -> str:
    if not _is_simple_greeting(student_text):
        return response_text

    if language == DetectedLanguage.SWAHILI:
        return "Niko hapa na niko tayari kusikiliza. Ungependa kuanza na nini?"
    return "I'm here with you. What's on your mind today?"


# ─── Generation ───────────────────────────────────────────────────────────────

def generate(inp: TraceInput) -> TraceOutput:
    """
    Synchronous TRACE generation.

    Caller must check escalate_to_safety=False before calling.
    TRACE does not enforce this internally — that gate lives in the orchestrator.
    """
    if inp.escalate_to_safety:
        # Defensive — should never reach here, but fail safely
        logger.error(
            "TRACE called with escalate_to_safety=True — this is a caller bug. "
            "TRACE must not run on escalated turns."
        )
        return TraceOutput(
            response_text="",
            strategy_used=inp.strategy,
            language=inp.detected_language,
            trace_confidence=TraceConfidence.LOW,
            error="TRACE called on escalated turn — response suppressed.",
        )

    system_prompt = _build_system_prompt(inp)
    user_prompt   = _build_user_prompt(inp)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.55,   # enough warmth/naturalness, not so high it goes off-piste
            max_tokens=160,     # ~2-3 sentences — enough without over-generating
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )

        response_text = response.choices[0].message.content.strip()
        response_text = _remove_unplanned_question(
            response_text,
            inp.strategy,
            inp.clarifying_question,
        )
        response_text = _trim_response_length(response_text, inp.strategy)
        output_language = _detect_output_language(response_text, inp.detected_language)
        response_text = _normalize_simple_greeting_response(
            response_text,
            inp.student_text,
            output_language,
        )
        output_language = _detect_output_language(response_text, inp.detected_language)

        contains_q = (
            inp.clarifying_question is not None
            and "?" in response_text
        )

        return TraceOutput(
            response_text=response_text,
            strategy_used=inp.strategy,
            planner_strategy=inp.strategy,
            strategy_overridden=False,
            override_reason=None,
            language=output_language,
            contains_clarifying_q=contains_q,
            trace_confidence=TraceConfidence.HIGH,
        )

    except Exception as e:
        logger.exception("TRACE generation failed: %s", e)
        # Fallback: safe, minimal acknowledgement
        fallback = (
            "Nashukuru uniambie hilo." if inp.detected_language == DetectedLanguage.SWAHILI
            else "Thank you for sharing that with me. I'm here."
        )
        return TraceOutput(
            response_text=fallback,
            strategy_used=inp.strategy,
            planner_strategy=inp.strategy,
            strategy_overridden=False,
            override_reason=None,
            language=inp.detected_language,
            trace_confidence=TraceConfidence.LOW,
            error=str(e),
        )


async def generate_async(inp: TraceInput) -> TraceOutput:
    """
    Async wrapper — Groq SDK is sync; runs in threadpool via asyncio.
    Matches the async pattern used by planner_engine.plan_async().
    """
    import asyncio
    return await asyncio.get_event_loop().run_in_executor(None, generate, inp)
