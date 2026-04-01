"""
rag/summariser.py
──────────────────
Lightweight Groq call: chunks → student-appropriate summary.
Strict grounding — returns None if chunks aren't relevant.
"""

import os
from groq import AsyncGroq
from dotenv import load_dotenv

from contracts.rag_contract import RAGChunk

load_dotenv()
client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a document summariser for a student mental health support system.

You receive excerpts from university policy and support documents.
Produce a SHORT, accurate summary for use in a supportive response to a student.

RULES:
1. Only use information in the provided excerpts. Never infer beyond them.
2. If the excerpts are not relevant to the student's situation, respond with exactly: NO_RELEVANT_CONTENT
3. Maximum 3-5 sentences.
4. Plain, warm, accessible language — not bureaucratic or legalistic.
5. Write in third person — this feeds a response generator, not the student directly.
6. Paraphrase and synthesise. Never reproduce chunks verbatim.
7. Return ONLY the summary or NO_RELEVANT_CONTENT. No preamble, no markdown."""


def _format_chunks(chunks: list[RAGChunk]) -> str:
    lines = ["Retrieved excerpts:"]
    for i, chunk in enumerate(chunks, 1):
        section = f" [{chunk.section}]" if chunk.section else ""
        lines.append(
            f"\n[Excerpt {i} — {chunk.source_document}{section} "
            f"(score: {chunk.score:.2f})]"
        )
        lines.append(chunk.text)
    return "\n".join(lines)


async def summarise(chunks: list[RAGChunk], student_context: str) -> str | None:
    if not chunks:
        return None

    user_content = (
        f"Student situation: {student_context}\n\n"
        f"{_format_chunks(chunks)}\n\n"
        f"Produce a summary relevant to this student's situation."
    )

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            temperature=0.05,
            max_tokens=200,
        )
        result = response.choices[0].message.content.strip()
        return None if (result == "NO_RELEVANT_CONTENT" or not result) else result
    except Exception:
        return None
