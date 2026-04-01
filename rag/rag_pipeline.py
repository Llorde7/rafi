"""
rag/rag_pipeline.py
────────────────────
Orchestrates retrieve → summarise → RAGResult.
Only file the planner imports from rag/.
"""

from contracts.rag_contract import RAGQuery, RAGResult
from contracts.planner_contract import PlannerInput
from rag.retriever import retrieve
from rag.summariser import summarise


def _build_query(inp: PlannerInput) -> RAGQuery:
    return RAGQuery(
        query_text=inp.text,
        cause_type=inp.cause_type,
        top_emotion=inp.top_emotion,
    )


def _build_student_context(inp: PlannerInput) -> str:
    return (
        f"A student experiencing {inp.top_emotion} "
        f"related to {inp.cause_type.replace('_', ' ')}. "
        f"They said: \"{inp.text[:200]}\""
    )


async def run_rag(inp: PlannerInput) -> RAGResult:
    query = _build_query(inp)

    try:
        chunks = await retrieve(query)
    except Exception as e:
        return RAGResult(
            query_text=query.query_text,
            chunks=[],
            retrieval_successful=False,
            error=f"Retrieval failed: {e}",
        )

    if not chunks:
        return RAGResult(
            query_text=query.query_text,
            chunks=[],
            retrieval_successful=True,
        )

    try:
        summary = await summarise(chunks, _build_student_context(inp))
    except Exception as e:
        return RAGResult(
            query_text=query.query_text,
            chunks=chunks,
            retrieval_successful=True,
            sources=list({c.source_document for c in chunks}),
            error=f"Summarisation failed: {e}",
        )

    return RAGResult(
        query_text=query.query_text,
        chunks=chunks,
        summary=summary,
        sources=list({c.source_document for c in chunks}),
        retrieval_successful=True,
    )