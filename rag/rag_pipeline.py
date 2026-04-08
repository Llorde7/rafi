"""
rag/rag_pipeline.py
────────────────────
Orchestrates retrieve → summarise → RAGResult.
Only file the planner imports from rag/.
"""

import logging
from time import perf_counter

from contracts.rag_contract import RAGQuery, RAGResult
from contracts.planner_contract import PlannerInput
from rag.retriever import retrieve
from rag.summariser import summarise


logger = logging.getLogger(__name__)


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
    rag_started = perf_counter()

    try:
        retrieve_started = perf_counter()
        chunks = await retrieve(query)
        logger.info(
            "RAG timing | stage=retrieve duration_ms=%.1f chunk_count=%d",
            (perf_counter() - retrieve_started) * 1000,
            len(chunks),
        )
    except Exception as e:
        return RAGResult(
            query_text=query.query_text,
            chunks=[],
            retrieval_successful=False,
            error=f"Retrieval failed: {e}",
        )

    if not chunks:
        logger.info(
            "RAG timing | stage=total duration_ms=%.1f retrieval_successful=true chunk_count=0",
            (perf_counter() - rag_started) * 1000,
        )
        return RAGResult(
            query_text=query.query_text,
            chunks=[],
            retrieval_successful=True,
        )

    try:
        summarise_started = perf_counter()
        summary = await summarise(chunks, _build_student_context(inp))
        logger.info(
            "RAG timing | stage=summarise duration_ms=%.1f summary_present=%s",
            (perf_counter() - summarise_started) * 1000,
            bool(summary),
        )
    except Exception as e:
        return RAGResult(
            query_text=query.query_text,
            chunks=chunks,
            retrieval_successful=True,
            sources=list({c.source_document for c in chunks}),
            error=f"Summarisation failed: {e}",
        )

    result = RAGResult(
        query_text=query.query_text,
        chunks=chunks,
        summary=summary,
        sources=list({c.source_document for c in chunks}),
        retrieval_successful=True,
    )
    logger.info(
        "RAG timing | stage=total duration_ms=%.1f retrieval_successful=%s chunk_count=%d summary_present=%s",
        (perf_counter() - rag_started) * 1000,
        result.retrieval_successful,
        len(result.chunks),
        bool(result.summary),
    )
    return result
