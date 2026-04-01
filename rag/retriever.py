"""
rag/retriever.py
─────────────────
Embeds query with FastEmbed and searches Qdrant async.
Module-level singletons — loaded once, reused across requests.
"""

import os
from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import AsyncQdrantClient

from contracts.rag_contract import RAGQuery, RAGChunk

load_dotenv()

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
MIN_SCORE   = 0.35

_embed_model: TextEmbedding | None = None
_qdrant_client: AsyncQdrantClient | None = None


def _get_embed_model() -> TextEmbedding:
    global _embed_model
    if _embed_model is None:
        _embed_model = TextEmbedding(model_name=EMBED_MODEL)
    return _embed_model


def _get_qdrant_client() -> AsyncQdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        url = os.getenv("QDRANT_URL")
        key = os.getenv("QDRANT_API_KEY")
        if not url or not key:
            raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set in .env")
        _qdrant_client = AsyncQdrantClient(url=url, api_key=key)
    return _qdrant_client


def _enrich_query(query: RAGQuery) -> str:
    """
    Append cause_type and top_emotion to improve retrieval precision.
    e.g. "I can't pay my fees financial stress identity_threat sadness"
    """
    return (
        f"{query.query_text} "
        f"university student support "
        f"{query.cause_type.replace('_', ' ')} "
        f"{query.top_emotion}"
    )


async def retrieve(query: RAGQuery) -> list[RAGChunk]:
    model  = _get_embed_model()
    client = _get_qdrant_client()

    vectors = list(model.embed([_enrich_query(query)]))
    if not vectors:
        return []

    results = await client.search(
        collection_name=query.collection,
        query_vector=vectors[0].tolist(),
        limit=query.top_k,
        with_payload=True,
        score_threshold=MIN_SCORE,
    )

    return [
        RAGChunk(
            chunk_id=str(hit.id),
            text=(hit.payload or {}).get("text", ""),
            source_document=(hit.payload or {}).get("source_document", "unknown"),
            section=(hit.payload or {}).get("section"),
            score=round(hit.score, 4),
            category=(hit.payload or {}).get("category"),
        )
        for hit in results
    ]