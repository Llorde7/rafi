from pydantic import BaseModel
from typing import Optional


class RAGQuery(BaseModel):
    query_text: str
    cause_type: str
    top_emotion: str
    collection: str = "university_kb"
    top_k: int = 4


class RAGChunk(BaseModel):
    chunk_id: str
    text: str
    source_document: str
    section: Optional[str] = None
    score: float
    category: Optional[str] = None


class RAGResult(BaseModel):
    query_text: str
    chunks: list[RAGChunk]
    summary: Optional[str] = None
    sources: list[str] = []
    retrieval_successful: bool = True
    error: Optional[str] = None