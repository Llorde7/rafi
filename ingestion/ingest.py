"""
ingestion/ingest.py
────────────────────
CLI pipeline: chunk → embed → upsert to Qdrant Cloud.

Usage:
    python -m ingestion.ingest --file path/to/doc.pdf --category financial
    python -m ingestion.ingest --dir path/to/docs/ --category academic
    python -m ingestion.ingest --file doc.txt --category health --recreate

Supported formats: .txt, .md, .pdf (text-extractable)

Required .env vars:
    QDRANT_URL       https://your-cluster.qdrant.io
    QDRANT_API_KEY   your Qdrant Cloud API key
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, PayloadSchemaType,
)
from fastembed import TextEmbedding

from ingestion.chunker import chunk_document, DocumentChunk

load_dotenv()

COLLECTION_NAME = "university_kb"
EMBED_MODEL     = "BAAI/bge-small-en-v1.5"
VECTOR_DIM      = 384
BATCH_SIZE      = 64

VALID_CATEGORIES = {
    "financial", "health", "academic", "welfare",
    "housing", "legal", "disability", "general"
}


def _get_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    key = os.getenv("QDRANT_API_KEY")
    if not url or not key:
        raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set in .env")
    return QdrantClient(url=url, api_key=key)


def _ensure_collection(client: QdrantClient, recreate: bool = False):
    exists = any(c.name == COLLECTION_NAME for c in client.get_collections().collections)
    if exists and recreate:
        print(f"Recreating collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)
        exists = False
    if not exists:
        print(f"Creating collection '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        for field in ("category", "source_document"):
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        print("Collection created with indexes on category + source_document.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")


def _extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            pages = [p.extract_text() for p in reader.pages if p.extract_text()]
            if not pages:
                raise ValueError("No extractable text in PDF.")
            return "\n\n".join(pages)
        except ImportError:
            raise RuntimeError("pypdf required for PDF: pip install pypdf")
    raise ValueError(f"Unsupported file type: {suffix}")


def ingest_file(
    path: Path, category: str, client: QdrantClient, model: TextEmbedding
) -> int:
    print(f"  Extracting text...")
    text = _extract_text(path)

    print(f"  Chunking...")
    chunks = chunk_document(text=text, source_document=path.name, category=category)
    print(f"  {len(chunks)} chunks.")

    if not chunks:
        print(f"  WARNING: No chunks. Skipping.")
        return 0

    total = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        embeddings = list(model.embed([c.text for c in batch]))
        points = [
            PointStruct(
                id=chunk.chunk_id,
                vector=emb.tolist(),
                payload={
                    "text":            chunk.text,
                    "source_document": chunk.source_document,
                    "section":         chunk.section,
                    "category":        chunk.category,
                    "char_start":      chunk.char_start,
                    "char_end":        chunk.char_end,
                }
            )
            for chunk, emb in zip(batch, embeddings)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        total += len(batch)
        print(f"  Upserted {total}/{len(chunks)}...")

    return len(chunks)


def main():
    parser = argparse.ArgumentParser(description="Ingest university documents into Qdrant")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=Path)
    group.add_argument("--dir",  type=Path)
    parser.add_argument("--category", default="general", choices=sorted(VALID_CATEGORIES))
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    if args.file and not args.file.exists():
        print(f"ERROR: File not found: {args.file}"); sys.exit(1)
    if args.dir and not args.dir.is_dir():
        print(f"ERROR: Directory not found: {args.dir}"); sys.exit(1)

    client = _get_client()
    _ensure_collection(client, recreate=args.recreate)

    print(f"Loading embedding model '{EMBED_MODEL}'...")
    model = TextEmbedding(model_name=EMBED_MODEL)

    files = [args.file] if args.file else [
        f for f in args.dir.iterdir()
        if f.suffix.lower() in (".txt", ".md", ".pdf")
    ]
    if not files:
        print("No supported files found."); sys.exit(1)

    print(f"\nIngesting {len(files)} file(s) as category='{args.category}'...")
    total = 0
    for f in files:
        print(f"\n[{f.name}]")
        total += ingest_file(f, args.category, client, model)

    print(f"\nDone. {total} chunks upserted to '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()