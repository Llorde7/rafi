"""
ingestion/chunker.py
─────────────────────
Splits raw document text into overlapping chunks with metadata.
Chunks by section heading → paragraph → sliding window fallback.
No external dependencies beyond stdlib.
"""

import re
import uuid
from dataclasses import dataclass

CHUNK_SIZE    = 400   # target tokens (chars / 4)
CHUNK_OVERLAP = 40
CHARS_PER_TOK = 4


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source_document: str
    section: str
    category: str
    char_start: int
    char_end: int


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]


def _split_by_window(text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_chars].strip())
        start += chunk_chars - overlap_chars
    return [c for c in chunks if c]


def _extract_sections(text: str) -> list[tuple[str, str]]:
    lines = text.split('\n')
    sections: list[tuple[str, str]] = []
    current_heading = "General"
    current_lines: list[str] = []
    heading_pattern = re.compile(r'^([A-Z][A-Z\s\d\-&/]{3,}|.{3,}:)\s*$')

    for line in lines:
        stripped = line.strip()
        if stripped and heading_pattern.match(stripped) and len(stripped) < 80:
            if current_lines:
                sections.append((current_heading, '\n'.join(current_lines).strip()))
            current_heading = stripped.rstrip(':')
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, '\n'.join(current_lines).strip()))

    return [(h, t) for h, t in sections if t.strip()]


def chunk_document(
    text: str,
    source_document: str,
    category: str = "general",
) -> list[DocumentChunk]:
    chunk_chars   = CHUNK_SIZE * CHARS_PER_TOK
    overlap_chars = CHUNK_OVERLAP * CHARS_PER_TOK
    sections      = _extract_sections(text)
    chunks: list[DocumentChunk] = []
    char_offset = 0

    for heading, section_text in sections:
        for para in _split_paragraphs(section_text):
            if len(para) <= chunk_chars:
                chunks.append(DocumentChunk(
                    chunk_id=str(uuid.uuid4()), text=para,
                    source_document=source_document, section=heading,
                    category=category,
                    char_start=char_offset, char_end=char_offset + len(para),
                ))
                char_offset += len(para)
            else:
                for sub in _split_by_window(para, chunk_chars, overlap_chars):
                    chunks.append(DocumentChunk(
                        chunk_id=str(uuid.uuid4()), text=sub,
                        source_document=source_document, section=heading,
                        category=category,
                        char_start=char_offset, char_end=char_offset + len(sub),
                    ))
                    char_offset += len(sub) - overlap_chars

    return chunks