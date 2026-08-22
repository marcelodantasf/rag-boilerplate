"""Deterministic text normalization, chunking, and identifier policy."""

import hashlib
import re
import unicodedata
from uuid import NAMESPACE_URL, uuid5

from rag_core.domain.models import Chunk


_SPACE = re.compile(r"[ \t\f\v]+")
_NEWLINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    value = "\n".join(_SPACE.sub(" ", line).strip() for line in value.split("\n"))
    return _NEWLINES.sub("\n\n", value).strip()


def content_version(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_document_id(provided: str | None, normalized_text: str) -> str:
    if provided is not None:
        return provided.strip()
    return f"doc_{content_version(normalized_text)[:24]}"


def stable_chunk_id(
    collection_id: str, generation_id: str, document_id: str, version: str, chunk_index: int
) -> str:
    identity = f"{collection_id}\x1f{generation_id}\x1f{document_id}\x1f{version}\x1f{chunk_index}"
    return f"chk_{uuid5(NAMESPACE_URL, identity)}"


def stable_point_id(chunk_id: str) -> str:
    return chunk_id.removeprefix("chk_")


def chunk_text(
    text: str,
    *,
    collection_id: str,
    generation_id: str,
    document_id: str,
    version: str,
    chunk_size: int,
    overlap: int,
) -> tuple[Chunk, ...]:
    """Split by a deterministic character budget, preferring word boundaries."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap smaller than chunk_size")
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(text.rfind(" ", start + 1, end + 1), text.rfind("\n", start + 1, end + 1))
            if boundary > start:
                end = boundary
        value = text[start:end].strip()
        if value:
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=stable_chunk_id(collection_id, generation_id, document_id, version, index),
                    index=index,
                    text=value,
                )
            )
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)
        while next_start < end and text[next_start].isspace():
            next_start += 1
        start = next_start
    return tuple(chunks)
