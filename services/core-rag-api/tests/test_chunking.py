from rag_core.domain.chunking import (
    chunk_text,
    content_version,
    normalize_text,
    stable_chunk_id,
    stable_point_id,
)


def test_normalization_and_ids_are_deterministic() -> None:
    text = normalize_text("  Caf\u00e9  \r\n\r\n\r\n  policy\ttext ")
    assert text == "Caf\u00e9\n\npolicy text"
    version = content_version(text)
    assert version == content_version(text)
    assert stable_chunk_id("handbook", "gen_1", "doc-1", version, 0) == stable_chunk_id(
        "handbook", "gen_1", "doc-1", version, 0
    )
    assert stable_point_id("chunk-1") == stable_point_id("chunk-1")


def test_chunking_respects_budget_and_overlap() -> None:
    text = "one two three four five six seven eight nine ten eleven twelve"
    version = content_version(text)
    chunks = chunk_text(
        text,
        collection_id="c",
        generation_id="gen_1",
        document_id="d",
        version=version,
        chunk_size=20,
        overlap=4,
    )
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 20 for chunk in chunks)
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_chunking_rejects_invalid_overlap() -> None:
    try:
        chunk_text("text", collection_id="c", generation_id="gen_1", document_id="d", version="v", chunk_size=4, overlap=4)
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("invalid overlap was accepted")
