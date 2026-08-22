#!/usr/bin/env python3
"""Fail if the published contract leaks vector-provider micro operations."""

from __future__ import annotations

import re
import sys
from pathlib import Path


REQUIRED_PATHS = {
    "/v1/rag/documents",
    "/v1/rag/documents/{document_id}",
    "/v1/rag/retrievals",
    "/v1/vector-collections",
    "/v1/vector-collections/{collection_id}",
    "/v1/vector-collections/{collection_id}/generations",
    "/v1/vector-collections/{collection_id}/activate",
    "/health/live",
    "/health/ready",
}
FORBIDDEN_PATH_PARTS = ("embed", "point", "qdrant", "vector-search")


def main() -> int:
    contract_path = Path(sys.argv[1]).resolve()
    source = contract_path.read_text(encoding="utf-8")
    paths = set(re.findall(r"^  (/[^:]+):\s*$", source, flags=re.MULTILINE))
    missing = REQUIRED_PATHS - paths
    unexpected = paths - REQUIRED_PATHS
    forbidden = sorted(
        path
        for path in paths
        if any(part in path.lower() for part in FORBIDDEN_PATH_PARTS)
        and not path.startswith("/v1/vector-collections")
    )
    required_fragments = (
        "Idempotency-Key",
        "traceparent",
        "embedding_schema_mismatch",
        "minimum_score",
        "metadata_fields",
        "GenerationState",
        "provisionVectorCollectionGeneration",
    )
    absent_fragments = [fragment for fragment in required_fragments if fragment not in source]
    forbidden_fragments = ("validation_queries", "migrateVectorCollection")
    present_forbidden_fragments = [
        fragment for fragment in forbidden_fragments if fragment in source
    ]
    transport_path = (
        contract_path.parents[2]
        / "services/core-rag-api/src/rag_core/transport/http.py"
    )
    transport_source = transport_path.read_text(encoding="utf-8")
    runtime_paths = set(
        re.findall(
            r'@app\.(?:get|post|delete|put|patch)\("([^"]+)"',
            transport_source,
        )
    )
    runtime_mismatch = runtime_paths != paths
    if (
        missing
        or unexpected
        or forbidden
        or absent_fragments
        or present_forbidden_fragments
        or runtime_mismatch
    ):
        print(f"missing paths: {sorted(missing)}", file=sys.stderr)
        print(f"unexpected paths: {sorted(unexpected)}", file=sys.stderr)
        print(f"forbidden paths: {forbidden}", file=sys.stderr)
        print(f"missing contract fragments: {absent_fragments}", file=sys.stderr)
        print(
            f"forbidden contract fragments: {present_forbidden_fragments}",
            file=sys.stderr,
        )
        if runtime_mismatch:
            print(
                f"runtime-only paths: {sorted(runtime_paths - paths)}",
                file=sys.stderr,
            )
            print(
                f"contract-only paths: {sorted(paths - runtime_paths)}",
                file=sys.stderr,
            )
        return 1
    print(f"OpenAPI boundary verified: {len(paths)} product and health paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
