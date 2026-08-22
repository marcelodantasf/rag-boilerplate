# Embedding API — Implemented Design

## Supported behavior

The service exposes one general-purpose operation:

```text
POST /v1/embeddings
```

The `input` field accepts either one JSON string or an array of JSON strings.
A single string is normalized to a one-item list before the use case reaches the
engine. An array is passed to the engine in one `encode` call, so request-level
batching is real model batching rather than a loop of single inference calls.

The API makes no assumptions about document shape or subject. Law excerpts,
recipes, papers, prose, source code, Unicode, and other non-empty UTF-8 text all
follow the same path. There is deliberately no `input_type`, document schema,
chunk metadata, document ID, collection ID, or Qdrant integration in this
service.

Single input example:

```json
{
  "model": "all-MiniLM-L6-v2",
  "input": "Mix flour and water, then knead for ten minutes."
}
```

Batch example:

```json
{
  "model": "all-MiniLM-L6-v2",
  "input": [
    "Article 1. All human beings are born free and equal.",
    "Mix flour and water, then knead for ten minutes.",
    "A learning rate controls the size of an optimizer step."
  ]
}
```

Both return the same response shape:

```json
{
  "model": "all-MiniLM-L6-v2",
  "dimension": 384,
  "vectors": [
    {"index": 0, "embedding": [0.012, -0.043]}
  ],
  "usage": {"input_count": 1, "input_tokens": 11}
}
```

Every real embedding contains exactly `dimension` finite floats. `index`
corresponds to the original input position. The response is an embedding
contract, not a Qdrant point: storage-aware callers remain responsible for IDs,
payloads, and mapping `embedding` to the vector field expected by their store.

## Internal architecture

The implementation follows the boundaries in the service architecture guide:

```text
transport/http
  request shape, single-to-list normalization, response serialization,
  trace IDs, safe error responses
        |
application
  model routing, byte/count limits, engine result verification
        |
ports/EmbeddingEngine
  capabilities(), embed(), warmup()
        |
adapters
  local Sentence Transformers runtime or deterministic test double
        |
domain + infrastructure
  provider-neutral results/errors and typed environment settings
```

`SentenceTransformerEngine` owns model loading, tokenization, inference, and
conversion to plain finite floats. Its default model source is
`sentence-transformers/all-MiniLM-L6-v2`, pinned to immutable revision
`c9745ed1d9f207416be6d2e6f8de32d1f16199bf`. The public model ID, source,
revision, expected dimension, cache, and inference device are configurable, but
startup fails if identity/configuration is invalid or the loaded dimension does
not match the expected dimension.

The adapter tokenizes with truncation disabled and rejects an item that exceeds
the model's token capacity. It also rejects batches over the configured token
budget. Consequently no text is silently shortened before inference.

The application verifies that the adapter returns:

- the requested model identity;
- one vector for every input;
- the advertised dimension for every vector;
- only finite numeric values.

Provider exceptions are normalized without exposing stack traces, input text,
tokens, or vectors. Error responses include a stable `code`, safe `message`, and
`trace_id`, with non-sensitive limit details when useful.

## Configuration

The implementation reads these environment variables. Container defaults and
deployment wiring are intentionally left to the deployment phase.

| Variable | Default | Purpose |
| --- | --- | --- |
| `DEFAULT_MODEL_ID` | `all-MiniLM-L6-v2` | Public model identity |
| `DEFAULT_EMBEDDING_MODEL` | same | Backward-compatible scaffold alias |
| `MODEL_SOURCE` | `sentence-transformers/all-MiniLM-L6-v2` | Model repository |
| `MODEL_REVISION` | pinned commit | Immutable artifact revision |
| `EXPECTED_DIMENSION` | `384` | Startup contract check |
| `INFERENCE_DEVICE` | `cpu` | Runtime device |
| `MODEL_CACHE_DIR` | unset | Model cache location |
| `NORMALIZE_EMBEDDINGS` | `true` | L2-normalize output |
| `ENGINE_BATCH_SIZE` | `32` | Internal model batch size |
| `MAX_BATCH_ITEMS` | `64` | Maximum request item count |
| `MAX_INPUT_BYTES` | `65536` | Per-item UTF-8 byte limit |
| `MAX_TOTAL_INPUT_BYTES` | `262144` | Whole-request UTF-8 byte limit |
| `MAX_BATCH_TOKENS` | `8192` | Whole-request token limit |
| `LOG_LEVEL` | `INFO` | Structured request log threshold |

`EMBEDDING_PORT` configures the process listen port and defaults to `8001`.

## Source map

| Path | Responsibility |
| --- | --- |
| `src/app.py` | ASGI entry point |
| `src/embedding_api/transport/http.py` | HTTP contract and errors |
| `src/embedding_api/application/embed.py` | Embed use case and invariants |
| `src/embedding_api/domain/` | Results, capabilities, safe errors |
| `src/embedding_api/ports/engine.py` | Engine interface |
| `src/embedding_api/adapters/sentence_transformer.py` | Real local model |
| `src/embedding_api/adapters/fake.py` | Deterministic test double |
| `src/embedding_api/infrastructure/settings.py` | Typed configuration |

## Deployment and observability

The service is installed from the committed `uv.lock` in a multi-stage image
and starts through the `embedding-api` console command. The runtime image uses
a non-root user, a read-only root filesystem under Compose, dropped Linux
capabilities, and a dedicated writable cache volume at
`/var/cache/embedding-model`. The model is not baked into the image: the first
startup downloads the exact configured revision and later starts reuse the
named volume. Deployments without network access must seed that volume or bake
the same pinned artifact in their own image pipeline.

`GET /health/live` reports process liveness. `GET /health/ready` reports success
only after the model loads, its dimension is verified, and warm-up inference
completes. The container health check uses readiness and allows an extended
initial model-download period.

Every HTTP request writes one compact JSON event containing route, method,
status, latency, trace ID, and a health-check marker. Embedding calls also log
model/revision, device class, input count, aggregate UTF-8 byte count, and token
count when inference provides it. Error codes are logged when available. Raw
input, output vectors, headers, provider exceptions, and secrets are excluded.

Compose exposes a standalone path with `docker compose up --build
embedding-api`; it does not depend on or start the unfinished Core RAG API.
Qdrant access remains outside this service.
