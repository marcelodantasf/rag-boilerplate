# RAG boilerplate

This repository is a learning-oriented boilerplate for independently deployed
RAG services:

- **RAG Core API** owns ingestion, chunking, retrieval, and vector-store access.
- **Embedding API** owns text-to-vector model execution.
- **Qdrant** is private vector-database infrastructure owned by RAG Core.

The architecture references live in [docs/architecture](docs/architecture).

## Embedding API

The implemented Embedding API accepts any non-empty UTF-8 text without a
document schema or domain-specific preprocessing. One endpoint handles both a
single string and a list sent to the model as one ordered batch. It returns
plain embedding vectors; document IDs, payloads, collections, and Qdrant
mapping remain RAG Core responsibilities.

The default runtime is `sentence-transformers/all-MiniLM-L6-v2` at immutable
revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`. It produces normalized
384-dimensional vectors.

### Run locally

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required. From the service
directory:

```sh
cd services/embedding-api
uv sync --frozen --extra test
uv run embedding-api
```

The first start downloads the pinned model. Set `MODEL_CACHE_DIR` to retain it
at a specific location. The service listens on `http://localhost:8001`; change
that with `EMBEDDING_PORT`.

### Run in a container

Copy the example configuration if overrides are needed, then start only the
implemented service. This path does not start or depend on the unfinished RAG
Core API:

```sh
cp .env.example .env
docker compose up --build embedding-api
```

Compose stores the pinned model artifact in the `embedding-model-cache` volume,
so subsequent starts do not download it again. Startup may take longer on an
empty cache. Wait for readiness before sending traffic:

```sh
curl --fail http://localhost:8001/health/ready
```

Qdrant remains independently runnable with `docker compose up qdrant`. Do not
use the full `application` profile yet because `core-rag-api` is still a
scaffold.

### API usage

Single document:

```sh
curl --request POST http://localhost:8001/v1/embeddings \
  --header 'content-type: application/json' \
  --header 'x-trace-id: example-single-1' \
  --data '{
    "model": "all-MiniLM-L6-v2",
    "input": "Any law excerpt, recipe, paper, source code, or other text."
  }'
```

Batch encoding uses the same endpoint and preserves array order with `index`:

```sh
curl --request POST http://localhost:8001/v1/embeddings \
  --header 'content-type: application/json' \
  --data '{
    "model": "all-MiniLM-L6-v2",
    "input": [
      "Article 1. Every person has the right to education.",
      "Whisk eggs and sugar, then fold in flour.",
      "Gradient descent updates parameters using the loss derivative."
    ]
  }'
```

Success responses use one shape for single and batch requests:

```json
{
  "model": "all-MiniLM-L6-v2",
  "dimension": 384,
  "vectors": [
    {"index": 0, "embedding": [0.012, -0.043]}
  ],
  "usage": {"input_count": 1, "input_tokens": 14}
}
```

The shortened vector above is illustrative; real vectors contain exactly 384
finite floats. Invalid input returns `422`, oversized input returns `413`, an
unsupported model returns `404`, and an unavailable engine returns `503`.
Errors have a stable `code`, safe `message`, `trace_id`, and optional limit
`details`. The response always returns `x-trace-id`; a valid incoming value is
propagated.

### Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `EMBEDDING_PORT` | `8001` | Process listen port |
| `EMBEDDING_API_PORT` | `8001` | Compose host port |
| `DEFAULT_MODEL_ID` | `all-MiniLM-L6-v2` | Public API model identity |
| `MODEL_SOURCE` | `sentence-transformers/all-MiniLM-L6-v2` | Artifact repository |
| `MODEL_REVISION` | pinned commit | Immutable artifact revision |
| `EXPECTED_DIMENSION` | `384` | Startup vector contract check |
| `INFERENCE_DEVICE` | `cpu` | Sentence Transformers device |
| `MODEL_CACHE_DIR` | unset locally | Artifact cache directory |
| `NORMALIZE_EMBEDDINGS` | `true` | L2-normalize vectors |
| `ENGINE_BATCH_SIZE` | `32` | Model inference batch size |
| `MAX_BATCH_ITEMS` | `64` | Maximum inputs per request |
| `MAX_INPUT_BYTES` | `65536` | UTF-8 bytes per input |
| `MAX_TOTAL_INPUT_BYTES` | `262144` | UTF-8 bytes per request |
| `MAX_BATCH_TOKENS` | `8192` | Token budget per request |
| `LOG_LEVEL` | `INFO` | Structured request log threshold |

The committed `uv.lock` pins transitive Python dependencies. Keep
`MODEL_REVISION` and `EXPECTED_DIMENSION` aligned; changing the model contract
requires reindexing downstream vector collections.

### Operations and security

The image runs as a non-root user with a read-only root filesystem, dropped
Linux capabilities, a writable temporary filesystem, and a dedicated model
cache volume. It starts one inference worker to avoid loading duplicate model
copies; scale replicas at the orchestrator level when capacity controls are
added.

`GET /health/live` checks the process. `GET /health/ready` becomes successful
only after the pinned model loads and its warm-up inference completes. Each
request emits one JSON log with trace ID, route, status, latency, health-check
marker, and—when available—model/revision, device, input count, aggregate byte
count, and token count. Raw document text, vectors, headers, and secrets are
never logged.

Keep the service private in production and remove the development host-port
mapping. The current milestone intentionally has no authentication, bounded
queue/concurrency scheduler, or metrics exporter.

Copy `.env.example` to `.env` for local overrides. Do not commit `.env`; the
root `.gitignore` excludes it, virtual environments, model/runtime state, and
macOS `.DS_Store` files.
