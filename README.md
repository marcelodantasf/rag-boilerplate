# RAG boilerplate

This repository is a learning-oriented, independently deployable RAG stack:

- **RAG Core API** owns product-level document ingestion, top-k retrieval,
  logical vector-collection management, and every Qdrant interaction.
- **Embedding API** owns text-to-vector model execution and is private to Core.
- **Qdrant** stores vectors and Core's logical collection catalog. Application
  clients never receive database credentials, physical collection names, point
  IDs, or raw vector operations.

The RAG Core API and its delivery stack are implemented and independently
verified. Start with the [OpenAPI contract](docs/api/rag-core-openapi.yaml), the
[implemented design](docs/impl/rag-core-implementation.md), and the
[delivery record](verification/rag-core-delivery.md).

## Run the complete application

Docker Compose is the shortest path. Copy the development defaults and start
Core, Embedding, and Qdrant:

```sh
cp .env.example .env
docker compose --profile application up --build --wait
```

The first start downloads the pinned embedding model into the
`embedding-model-cache` volume. Confirm the public service is ready:

```sh
curl --fail http://localhost:8000/health/ready
```

Core is available at `http://localhost:8000`; Embedding and Qdrant have
development host mappings at ports `8001` and `6333`. Keep both dependencies
private and remove their host mappings in production.

To include the local telemetry pipeline and provisioned dashboard:

```sh
docker compose \
  --profile application \
  --profile observability \
  up --build --wait
```

Grafana is then available at `http://localhost:3000` and Prometheus at
`http://localhost:9090`. The credentials in `.env.example` are local-only
defaults and must be replaced before any shared deployment.

## RAG Core API

Core exposes two deliberately separate controller surfaces. RAG routes express
complete product operations; there are no public embed, vector-search, point,
physical-collection, or Qdrant endpoints.

| Controller | Method and path | Product operation |
| --- | --- | --- |
| RAG | `POST /v1/rag/documents` | Normalize, chunk, embed, and index one document |
| RAG | `DELETE /v1/rag/documents/{document_id}` | Delete every indexed chunk for a document |
| RAG | `POST /v1/rag/retrievals` | Embed a query once and return ranked top-k chunks |
| Vector collections | `POST /v1/vector-collections` | Create a logical collection and active generation |
| Vector collections | `GET /v1/vector-collections` | List active logical collections |
| Vector collections | `GET /v1/vector-collections/{collection_id}` | Inspect all retained generations |
| Vector collections | `POST /v1/vector-collections/{collection_id}/generations` | Provision and verify an empty compatible generation |
| Vector collections | `POST /v1/vector-collections/{collection_id}/activate` | Cut over to a ready generation with optimistic concurrency |
| Vector collections | `DELETE /v1/vector-collections/{collection_id}` | Deliberately retire a logical collection |

Liveness is `GET /health/live`. Readiness is `GET /health/ready` and performs
bounded, concurrent checks of configuration, Embedding, Qdrant, and the catalog.

### Create, index, and retrieve

Create a logical collection with its immutable embedding and metadata contract:

```sh
curl --fail-with-body \
  --request POST http://localhost:8000/v1/vector-collections \
  --header 'content-type: application/json' \
  --data '{
    "collection_id": "employee-handbook",
    "embedding": {
      "model_id": "all-MiniLM-L6-v2",
      "revision": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
      "dimension": 384,
      "normalized": true,
      "distance_metric": "cosine"
    },
    "index_schema_version": 1,
    "metadata_fields": [
      {"name": "department", "type": "keyword", "indexed": true},
      {"name": "year", "type": "integer", "indexed": true}
    ],
    "isolation_policy": "shared"
  }'
```

Indexing is one synchronous product operation. A valid `Idempotency-Key`
replays the completed response and rejects a different payload with
`409/idempotency_conflict`:

```sh
curl --fail-with-body \
  --request POST http://localhost:8000/v1/rag/documents \
  --header 'content-type: application/json' \
  --header 'Idempotency-Key: handbook-load-0001' \
  --data '{
    "collection_id": "employee-handbook",
    "document_id": "leave-policy-2026",
    "content": "Parental leave is available for sixteen weeks.",
    "metadata": {"department": "people", "year": 2026}
  }'
```

Retrieval embeds the query, applies provider-neutral filters and the server's
trusted tenant filter, searches the active generation, normalizes scores, and
returns citable chunks in one request:

```sh
curl --fail-with-body \
  --request POST http://localhost:8000/v1/rag/retrievals \
  --header 'content-type: application/json' \
  --data '{
    "collection_id": "employee-handbook",
    "query": "How long is parental leave?",
    "top_k": 5,
    "minimum_score": 0.65,
    "filter": {
      "all": [
        {"field": "department", "operator": "eq", "value": "people"},
        {"field": "year", "operator": "gte", "value": 2025}
      ]
    }
  }'
```

Every response includes `x-trace-id`. Clients may send W3C `traceparent` and
`tracestate`, a legacy `x-trace-id`, and a bounded `x-request-timeout-ms`.
Errors use a stable JSON envelope with `code`, safe `message`, `trace_id`, and
optional `details`. See the [OpenAPI contract](docs/api/rag-core-openapi.yaml)
for complete schemas, examples, limits, filters, lifecycle responses, and error
codes.

## Core development and verification

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required. To run Core
outside Compose, start Embedding and Qdrant first, then point Core at their host
ports:

```sh
docker compose up --build --wait embedding-api qdrant
cd services/core-rag-api
uv sync --frozen --extra test
EMBEDDING_BASE_URL=http://127.0.0.1:8001 \
VECTOR_DB_URL=http://127.0.0.1:6333 \
uv run core-rag-api
```

Run the complete delivery gate from the repository root:

```sh
./scripts/verify-rag-core.sh all
```

Individual gates are `unit`, `contract`, `model`, `integration`, `image`,
`recovery`, `e2e`, `load`, and `observability`. The full gate validates both
services, OpenAPI boundaries, real model and Qdrant behavior, the locked image,
the complete product workflow, snapshot recovery, load, and telemetry assets.

Core emits structured request logs and optional OTLP traces and metrics. The
instrumentation covers HTTP, ingestion, retrieval, dependency calls,
idempotency, schema mismatches, generation lifecycle, and validation without placing
queries, content, vectors, collection IDs, document IDs, or tenant IDs in
telemetry attributes. Deployment details, dashboards, and alert rules are in
[observability/README.md](observability/README.md).

### Deployment boundary

The catalog is durable in Core's reserved Qdrant collection and uses
expected-state checks, batched writes, post-write verification, and a process
lock. Qdrant cannot provide a transactional compare-and-set across the catalog
records involved in activation. Therefore, the delivered and verified topology
runs **one Core replica**. Before horizontally scaling Core, move the catalog to
a transactional store or add equivalent distributed serialization.

The current milestone also uses a process-local idempotency store and
a fixed `local` trusted tenant. Production deployments need a durable shared
idempotency adapter and an authenticated tenant/access-policy resolver.
Generation provisioning creates and verifies an empty target; an authorized
out-of-band workflow must rebuild and validate documents before activation.
These boundaries and the safe cutover procedure are detailed in the
[implemented design](docs/impl/rag-core-implementation.md).

## Embedding API

The Embedding API accepts non-empty UTF-8 text without a document schema or
domain-specific preprocessing. One endpoint handles either a string or an
ordered list sent to the model as a single batch. It returns vectors only;
document IDs, payloads, collections, chunking, and Qdrant mapping remain Core
responsibilities.

The default runtime is `sentence-transformers/all-MiniLM-L6-v2` at immutable
revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`. It produces normalized
384-dimensional vectors.

### Run Embedding independently

```sh
cd services/embedding-api
uv sync --frozen --extra test
uv run embedding-api
```

The service listens on `http://localhost:8001`; change that with
`EMBEDDING_PORT`. Set `MODEL_CACHE_DIR` to retain the model at a specific local
path. To run only its container:

```sh
docker compose up --build --wait embedding-api
curl --fail http://localhost:8001/health/ready
```

### Embedding usage

```sh
curl --request POST http://localhost:8001/v1/embeddings \
  --header 'content-type: application/json' \
  --header 'x-trace-id: example-batch-1' \
  --data '{
    "model": "all-MiniLM-L6-v2",
    "input": [
      "Article 1. Every person has the right to education.",
      "Whisk eggs and sugar, then fold in flour."
    ]
  }'
```

The response contains the pinned model ID, dimension, ordered vectors with an
`index`, and usage counts. Real vectors contain exactly 384 finite floats.
Invalid input returns `422`, oversized input returns `413`, an unsupported model
returns `404`, and an unavailable engine returns `503`. See the
[Embedding implementation guide](docs/impl/embedding-api-implementation.md)
for its full configuration, operations, and security model.

Copy `.env.example` to `.env` for local overrides. Do not commit `.env`; the
root `.gitignore` excludes it, virtual environments, model/runtime state, and
macOS `.DS_Store` files.
