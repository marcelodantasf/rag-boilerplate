# RAG Core API implementation

Status: implemented and verified for the single-replica boilerplate topology.

This guide describes the code that is running now. The
[OpenAPI 3.1 document](../api/rag-core-openapi.yaml) remains the public contract;
the [architecture decisions](../architecture/rag-core-api-decisions.md) explain
why the boundary is shaped this way. This document connects both to source,
runtime operations, verified behavior, and the limitations that must be closed
before a production rollout.

## 1. Delivered boundary

RAG Core is the only application-facing RAG boundary and the only application
client of Qdrant. It exposes complete product operations instead of asking a
consumer to coordinate embedding, search, point lookup, and collection-provider
requests.

| Controller | Public operations | Owns |
| --- | --- | --- |
| RAG operations | Index a document, delete a document, retrieve top-k chunks | Normalization, deterministic chunking, embedding calls, compatibility checks, metadata/filter validation, vector writes/search/deletion, public scoring |
| Vector collections | Create, list, inspect, provision generation, activate, retire | Logical IDs, immutable generation contracts, Qdrant collection/index provisioning, alias cutover, durable catalog state |
| Health | Live, ready | Process liveness and bounded dependency readiness |

The public surface intentionally has no endpoint for raw embedding, vector
search, vector upsert, point retrieval, Qdrant filters, physical collection
names, snapshots, replication, patching, or database credentials. Embedding's
`POST /v1/embeddings` is an internal dependency call, not part of Core's
consumer workflow.

The nine public and health paths are:

| Method | Path | Current behavior |
| --- | --- | --- |
| `POST` | `/v1/rag/documents` | Synchronously normalize, chunk, embed, upsert, then remove older document versions |
| `DELETE` | `/v1/rag/documents/{document_id}` | Delete all chunks and versions for the document in the active generation |
| `POST` | `/v1/rag/retrievals` | Embed one query and return thresholded, ranked, citable chunks |
| `POST` | `/v1/vector-collections` | Provision and activate the first immutable generation |
| `GET` | `/v1/vector-collections` | Cursor-page active logical collections |
| `GET` | `/v1/vector-collections/{collection_id}` | Return the logical collection and every retained generation |
| `POST` | `/v1/vector-collections/{collection_id}/generations` | Provision and verify a new empty generation, leaving it `ready` |
| `POST` | `/v1/vector-collections/{collection_id}/activate` | Verify and activate a `ready` target if the expected active generation still matches |
| `DELETE` | `/v1/vector-collections/{collection_id}` | Retire the active generation after confirmation and optimistic concurrency checks |
| `GET` | `/health/live` | Return process liveness without calling dependencies |
| `GET` | `/health/ready` | Concurrently check Embedding, Qdrant, and the Qdrant-backed catalog under bounded timeouts |

## 2. Runtime architecture

```text
application client
        |
        | HTTP: logical IDs, text, metadata, provider-neutral filters
        v
RAG Core / transport controller
        |
        +--> RAG application service
        |      +--> deterministic chunking
        |      +--> EmbeddingGateway ----HTTP----> Embedding API
        |      +--> VectorStore --------HTTP-----> Qdrant data collections
        |
        +--> Collection application service
               +--> VectorStore -----------------> Qdrant collections/indexes/aliases
               +--> CollectionCatalog -----------> __rag_core_catalog_v1

OTLP traces + metrics --> OpenTelemetry Collector --> Prometheus / Tempo / Loki
structured stdout logs --------------------------------------------> operator
```

The implementation follows ports and adapters:

- transport parses strict external schemas and maps stable errors;
- application services orchestrate product operations without Qdrant request
  types;
- domain models hold provider-neutral contracts and deterministic identities;
- ports define Embedding, vector-store, catalog, and idempotency capabilities;
- adapters translate those ports to HTTP calls or test fakes;
- infrastructure owns environment validation and telemetry setup.

Core starts from the installed `rag_core.app:app` ASGI object through the
`core-rag-api` console entrypoint. The container does not depend on a
repository-only top-level module.

## 3. Logical collections and compatibility

A logical collection ID remains stable while physical generations change. Each
generation persists an immutable contract:

- embedding model ID and configured revision identity;
- vector dimension and normalization flag;
- distance metric: `cosine`, `dot`, or `euclid`;
- index schema version;
- typed metadata fields and whether each field is indexed;
- isolation policy;
- generation ID, physical name, state, timestamps, and source generation.

Core compares the configured revision string exactly but cannot prove that a
remote model label is immutable; operators must supply a pinned artifact
revision. The default contract is the normalized 384-dimensional
`all-MiniLM-L6-v2` model at revision
`c9745ed1d9f207416be6d2e6f8de32d1f16199bf` with cosine distance.
Unnormalized dot-product contracts are rejected. Embedding responses are
checked against model, revision, dimension, and normalization before any vector
write or search.

Metadata types are `keyword`, `integer`, `float`, and `boolean`. Ingestion
rejects undeclared fields and type mismatches. Consumer filters may use `eq`,
`in`, `gt`, `gte`, `lt`, and `lte`; range comparisons are numeric only.
Groups support `all`, `any`, and a single-clause `not`, with at most three
levels, ten conditions, and twenty values in an `in` predicate. Only fields
declared as indexed may be filtered.

Core maps provider-independent metadata types to Qdrant payload indexes. System
payload fields—including document/version/chunk contracts and the private
`__tenant_id`—are indexed separately from consumer metadata. Physical names
are derived from the logical and generation IDs but never leave the adapter.

### Catalog persistence

The Qdrant catalog adapter stores generation contracts in the reserved
`__rag_core_catalog_v1` collection and indexes `collection_id`,
`generation_id`, and `state`. It supports durable lookup, generation history,
active collection paging, state transitions, retirement retention metadata, and
readiness checks.

Catalog durability does not make its multi-record state change transactional.
The adapter uses expected-state checks, a process-local lock, one batched Qdrant
upsert, and post-read validation. Qdrant 1.12 does not provide a compare-and-set
transaction spanning the old and new catalog records. Consequently:

1. the verified deployment runs exactly one Core replica;
2. activation fails closed on stale expected state in that topology;
3. multiple active Core writers could race across replicas;
4. horizontal scaling requires a transactional catalog implementation or
   equivalent distributed serialization before adding a second Core replica.

This is the most important deployment constraint in the current milestone.

## 4. Collection lifecycle

Generation states are `building`, `ready`, `active`, `retired`, and
`failed`.

### Initial creation

1. Validate the logical ID and immutable contract.
2. Reject an existing logical collection.
3. Add the `building` catalog generation.
4. Create the physical Qdrant collection and approved payload indexes.
5. Verify the provider collection against the contract.
6. Mark the generation `ready`.
7. Activate the logical alias.
8. Mark the generation `active`.

A provisioning failure while still `building` records `failed`. The logical
ID is the public resource; the generated physical name stays private.

### Generation provisioning and activation

The generation route provisions a target and verifies its schema, indexes, and
availability. It returns `201` with state `ready`, or `200` when replaying a
completed idempotent request. It does **not** copy retained chunk text, call an
external source of truth, re-embed, re-chunk, or perform quality validation.
Unknown fields are rejected, so a caller cannot submit validation inputs that
Core would silently ignore.

Before activation, an authorized operator must rebuild all required documents
into the target and validate counts, isolation, filtering, and retrieval quality
through an out-of-band controlled process. The public ingestion route always
targets the active generation, so this milestone does not expose a public
target-generation ingestion route. Do not activate an empty target in a
data-bearing environment.

Activation accepts the target generation and
`expected_active_generation_id`. Core re-reads and verifies the target,
rejects a stale predecessor, switches the Qdrant alias, promotes the target, and
retires the predecessor. The physical predecessor remains retained.

The lower-level catalog can revalidate a retained predecessor to `ready` and
activate it for rollback, and this path is covered by integration tests. That
revalidation transition is not yet a public HTTP operation; a production
rollback workflow needs an authenticated operator endpoint or runbook before
relying on it.

### Retirement

Retirement requires:

- `X-Confirm-Retirement` equal to the logical collection ID; and
- `If-Match` equal to the active generation ID.

Core removes the active alias and marks the generation retired with a default
seven-day retention timestamp. It does not immediately delete physical data.
Automatic garbage collection after retention is not implemented; platform
operations must manage final deletion.

## 5. RAG operation behavior

### Document ingestion

The synchronous ingestion flow is:

1. validate the logical collection ID and load its active contract;
2. verify the physical Qdrant collection;
3. normalize line endings and whitespace deterministically;
4. validate UTF-8 size, document ID, and declared metadata;
5. derive a SHA-256 content version;
6. chunk text with configured size/overlap and deterministic chunk IDs;
7. call Embedding in bounded batches;
8. verify every embedding result against the active contract;
9. derive stable point IDs and upsert all new-version chunks;
10. delete older versions for the same document and trusted tenant.

The order is deliberate: the new version is present before old versions are
removed. If cleanup fails after the upsert, the request returns an error;
replaying the same content reuses deterministic identities and repairs cleanup.

`Idempotency-Key` is optional on ingestion and accepts 8–128 printable ASCII
characters. A completed same-payload replay returns `200` and
`Idempotency-Replayed: true`; a different payload under the same key returns
`409` with `idempotency_conflict`.

### Document deletion

Deletion resolves the active generation, verifies it, and removes every chunk
and version matching the logical document ID plus the adapter's trusted tenant.
Deleting an absent document succeeds with `chunks_deleted: 0`. Core never
deletes an authoritative source document.

### Top-k retrieval

Retrieval:

1. validates query size, `top_k`, `minimum_score`, and the filter tree;
2. embeds the normalized query exactly once;
3. validates embedding compatibility;
4. ANDs the consumer filter with the trusted server-derived tenant filter;
5. requests bounded matches from Qdrant;
6. converts the metric value to a public `[0,1]` score;
7. sorts descending by score with `chunk_id` as the stable tie-breaker;
8. applies `minimum_score`, limits to `top_k`, and returns citable text and
   metadata.

Public scoring is:

| Metric | Public score |
| --- | --- |
| Cosine or normalized dot | `(clamp(similarity, -1, 1) + 1) / 2` |
| Euclidean | `1 / (1 + max(distance, 0))` |

Scores are rounded to six decimal places. They are monotonic within one
collection contract, not calibrated probabilities, and must not be compared
across collections or embedding models. An empty result list is a successful
no-match response.

## 6. Isolation and security

The Qdrant adapter never trusts consumer metadata as an access-control source.
It stores a private `__tenant_id`, requires trusted tenant criteria for search,
and includes the configured tenant in document deletion and old-version
cleanup. Real-Qdrant tests prove that two adapter tenants cannot search or delete
each other's points.

The shipped HTTP application has no authentication middleware and constructs
the adapter and RAG trusted filter with tenant `local`. V1 accepts only the
`shared` isolation policy; `collection_per_tenant` is rejected because its
provisioning and authenticated routing do not exist yet. Before exposing Core
beyond a trusted local environment:

- authenticate every request;
- resolve tenant and access policy from trusted identity, never the body;
- inject that trusted context consistently into RAG and vector adapters;
- implement and test `collection_per_tenant` before adding it to a future
  public contract;
- use a secret manager for `VECTOR_DB_API_KEY`;
- remove Embedding and Qdrant host-port mappings.

The Core image runs as a non-root user with a read-only root filesystem,
`no-new-privileges`, all Linux capabilities dropped, bounded temporary storage,
and a health check. The container gate also verifies termination behavior and
the installed entrypoint.

## 7. HTTP conventions and limits

Request and response models reject unknown fields. Logical collection IDs,
document IDs, generation IDs, metadata types, enum values, and filter shapes
are validated before orchestration.

| Concern | Behavior |
| --- | --- |
| Trace context | Accepts valid W3C `traceparent`/`tracestate`; otherwise accepts a safe `x-trace-id` or creates one |
| Response correlation | Always returns `x-trace-id` |
| Request timeout | `x-request-timeout-ms`, clamped to 100–30,000 ms; default 10,000 ms |
| Document size | 262,144 UTF-8 bytes by default |
| Query size | 16,384 UTF-8 bytes at the service layer; the current HTTP schema additionally caps characters at 8,192 |
| Chunking | 1,000 characters with 200-character overlap by default |
| Chunks per document | 512 by default |
| Embedding batch | 64 chunks by default |
| Retrieval | `top_k` 1–100 and `minimum_score` 0–1 |
| Metadata | 32 fields, scalar keyword/integer/float/boolean values |
| Collection list | `limit` 1–100, default 50, with an opaque cursor |

Document ingestion, logical collection creation, and target-generation
provisioning accept `Idempotency-Key`. Keys are scoped to the fixed local
tenant and operation; generation provisioning also binds the path collection
ID. Completed same-request replays return `200` with
`Idempotency-Replayed: true`, while a different canonical request returns
`409/idempotency_conflict`. The default store is in memory and cleared on
restart. Although its port accepts a TTL, the local adapter is not a durable
cross-process guarantee. Production requires a shared, TTL-enforcing adapter.

Stable application errors use:

```json
{
  "code": "idempotency_conflict",
  "message": "The idempotency key is already bound to a different request",
  "trace_id": "8b93b22d0db947b1abf0938d80fc8d4b"
}
```

Implemented codes cover invalid/oversized input, not found, state and
idempotency conflicts, embedding schema/contract failures, and embedding/vector
unavailability or timeout. Pydantic validation is normalized to the same safe
envelope with field paths.

### Contract alignment

The checked-in OpenAPI describes the shipped route names, strict request
schemas, success bodies and headers, replay status, precondition failures, and
stable error envelope. Contract tests cover first execution and idempotent
replay for mutating operations, reject unknown generation fields, and assert
that collection and RAG response bodies contain no undeclared properties.

## 8. Observability

Setting `OTEL_EXPORTER_OTLP_ENDPOINT` enables OTLP/HTTP metrics and traces.
FastAPI request spans and metrics are joined by explicit spans around chunking,
embedding, vector-store work, and generation activation. Active W3C trace context
is propagated from Core to Embedding.

Structured request logs contain trace ID, method, route template, status,
latency, and health-check classification. They do not log bodies, raw queries,
chunk text, vectors, secrets, or arbitrary headers.

Product metrics cover:

- ingestion duration, indexed documents/chunks, and embedding batch size;
- retrieval duration, requested top-k, result count, and no-match count;
- dependency request count, duration, errors, timeouts, and retries;
- embedding contract mismatches and validation rejections;
- idempotency phases;
- generation provisioning/activation operations, phase, and duration.

Metric attributes are allowlisted to `dependency`, `operation`, `outcome`,
`error_code`, and `phase`. Tests reject query, content, vector,
`collection_id`, `document_id`, and `tenant_id` labels to control cardinality
and data exposure.

The optional local observability profile includes:

- OpenTelemetry Collector for OTLP ingestion and routing;
- Prometheus for metrics and alert evaluation;
- Tempo for traces;
- Loki for OTLP logs;
- Grafana with provisioned data sources and the RAG overview dashboard;
- cAdvisor for explicitly privileged local container metrics.

Alert rules cover Core absence, 5xx ratio, retrieval latency, contract mismatch,
generation-lifecycle failure, dependency availability, container restarts/resources, and
snapshot freshness when an external backup job publishes the required metric.
See [the observability runbook](../../observability/README.md). This local
single-node stack is not a production high-availability telemetry platform.

## 9. Health and failure behavior

`/health/live` only proves that the process can answer. It does not call
dependencies or report capacity.

`/health/ready` validates the configured embedding contract and concurrently
checks Embedding, the vector store, and the catalog. Each dependency has a
one-second default bound and all checks share a three-second total default.
Exceptions and total timeout fail safe to `503` with per-check codes; readiness
does not leak dependency exception text.

Dependency HTTP adapters apply caller deadlines, map timeouts separately from
unavailability, validate provider response shapes, and retry only the bounded
operations encoded in their adapters. Application compatibility failures occur
before vector writes.

## 10. Configuration

Compose supplies the complete development configuration through
[the environment template](../../.env.example). Core's direct settings are:

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_PORT` | `8000` | Process listen port |
| `EMBEDDING_BASE_URL` | `http://embedding-api:8001` | Private Embedding endpoint |
| `EMBEDDING_TIMEOUT_MS` | `10000` | Backward-compatible Embedding timeout input |
| `EMBEDDING_TIMEOUT_SECONDS` | derived from milliseconds | Direct timeout override |
| `READINESS_DEPENDENCY_TIMEOUT_SECONDS` | `1` | Per-dependency readiness bound |
| `READINESS_TOTAL_TIMEOUT_SECONDS` | `3` | Overall readiness bound |
| `VECTOR_DB_URL` | `http://qdrant:6333` | Private Qdrant endpoint |
| `VECTOR_DB_API_KEY` | unset | Qdrant credential |
| `DEFAULT_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Active model contract |
| `EMBEDDING_MODEL_REVISION` | pinned commit | Immutable model revision |
| `EMBEDDING_DIMENSION` | `384` | Expected vector dimension |
| `NORMALIZE_EMBEDDINGS` | `true` | Expected embedding normalization |
| `DISTANCE_METRIC` | `cosine` | Default metric contract |
| `DEFAULT_CHUNK_SIZE` | `1000` | Chunk characters |
| `DEFAULT_CHUNK_OVERLAP` | `200` | Adjacent overlap |
| `MAX_DOCUMENT_BYTES` | `262144` | Normalized document UTF-8 limit |
| `MAX_CHUNKS_PER_DOCUMENT` | `512` | Chunk fan-out limit |
| `MAX_EMBEDDING_BATCH_ITEMS` | `64` | Internal Embedding batch |
| `MAX_SEARCH_TOP_K` | `100` | Retrieval result limit |
| `MAX_QUERY_BYTES` | `16384` | Normalized query UTF-8 limit |
| `MAX_METADATA_FIELDS` | `32` | Metadata field limit |
| `MAX_METADATA_VALUE_BYTES` | `1024` | Reserved metadata value limit |
| `LOG_LEVEL` | `INFO` | Structured request log threshold |
| `OTEL_SERVICE_NAME` | `rag-core-api` | Telemetry resource identity |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset in direct process | Enables OTLP export |

Configuration is parsed once and fails fast for invalid values, incompatible
chunk overlap, invalid URLs, unsupported log levels, and inconsistent readiness
timeouts.

## 11. Source map

| Concern | Primary source |
| --- | --- |
| Installed ASGI application | [`src/rag_core/app.py`](../../services/core-rag-api/src/rag_core/app.py) |
| HTTP schemas/controllers/errors/health | [`transport/http.py`](../../services/core-rag-api/src/rag_core/transport/http.py) |
| RAG orchestration and public scoring | [`application/rag.py`](../../services/core-rag-api/src/rag_core/application/rag.py) |
| Collection lifecycle | [`application/collections.py`](../../services/core-rag-api/src/rag_core/application/collections.py) |
| Validation and filter rules | [`application/validation.py`](../../services/core-rag-api/src/rag_core/application/validation.py) |
| Deterministic normalization/chunk IDs | [`domain/chunking.py`](../../services/core-rag-api/src/rag_core/domain/chunking.py) |
| Domain contracts and filter AST | [`domain/models.py`](../../services/core-rag-api/src/rag_core/domain/models.py) |
| Stable application errors | [`domain/errors.py`](../../services/core-rag-api/src/rag_core/domain/errors.py) |
| Embedding port and HTTP adapter | [`ports/embedding.py`](../../services/core-rag-api/src/rag_core/ports/embedding.py), [`adapters/embedding/http.py`](../../services/core-rag-api/src/rag_core/adapters/embedding/http.py) |
| Vector/catalog ports | [`ports/vector_store.py`](../../services/core-rag-api/src/rag_core/ports/vector_store.py) |
| Qdrant data adapter | [`adapters/vector_store/qdrant.py`](../../services/core-rag-api/src/rag_core/adapters/vector_store/qdrant.py) |
| Qdrant catalog adapter | [`adapters/catalog/qdrant.py`](../../services/core-rag-api/src/rag_core/adapters/catalog/qdrant.py) |
| Idempotency port/default adapter | [`ports/idempotency.py`](../../services/core-rag-api/src/rag_core/ports/idempotency.py), [`adapters/fakes.py`](../../services/core-rag-api/src/rag_core/adapters/fakes.py) |
| Settings | [`infrastructure/settings.py`](../../services/core-rag-api/src/rag_core/infrastructure/settings.py) |
| OTLP setup and safe product instruments | [`infrastructure/observability.py`](../../services/core-rag-api/src/rag_core/infrastructure/observability.py), [`infrastructure/instruments.py`](../../services/core-rag-api/src/rag_core/infrastructure/instruments.py) |
| Container and local topology | [`Dockerfile`](../../services/core-rag-api/Dockerfile), [`compose.yaml`](../../compose.yaml) |
| Verification orchestrator | [`scripts/verify-rag-core.sh`](../../scripts/verify-rag-core.sh) |

## 12. Run and verify

From the repository root:

```sh
cp .env.example .env
docker compose --profile application up --build --wait
curl --fail http://localhost:8000/health/ready
```

With telemetry:

```sh
docker compose \
  --profile application \
  --profile observability \
  up --build --wait
```

The complete delivery gate is one command:

```sh
./scripts/verify-rag-core.sh all
```

The gate runs Embedding and Core unit suites, OpenAPI validation and forbidden
micro-endpoint checks, pinned real-model tests, real-Qdrant integration,
container security checks, a snapshot restore drill, the Compose product
workflow, reference load, and observability configuration validation. Use an
individual gate during development:

```sh
./scripts/verify-rag-core.sh unit
./scripts/verify-rag-core.sh contract
./scripts/verify-rag-core.sh integration
./scripts/verify-rag-core.sh e2e
./scripts/verify-rag-core.sh observability
```

The final independently observed results and environment are recorded in the
[delivery verification report](../../verification/rag-core-delivery.md).

## 13. Production-readiness checklist

The implemented single-replica boilerplate is a sound baseline, not a claim that
the following production work is complete:

- replace the Qdrant catalog or serialize it across replicas before scaling
  Core horizontally;
- replace process-local idempotency with a durable, shared, TTL-enforcing store;
- add authentication and trusted request-derived tenant/access context;
- implement target-generation rebuild/reindex and validation before generation
  activation;
- expose an authenticated revalidation/rollback workflow and retention garbage
  collection;
- provision Qdrant replication, backups, remote snapshot storage, restore
  objectives, upgrades, and capacity management at the platform layer;
- remove development host ports and local credentials;
- add workload-specific SLOs, admission/concurrency controls, and production
  telemetry retention.

None of these items changes the intended boundary: consumers continue to call
product-level RAG operations, while Core alone owns vector-database semantics.
