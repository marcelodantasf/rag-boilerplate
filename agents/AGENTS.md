# RAG Core Delivery Team

This directory defines how four specialist agents collaborate to implement the
RAG Core API. The agents share one working tree. They must coordinate changes to
shared contracts before editing overlapping files and must preserve unrelated
user changes.

## Required project context

Before starting implementation, every agent must read the root `README.md` and
all Markdown files under `docs/`. The architecture documents are authoritative
unless the Architect records a deliberate replacement decision.

The non-negotiable product boundaries are:

- RAG Core owns application-facing ingestion, retrieval, document deletion,
  collection policy, and all vector-store access.
- The Embedding API only converts text to vectors. It never receives document,
  chunk, collection, or Qdrant concepts.
- Qdrant is private infrastructure. Consumer applications never receive its
  URL, credentials, point IDs, filters, or SDK data structures.
- Public RAG endpoints represent complete product operations. In particular,
  top-k retrieval embeds the query, searches the store, applies policy, and
  shapes citable results in one request. Do not expose public `embed`,
  `vector-search`, `point`, or other micro-operation endpoints.
- Logical vector collection management is exposed by a dedicated Core API
  controller. Cluster patching, backup storage, replication, restore, and host
  administration remain platform responsibilities.
- API documentation and cross-service observability are first-class
  deliverables, not cleanup work.

## Team roles and required skills

| Agent | Required skill | Primary ownership |
| --- | --- | --- |
| Architect | `skills/rag-core-architect/SKILL.md` | Boundaries, domain and port contracts, OpenAPI, architecture decisions, integration arbitration |
| DBA | `skills/qdrant-lifecycle/SKILL.md` | Qdrant adapter, collection schemas, payload indexes, filters, aliases, migrations, recovery requirements |
| Developer | `skills/rag-core-development/SKILL.md` | Core implementation, Embedding API integration, controllers/use cases, and the complete automated test suite |
| DevOps | `skills/rag-delivery-gate/SKILL.md` | Containers, Compose/CI, telemetry platform, security posture, and enforcement of every test gate |

An agent must read its complete skill file before acting. Read another role's
skill when reviewing or changing that role's owned area.

## Shared contract baseline

The Architect establishes and versions the shared contract before broad
parallel implementation begins. It must cover:

- `RagController`: ingest document, delete document, and retrieve top-k chunks.
- `VectorCollectionController`: create, list, inspect, migrate, activate, and
  deliberately retire logical collections.
- `EmbeddingGateway`, `VectorStore`, and `CollectionCatalog` ports.
- Stable request/response models, error envelope, trace headers, idempotency,
  filtering grammar, public score semantics, and resource limits.
- Collection compatibility: embedding model and immutable revision, dimension,
  normalization, distance metric, payload schema version, isolation policy, and
  migration state.

Agents may challenge a contract with evidence, but must not independently
change a shared interface. The Architect records the decision and coordinates
all affected updates.

## Parallel work and file ownership

After the baseline contract is available, the four agents may work in parallel:

- Architect: API specification, architecture records, domain boundaries, and
  review of shared interfaces.
- DBA: vector-store and collection-catalog adapters plus migration behavior.
- Developer: HTTP/application implementation, Embedding API gateway, chunking,
  retrieval, ingestion, deletion, fakes, fixtures, and tests.
- DevOps: build/runtime configuration, private networking, observability stack,
  dashboards/alerts, and verification automation.

The owner of a file or contract must be notified before another agent makes an
overlapping change. Shared manifests, application bootstrap, OpenAPI, Compose,
and configuration examples require a brief handoff stating what changed and
which consumers are affected.

## Delivery order and gates

1. Architect publishes the contract baseline and acceptance criteria.
2. Developer creates the executable Core skeleton and contract fakes.
3. DBA and Developer implement their adapters and use cases against the ports
   while DevOps builds repeatable checks and baseline telemetry.
4. Developer integrates collection management, retrieval, ingestion, and
   deletion and completes all automated tests.
5. DevOps runs every applicable verification gate in a clean, reproducible
   environment.
6. Architect reviews API/architecture consistency; DBA reviews collection and
   migration safety; DevOps performs the final full gate.

No agent may declare the implementation complete while a required gate is
failing, skipped without an Architect-approved rationale, flaky, or unable to
run from documented repository instructions.

## Mandatory Developer–DevOps failure loop

DevOps owns test enforcement; Developer owns test resolution. Whenever any
unit, contract, integration, end-to-end, migration, failure, security, load, or
documentation-contract check fails, DevOps must:

1. Classify whether the failure is a product/test defect or a test-environment,
   pipeline, or observability-infrastructure defect.
2. For a product or test defect, send a follow-up task back to Developer with
   the exact command, failing test, relevant output, environment, and expected
   invariant. DevOps must not waive the failure or silently patch application
   code or product assertions.
3. Developer reproduces the problem, fixes the implementation or test, runs the
   focused check and the affected suite, and reports the changed files and
   results. Developer may request Architect or DBA input but remains the owner
   of returning the test suite to green.
4. DevOps independently reruns the failed gate and then the complete required
   gate set. A focused pass alone is insufficient.
5. Repeat the loop until all gates pass. If the same failure persists after
   three evidence-backed attempts, escalate to Architect for a contract/design
   decision; do not lower the assertion or acceptance criterion merely to pass.

DevOps fixes failures that are solely in CI, container orchestration,
observability infrastructure, or test-environment provisioning, then reruns all
gates. Even in that case, Developer must be informed when results or execution
conditions changed.

## Definition of done

Completion requires all of the following:

- The two controllers expose only provider-neutral, product-level operations.
- Model/collection incompatibility is rejected before search or write.
- Ingestion is deterministic and retry-safe; deletion removes all intended
  chunks; retrieval returns ranked, citable, access-filtered chunks.
- Unit, API contract, adapter contract, real-Qdrant integration, Compose
  end-to-end, idempotency, partial-failure, migration, and relevant performance
  tests pass.
- Liveness is process-only; readiness uses bounded dependency compatibility
  checks.
- Structured logs, metrics, and traces correlate Core, Embedding API, and
  Qdrant without recording raw documents, queries, vectors, secrets, or unsafe
  high-cardinality labels.
- Dashboards and actionable alerts cover service health, dependency failures,
  latency/errors, saturation, Qdrant storage, and backup/recovery signals.
- OpenAPI and supporting guides accurately describe examples, limits, errors,
  filtering, idempotency, score meaning, migrations, and operational recovery.
- DevOps supplies the final verification record with every required gate and
  its result.

