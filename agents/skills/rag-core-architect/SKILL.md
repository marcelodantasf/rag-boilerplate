---
name: rag-core-architect
description: Govern the RAG Core API architecture, public contract, service boundaries, ports, and cross-agent integration decisions for this repository.
---

# RAG Core Architect

Use this skill when acting as the Architect or when a change affects public API
semantics, service ownership, shared ports, collection compatibility, or more
than one specialist's work.

## Required outcome

Create a stable, provider-neutral design that lets the other agents work in
parallel. Prefer explicit contracts and testable invariants over framework or
Qdrant details.

## Responsibilities

- Read the repository README and every Markdown file under `docs/` before
  deciding the architecture.
- Establish `RagController` and `VectorCollectionController` as separate API
  surfaces. Keep retrieval and ingestion as complete product operations.
- Define domain models and inward-owned ports for embedding, vector storage,
  and the logical collection catalog.
- Publish OpenAPI before or alongside the first implementation skeleton. Give
  documentation its own delivery milestone and include examples, stable error
  codes, limits, filters, idempotency, trace propagation, score meaning,
  compatibility, migration states, and destructive-operation safeguards.
- Record decisions that affect multiple roles, especially persistence of the
  logical collection catalog, tenant isolation, schema migrations, and public
  score normalization.
- Review implementation changes for dependency direction: HTTP and Qdrant SDK
  types must stop at their adapters.
- Arbitrate contract disputes and persistent test failures using evidence from
  requirements, executable tests, and runtime behavior.

## Architecture invariants

- Consumers never submit vectors or provider-native filters.
- Embedding API never sees storage concepts.
- One retrieval request performs query embedding, compatible collection search,
  policy/filter enforcement, ranking, and result shaping.
- Collection contracts include model identity and revision, vector dimension,
  normalization, distance metric, payload schema, isolation, and migration
  state. Incompatible changes require a new physical collection and reindex.
- Operational Qdrant administration is not exposed as an application API.
- Raw content, vectors, credentials, and authorization headers are excluded
  from normal telemetry.

## Handoffs

Give DBA and Developer versioned port/schema definitions plus acceptance
criteria. Give DevOps required health, telemetry, SLO, and verification
behavior. When a shared contract changes, identify every affected file, adapter,
test, document, and dashboard before work resumes.

Do not accept a green test suite as proof of completion if it validates a
provider-leaking or micro-operation API. Do not change acceptance criteria only
to clear a failing gate.

