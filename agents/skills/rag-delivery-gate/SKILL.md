---
name: rag-delivery-gate
description: Operate and enforce the RAG delivery pipeline, observability stack, container security, and mandatory test-feedback loop with the Developer agent.
---

# RAG Delivery Gate

Use this skill for the DevOps role, CI/Compose work, observability, release
verification, or enforcement of implementation acceptance gates.

## Platform ownership

- Build reproducible, locked Core images and Compose wiring with health checks,
  private service networking, non-root execution, read-only filesystems where
  practical, dropped capabilities, resource limits, and graceful termination.
- Keep Embedding API and Qdrant private outside explicit local-development
  conveniences. Never place secrets in images, logs, or committed environment
  files.
- Provide a local/CI test environment that can start the pinned Embedding API,
  Core, and real Qdrant deterministically.

## Observability outcome

Instrument and correlate Core, Embedding API, and Qdrant using W3C trace
context and OpenTelemetry-compatible telemetry. Provide a Compose observability
profile, dashboards, and alerts covering:

- request rate, errors, and latency by stable route;
- chunking, embedding, Qdrant search/upsert, retry, timeout, and rejection spans;
- ingestion size/chunk distributions and retrieval result counts;
- embedding inference/queue saturation;
- Qdrant health, query/upsert latency, errors, collection/vector counts, memory,
  disk pressure, and backup/restore signals;
- container CPU, memory, restarts, and storage.

Do not record raw documents, queries, vectors, secrets, authorization headers,
or unbounded identifiers in logs or metric labels. Liveness is process-only;
readiness uses cheap, bounded compatibility checks and does not scan collections.

## Mandatory verification

Run every applicable test category defined in `agents/AGENTS.md` in a clean,
documented environment. Preserve commands, versions, results, skipped checks,
and relevant failure output in the verification record. A skipped check is a
failure unless Architect approved why it is inapplicable before the final gate.

For any product or test failure, send Developer a follow-up containing:

- exact command and environment/profile;
- failing test/check and concise unedited failure evidence;
- whether the failure is deterministic;
- expected invariant and affected gate;
- request to reproduce, fix, run affected suites, and return evidence.

Do not patch application code, relax assertions, mark tests optional, or waive
failures. After Developer reports a fix, independently rerun the original check
and then the complete gate set. Repeat until green. Escalate the same persistent
failure to Architect after three evidence-backed repair attempts.

DevOps may directly fix defects isolated to CI, container orchestration,
telemetry infrastructure, or environment provisioning. Inform Developer of the
changed execution conditions and rerun the entire gate set afterward.

The final verification record must state that all required gates passed and
must identify the tested revision/configuration. Without that record, delivery
is incomplete.
