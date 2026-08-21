# Embedding API Service — Architecture Implementation Guide

## Purpose and boundary

The Embedding API converts text to vectors. It hides whether inference comes from a local model, a GPU runtime, or a hosted provider. This is its entire product boundary.

| Owns | Does not own |
| --- | --- |
| Model loading/invocation and preprocessing | Chunking and document identity |
| Input batching and runtime resource management | Vector storage and collections |
| Stable model capabilities | Retrieval, filtering, ranking |
| Provider-error normalization | Public consumer-facing RAG behavior |

Keep it private to the service network and authenticate RAG Core. The service must never accept collection IDs or write to the vector database.

## Contract

Version the internal API: independent deployment still requires compatibility.

| Endpoint | Role |
| --- | --- |
| `POST /v1/embeddings` | Embed one or more inputs. |
| `GET /v1/models` | Advertise supported immutable capabilities. |
| `GET /health/live`, `GET /health/ready` | Process and runtime readiness. |

```json
{
"model":"bge-small-en-v1.5",
"input":[
  "first passage",
  "second passage"
  ],
"input_type":"document"
}
```

```json
{
"model":"bge-small-en-v1.5",
"dimension":384,
"vectors":[
  {
  "index":0,
  "embedding":[0.012,-0.043]
  },
  {
    "index":1,
    "embedding":[-0.008,0.091]
  }
  ],
  "usage":{
    "input_count":2,
    "input_tokens":5
  }
}
```

Real embeddings have exactly `dimension` floats. Preserve input order with `index`; return model and dimension on every success. If `input_type` changes model behavior (such as query/document prefixes), make it required and document it. Otherwise omit it.

## Internal layering

```text
transport/http    validation, authentication, serialization
application       EmbedBatch/ListModels, quotas, batching rules
domain            capabilities, result and input policies
ports             EmbeddingEngine interface
adapters/engine   local transformer, hosted provider, test double
infrastructure    runtime settings, model cache, scheduler, telemetry
```

The application layer owns this narrow port:

```text
EmbeddingEngine.capabilities() -> ModelCapability[]
EmbeddingEngine.embed(model_id, inputs, input_type?) -> EmbeddingBatch
EmbeddingEngine.warmup(model_id) -> void
```

An adapter owns tokenization, device/tensor management or provider auth, response validation, and conversion to JSON-safe floats. It must not leak a model SDK/provider response into application code.

## Flow, batching, and model lifecycle

```text
RAG Core → API: batch + model + trace context
API: authenticate, validate bytes/count/deadline
Application: enforce model and token/batch limits
Engine: tokenize and run model/provider
Application: verify count, dimensions, ordering
API → RAG Core: vectors + model metadata + usage
```

Set limits on item count, total bytes, total tokens, and queue wait. Start with one bounded batch per request. Add cross-request dynamic batching only after telemetry shows a need; it complicates latency, fairness, and cancellation.

Treat a model as a versioned capability: ID, exact revision, dimension, distance metric, max tokens, supported input types, and normalization. Pin its revision in deployment. A model replacement that changes dimension or normalization requires a new/reindexed RAG collection; never hide it behind an unchanged model ID.

Do not silently truncate oversized text. Reject it clearly, or make truncation an explicit contract with per-item metadata.

## Configuration and capacity control

| Setting | Purpose |
| --- | --- |
| `EMBEDDING_PORT` | HTTP port |
| `DEFAULT_MODEL_ID`, `ALLOWED_MODEL_IDS` | Model routing |
| `MODEL_CACHE_DIR`, `INFERENCE_DEVICE` | Local runtime |
| `MAX_BATCH_ITEMS`, `MAX_BATCH_TOKENS`, `MAX_INPUT_BYTES` | Admission control |
| `INFERENCE_TIMEOUT_MS`, `QUEUE_TIMEOUT_MS` | Latency budget |
| `PROVIDER_API_KEY` | Hosted adapter secret |
| `OTEL_EXPORTER_OTLP_ENDPOINT`, `LOG_LEVEL` | Telemetry |

Validate model configuration at startup. Fail fast if a local model cannot load or device constraints cannot be met. Use bounded inference concurrency **and** a bounded queue; without both, saturation becomes memory exhaustion. Honor cancellation and deadlines where the runtime allows it.

## Errors, health, and observability

Return `code`, safe `message`, `trace_id`, and optional details. Never expose provider stacks or raw input.

| Condition | Status | Code |
| --- | --- | --- |
| Unknown/disallowed model | 400/404 | `unsupported_model` |
| Invalid/oversized text | 422/413 | `invalid_input` / `input_too_large` |
| Queue full/quota | 429/503 | `capacity_exhausted` |
| Inference deadline | 504 | `inference_timeout` |
| Runtime/provider failure | 503 | `embedding_engine_unavailable` |
| Bad adapter output | 500 | `embedding_contract_violation` |

`/health/live` checks the process. `/health/ready` checks that the configured model is loaded/available and scheduler can accept work; it may include safe model revision/dimension status. Busy is not unready—show load in metrics and control it through admission rules.

Never log raw input or vectors. Log model/revision, input count/tokens/bytes, queue wait, inference duration, device class, response status, and trace IDs. Trace validation, queue time, inference, and remote provider calls. Measure error/latency by model, queue depth, active inference, rejections, batch distribution, load state, and device memory/utilization where available.

## Testing and deployment

| Test | Proves |
| --- | --- |
| Unit | Limits, ordering, routing, config and error mapping |
| Engine contract | One finite vector per input; expected dimension/index |
| Integration | Pinned model or provider stub serves and advertises capability |
| Performance | Bounded load meets p95 target and rejects overload safely |
| Compatibility | RAG collection fixtures match model revision/dimension |

Avoid bit-for-bit float expectations across hardware. Verify dimensions, invariants such as normalization when applicable, order, and end-to-end retrieval properties.

Use distinct CPU/GPU deployment shapes when helpful. Pre-bake or securely cache pinned artifacts; do not download a moving “latest” model on every cold start. Declare CPU/memory/GPU requests and limits. Autoscale with queue depth/concurrency as well as CPU, use readiness-aware load balancing, and keep the API private.

Upgrade by advertising a new model revision, reindexing compatible RAG collections, switching RAG Core configuration after validation, then retiring the old model.

## Anti-patterns and milestones

Review red flags:

- This service connects to Qdrant or sees collection IDs.
- Model revision or dimension changes silently.
- Batches, queues, or concurrency are unbounded.
- Text, vectors, tokens, or secrets are written to normal logs.
- Query/document modes are treated as interchangeable without model evidence.
- Responses omit model and dimension metadata.

1. Contract, typed config, health, and logs.
2. One pinned engine adapter with strict limits and contract tests.
3. RAG Core integration and trace propagation.
4. Bounded queue/concurrency, metrics, and load test.
5. Capabilities endpoint, warm-up, and model migration playbook.
