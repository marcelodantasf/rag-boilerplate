"""Low-cardinality product metrics and explicit operation/dependency spans."""

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator

from opentelemetry import metrics, trace


ALLOWED_ATTRIBUTE_KEYS = frozenset(
    {"dependency", "operation", "outcome", "error_code", "phase"}
)

PRODUCT_METRIC_NAMES = frozenset(
    {
        "rag_core_ingestion_duration",
        "rag_core_documents_indexed",
        "rag_core_chunks_indexed",
        "rag_core_embedding_batch_size",
        "rag_core_retrieval_duration",
        "rag_core_retrieval_top_k",
        "rag_core_retrieval_results",
        "rag_core_retrieval_no_match",
        "rag_core_dependency_requests",
        "rag_core_dependency_duration",
        "rag_core_dependency_errors",
        "rag_core_dependency_timeouts",
        "rag_core_dependency_retries",
        "rag_core_contract_mismatches",
        "rag_core_validation_rejections",
        "rag_core_idempotency_operations",
        "rag_core_generation_operations",
        "rag_core_generation_duration",
    }
)

_meter = metrics.get_meter("rag_core.product", "0.1.0")
tracer = trace.get_tracer("rag_core.product", "0.1.0")

ingestion_duration = _meter.create_histogram("rag_core_ingestion_duration", unit="ms")
documents_indexed = _meter.create_counter("rag_core_documents_indexed", unit="{document}")
chunks_indexed = _meter.create_counter("rag_core_chunks_indexed", unit="{chunk}")
embedding_batch_size = _meter.create_histogram("rag_core_embedding_batch_size", unit="{item}")
retrieval_duration = _meter.create_histogram("rag_core_retrieval_duration", unit="ms")
retrieval_top_k = _meter.create_histogram("rag_core_retrieval_top_k", unit="{item}")
retrieval_results = _meter.create_histogram("rag_core_retrieval_results", unit="{item}")
retrieval_no_match = _meter.create_counter("rag_core_retrieval_no_match", unit="{request}")
dependency_requests = _meter.create_counter("rag_core_dependency_requests", unit="{request}")
dependency_duration = _meter.create_histogram("rag_core_dependency_duration", unit="ms")
dependency_errors = _meter.create_counter("rag_core_dependency_errors", unit="{error}")
dependency_timeouts = _meter.create_counter("rag_core_dependency_timeouts", unit="{timeout}")
dependency_retries = _meter.create_counter("rag_core_dependency_retries", unit="{retry}")
contract_mismatches = _meter.create_counter("rag_core_contract_mismatches", unit="{mismatch}")
validation_rejections = _meter.create_counter("rag_core_validation_rejections", unit="{rejection}")
idempotency_operations = _meter.create_counter("rag_core_idempotency_operations", unit="{operation}")
generation_operations = _meter.create_counter("rag_core_generation_operations", unit="{operation}")
generation_duration = _meter.create_histogram("rag_core_generation_duration", unit="ms")


def safe_attributes(**values: str) -> dict[str, str]:
    unknown = set(values).difference(ALLOWED_ATTRIBUTE_KEYS)
    if unknown:
        raise ValueError(f"unsafe telemetry attribute keys: {sorted(unknown)}")
    return values


@contextmanager
def dependency_call(dependency: str, operation: str) -> Iterator[None]:
    attributes = safe_attributes(dependency=dependency, operation=operation)
    started = perf_counter()
    with tracer.start_as_current_span(
        f"rag.{dependency}.{operation}", attributes=attributes
    ) as span:
        try:
            yield
        except Exception as error:
            code = str(getattr(error, "code", "internal_error"))
            failure = safe_attributes(
                dependency=dependency,
                operation=operation,
                outcome="error",
                error_code=code,
            )
            dependency_requests.add(1, failure)
            dependency_errors.add(1, failure)
            if code.endswith("_timeout"):
                dependency_timeouts.add(1, failure)
            span.set_attribute("rag.outcome", "error")
            span.set_attribute("error.type", code)
            raise
        else:
            success = safe_attributes(
                dependency=dependency, operation=operation, outcome="success"
            )
            dependency_requests.add(1, success)
            span.set_attribute("rag.outcome", "success")
        finally:
            dependency_duration.record(
                (perf_counter() - started) * 1_000,
                attributes,
            )


def record_retry(dependency: str, operation: str) -> None:
    dependency_retries.add(
        1,
        safe_attributes(dependency=dependency, operation=operation),
    )
