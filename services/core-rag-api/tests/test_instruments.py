import importlib

import pytest
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import rag_core.infrastructure.instruments as instrument_module


def test_product_metrics_and_child_spans_use_only_low_cardinality_attributes() -> None:
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(meter_provider)
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)
    instruments = importlib.reload(instrument_module)

    instruments.documents_indexed.add(1)
    instruments.chunks_indexed.add(2)
    instruments.retrieval_results.record(1)
    instruments.idempotency_operations.add(
        1, instruments.safe_attributes(operation="ingestDocument", phase="replay")
    )
    with instruments.dependency_call("embedding", "embed"):
        pass
    meter_provider.force_flush()

    exported_names = {
        metric.name
        for resource in reader.get_metrics_data().resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
    }
    assert {
        "rag_core_documents_indexed",
        "rag_core_chunks_indexed",
        "rag_core_retrieval_results",
        "rag_core_idempotency_operations",
        "rag_core_dependency_requests",
        "rag_core_dependency_duration",
    } <= exported_names
    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["rag.embedding.embed"]
    assert not ({"query", "content", "vector", "collection_id", "document_id"} & set(spans[0].attributes))

    for forbidden in ("query", "content", "vector", "collection_id", "document_id", "tenant_id"):
        with pytest.raises(ValueError, match="unsafe telemetry"):
            instruments.safe_attributes(**{forbidden: "sensitive"})
