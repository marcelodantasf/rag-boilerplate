"""Optional OTLP traces and metrics for Core request/dependency correlation."""

from dataclasses import dataclass

from fastapi import FastAPI
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.metrics import Counter, Histogram
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from rag_core.infrastructure.settings import Settings


@dataclass(slots=True)
class Telemetry:
    request_count: Counter
    request_duration: Histogram
    tracer_provider: TracerProvider
    meter_provider: MeterProvider

    def shutdown(self) -> None:
        self.tracer_provider.force_flush(timeout_millis=2_000)
        self.meter_provider.force_flush(timeout_millis=2_000)
        self.tracer_provider.shutdown()
        self.meter_provider.shutdown()


def configure_telemetry(app: FastAPI, settings: Settings) -> Telemetry | None:
    endpoint = settings.otel_exporter_otlp_endpoint
    if endpoint is None:
        return None
    resource = Resource.create({"service.name": settings.otel_service_name, "service.version": "0.1.0"})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")))
    trace.set_tracer_provider(tracer_provider)
    metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=f"{endpoint.rstrip('/')}/v1/metrics"), export_interval_millis=15_000)
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)
    FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider, meter_provider=meter_provider)
    meter = meter_provider.get_meter("rag_core.http")
    return Telemetry(
        request_count=meter.create_counter("rag_core_http_requests", unit="{request}"),
        request_duration=meter.create_histogram("rag_core_http_request_duration", unit="ms"),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )
