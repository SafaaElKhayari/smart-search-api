import logging

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from app.config import settings

tracer = trace.get_tracer("smart-search-api")
meter = metrics.get_meter("smart-search-api")

# Product KPIs: what people search, how fast, and how often they get nothing back.
search_requests = meter.create_counter(
    "search.requests", description="Search requests by type and category"
)
zero_result_searches = meter.create_counter(
    "search.zero_results", description="Searches that returned no results"
)
search_results = meter.create_histogram(
    "search.results", unit="{result}", description="Number of results returned per search"
)
embedding_duration = meter.create_histogram(
    "search.embedding.duration", unit="ms", description="Time spent encoding the query"
)


def setup_telemetry(app) -> None:
    """Send traces and metrics to the OpenTelemetry Collector over OTLP."""
    if not settings.otel_enabled:
        return

    resource = Resource.create({SERVICE_NAME: settings.otel_service_name})

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_endpoint, insecure=True))
    )
    trace.set_tracer_provider(tracer_provider)

    reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=settings.otel_endpoint, insecure=True),
        export_interval_millis=10_000,
    )
    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[reader]))

    # Adds trace_id / span_id to every log line, so logs and traces can be joined.
    LoggingInstrumentor().instrument(set_logging_format=True, log_level=logging.INFO)
    # Elasticsearch spans come from the client itself (built in since elasticsearch-py 8.13).
    FastAPIInstrumentor.instrument_app(app, excluded_urls="health")


def record_search(search_type: str, category: str | None, result_count: int) -> None:
    attributes = {"search.type": search_type, "search.category": category or "all"}
    search_requests.add(1, attributes)
    search_results.record(result_count, attributes)
    if result_count == 0:
        zero_result_searches.add(1, attributes)
    trace.get_current_span().set_attributes({**attributes, "search.result_count": result_count})
