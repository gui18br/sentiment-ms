from prometheus_client import Counter, Histogram
from src.application.ports.sentiment_metrics_port import SentimentMetricsPort
from src.domain.enums.sentiment_label import SentimentLabel


REQUESTS_TOTAL = Counter(
    "sentiment_ms_requests_total",
    "Total de análises de sentimento",
    ["result", "service"],
)

PROCESSING_DURATION = Histogram(
    "sentiment_ms_processing_duration_seconds",
    "Tempo de processamento do módulo sentiment",
    ["service"],
    buckets=[0.1, 0.3, 0.5, 1, 2, 3, 5],
)

CPU_USAGE = Histogram(
    "sentiment_ms_cpu_usage_seconds",
    "CPU usada por análise de sentimento",
    ["service"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1],
)

HEAP_USAGE = Histogram(
    "sentiment_ms_heap_used_bytes",
    "Heap consumido por análise",
    ["service"],
    buckets=[1024, 10000, 50000, 100000, 500000],
)

RSS_USAGE = Histogram(
    "sentiment_ms_rss_used_bytes",
    "RSS delta por análise",
    ["service"],
)


class PrometheusSentimentMetrics(SentimentMetricsPort):

    def __init__(self, service_name: str = "sentiment-ms"):
        self.service_name = service_name

    def increment_request(self, result: SentimentLabel) -> None:
        REQUESTS_TOTAL.labels(
            result=result.value,
            service=self.service_name
        ).inc()

    def increment_error(self) -> None:
        REQUESTS_TOTAL.labels(
            result="error",
            service=self.service_name
        ).inc()

    def observe_processing_time(self, duration_ms: float) -> None:
        PROCESSING_DURATION.labels(
            service=self.service_name
        ).observe(duration_ms / 1000)

    def observe_heap_usage(self, bytes_used: int) -> None:
        HEAP_USAGE.labels(
            service=self.service_name
        ).observe(max(bytes_used, 0))

    def observe_rss_usage(self, bytes_used: int) -> None:
        RSS_USAGE.labels(
            service=self.service_name
        ).observe(max(bytes_used, 0))

    def observe_cpu_usage(self, seconds: float) -> None:
        CPU_USAGE.labels(
            service=self.service_name
        ).observe(max(seconds, 0))