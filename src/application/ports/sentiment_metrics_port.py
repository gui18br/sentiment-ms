from abc import ABC, abstractmethod
from src.domain.enums.sentiment_label import SentimentLabel


class SentimentMetricsPort(ABC):

    @abstractmethod
    def increment_request(self, result: SentimentLabel) -> None:
        pass

    @abstractmethod
    def increment_error(self) -> None:
        pass

    @abstractmethod
    def observe_processing_time(self, duration_ms: float) -> None:
        pass

    @abstractmethod
    def observe_heap_usage(self, bytes_used: int) -> None:
        pass

    @abstractmethod
    def observe_rss_usage(self, bytes_used: int) -> None:
        pass

    @abstractmethod
    def observe_cpu_usage(self, seconds: float) -> None:
        pass