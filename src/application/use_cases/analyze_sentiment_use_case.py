from src.application.ports.sentiment_metrics_port import SentimentMetricsPort
from src.domain.repositories.sentiment_analyzer import SentimentAnalyzer
from src.domain.value_objects.sentiment_result import SentimentResult
import time

class AnalyzeSentimentUseCase:
    def __init__(
        self, 
        analyzer: SentimentAnalyzer,
        metrics: SentimentMetricsPort,
    ):
        self.analyzer = analyzer
        self.metrics = metrics
        
    def execute(self, sentiment: str) -> SentimentResult:
        start_time = time.time()
            
        try:
            result = self.analyzer.analyze(sentiment)
            self.metrics.increment_request(result.label)
            return result

        except Exception as error:
            self.metrics.increment_error()
            raise error
        
        finally:
            duration = (time.time() - start_time) * 1000
            self.metrics.observe_processing_time(duration)