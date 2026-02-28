from src.application.ports.sentiment_metrics_port import SentimentMetricsPort
from src.domain.entities.sentiment import Sentiment
from src.domain.repositories.sentiment_analyzer import SentimentAnalyzer
from src.domain.value_objects.sentiment_result import SentimentResult
import time
import os
import psutil

class AnalyzeSentimentUseCase:
    def __init__(
        self, 
        analyzer: SentimentAnalyzer,
        metrics: SentimentMetricsPort,
        ):
        self.analyzer = analyzer
        self.metrics = metrics
        
    def execute(self, sentiment: Sentiment) -> SentimentResult:
        start_time = time.time()
        process = psutil.Process(os.getpid())
        cpu_start = process.cpu_times().user
        mem_start = process.memory_info()
            
        try:
            result = self.analyzer.analyze(sentiment)
            self.metrics.increment_request(result.label)
            return result
        except Exception as error:
            self.metrics.increment_error()
            raise error
        
        finally:
            duration = (time.time() - start_time) * 1000
            cpu_end = process.cpu_times().user
            mem_end = process.memory_info()

            self.metrics.observe_processing_time(duration)
            self.metrics.observe_cpu_usage(cpu_end - cpu_start)
            self.metrics.observe_heap_usage(mem_end.vms - mem_start.vms)
            self.metrics.observe_rss_usage(mem_end.rss - mem_start.rss)