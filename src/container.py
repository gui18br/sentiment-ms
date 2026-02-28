from src.application.use_cases.analyze_sentiment_use_case import AnalyzeSentimentUseCase
from src.infrastructure.simple_sentiment_analyzer import SimpleSentimentAnalyzer
from src.infrastructure.observability.prometheus_sentiment_metrics import PrometheusSentimentMetrics

_analyzer = SimpleSentimentAnalyzer()
_metrics = PrometheusSentimentMetrics(service_name="sentiment-ms")

def build_analyze_sentiment_use_case():
    return AnalyzeSentimentUseCase(
        analyzer=_analyzer,
        metrics=_metrics
    )