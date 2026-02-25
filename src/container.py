from src.application.use_cases.analyze_sentiment_use_case import AnalyzeSentimentUseCase
from src.infrastructure.simple_sentiment_analyzer import SimpleSentimentAnalyzer

def build_analyze_sentiment_use_case():
    analyzer = SimpleSentimentAnalyzer()
    return AnalyzeSentimentUseCase(analyzer)