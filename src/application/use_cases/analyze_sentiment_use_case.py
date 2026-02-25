from src.domain.entities.sentiment import Sentiment
from src.domain.repositories.sentiment_analyzer import SentimentAnalyzer
from src.domain.value_objects.sentiment_result import SentimentResult

class AnalyzeSentimentUseCase:
    def __init__(self, analyzer: SentimentAnalyzer):
        self.analyzer = analyzer
        
    def execute(self, sentiment: Sentiment) -> SentimentResult:
        
        try:
            result = self.analyzer.analyze(sentiment)
            return result
        except Exception as error:
            raise error