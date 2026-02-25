from abc import ABC, abstractmethod
from ..entities.sentiment import Sentiment
from ..value_objects.sentiment_result import SentimentResult

class SentimentAnalyzer(ABC):
    
    @abstractmethod
    def analyze(self, sentimet: Sentiment) -> SentimentResult:
        pass