from abc import ABC, abstractmethod
from ..value_objects.sentiment_result import SentimentResult

class SentimentAnalyzer(ABC):
    
    @abstractmethod
    def analyze(self, sentimet: str) -> SentimentResult:
        pass