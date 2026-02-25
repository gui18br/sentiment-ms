from dataclasses import dataclass
from ..enums.sentiment_label import SentimentLabel


@dataclass(frozen=True)
class SentimentResult:
    score: float
    label: SentimentLabel