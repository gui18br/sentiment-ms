from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.container import build_analyze_sentiment_use_case
from src.application.use_cases.analyze_sentiment_use_case import AnalyzeSentimentUseCase
from src.domain.entities.sentiment import Sentiment

router = APIRouter(prefix="/sentiment", tags=["sentiment"])

class SentimentRequest(BaseModel):
    text: str
    
def get_use_case():
    return build_analyze_sentiment_use_case()

@router.post("/")
def analyze_sentiment(
    request: SentimentRequest,
    use_case: AnalyzeSentimentUseCase = Depends(get_use_case)
):
    sentiment = Sentiment(text=request.text)
    result = use_case.execute(sentiment)
    return {"label": result.label, "score": result.score}