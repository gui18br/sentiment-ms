from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.container import build_analyze_sentiment_use_case
from src.application.use_cases.analyze_sentiment_use_case import AnalyzeSentimentUseCase

router = APIRouter(prefix="/sentiment", tags=["sentiment"])

use_case = build_analyze_sentiment_use_case()

class SentimentRequest(BaseModel):
    text: str

    class Config:
        str_strip_whitespace = True
    
@router.post("/")
def analyze_sentiment(request: SentimentRequest):
    sentiment = request.text
    result = use_case.execute(sentiment)
    return {"label": result.label, "score": result.score}