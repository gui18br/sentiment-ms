from pydantic import BaseModel

class SentimentResponseDTO(BaseModel):
    feedback_id: str
    sentiment: str
    score: float