from pydantic import BaseModel

class FeedbackRequestDTO(BaseModel):
    feedback_id: str
    text: str