from pydantic import BaseModel

class FeedbackRequestDTO(BaseModel):
    label: str