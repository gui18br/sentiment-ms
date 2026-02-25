from fastapi import FastAPI
from src.presentation.routes.sentiment_routes import router


app = FastAPI(
        title="Sentiment Analysis MS", 
        version="1.0.0"
)

app.include_router(router)