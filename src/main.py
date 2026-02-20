from fastapi import FastAPI

app = FastAPI(
        title="Sentiment Analysis MS", 
        version="1.0.0"
)

@app.get('/health')
def health():
    return {"status": "ok"}