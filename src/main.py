import os
import multiprocessing

os.environ.setdefault("PROMETHEUS_MULTIPROC_DIR", "/tmp/prometheus")
os.makedirs(os.environ["PROMETHEUS_MULTIPROC_DIR"], exist_ok=True)

from fastapi import FastAPI
from src.presentation.routes.sentiment_routes import router
from src.presentation.routes.metrics_router import routerAPI

app = FastAPI(
    title="Sentiment Analysis MS",
    version="1.0.0"
)

app.include_router(router)
app.include_router(routerAPI)