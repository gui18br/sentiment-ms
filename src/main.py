import os
import multiprocessing
from contextlib import asynccontextmanager

os.environ.setdefault("PROMETHEUS_MULTIPROC_DIR", "/tmp/prometheus")
os.makedirs(os.environ["PROMETHEUS_MULTIPROC_DIR"], exist_ok=True)

from fastapi import FastAPI
from src.presentation.routes.sentiment_routes import router
from src.presentation.routes.metrics_router import routerAPI
from src.infrastructure.metrics.memory_collector import MemoryCollector


@asynccontextmanager
async def lifespan(app: FastAPI):
    collector = MemoryCollector(interval_seconds=15.0)
    collector.start()  # inicia no worker que subiu
    yield
    collector.stop()


app = FastAPI(
    title="Sentiment Analysis MS",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
app.include_router(routerAPI)