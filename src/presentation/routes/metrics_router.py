from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.multiprocess import MultiProcessCollector

routerAPI = APIRouter()

@routerAPI.get("/metrics")
def metrics():
    registry = CollectorRegistry()
    MultiProcessCollector(registry)

    return Response(
        generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST
    )