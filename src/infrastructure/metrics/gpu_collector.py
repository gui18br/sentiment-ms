import os
import threading

import torch
from prometheus_client import Gauge, REGISTRY
from prometheus_client.metrics import MetricWrapperBase


def _get_or_create(metric_type, name, *args, **kwargs) -> MetricWrapperBase:
    collector = REGISTRY._names_to_collectors.get(name)
    if collector:
        return collector
    return metric_type(name, *args, **kwargs)


GPU_VRAM_USED = _get_or_create(
    Gauge,
    "sentiment_ms_gpu_vram_used_bytes",
    "VRAM usada pelo worker na GPU (bytes)",
    ["device", "pid"],
    multiprocess_mode="liveall",
)

GPU_VRAM_TOTAL = _get_or_create(
    Gauge,
    "sentiment_ms_gpu_vram_total_bytes",
    "VRAM total disponível na GPU (bytes)",
    ["device", "pid"],
    multiprocess_mode="liveall",
)

GPU_UTILIZATION = _get_or_create(
    Gauge,
    "sentiment_ms_gpu_utilization_percent",
    "Utilização de VRAM do worker em % (proxy de utilização da GPU)",
    ["device", "pid"],
    multiprocess_mode="liveall",
)


class GpuCollector:
    """
    Coleta métricas de GPU periodicamente e as expõe via Prometheus.
    Cada worker reporta sua própria alocação de VRAM, identificado pelo PID.
    Para obter o total real da GPU, some todos os workers no dashboard:
        sum(sentiment_ms_gpu_vram_used_bytes)
    Se CUDA não estiver disponível, não faz nada.
    """

    def __init__(self, interval_seconds: float = 15.0):
        self._interval = interval_seconds
        self._timer: threading.Timer | None = None
        self._available = torch.cuda.is_available()
        self._pid = str(os.getpid())
        if self._available:
            self._device_name = torch.cuda.get_device_name(0)
        
    def start(self):
        if not self._available:
            return
        self._collect()

    def _collect(self):
        try:
            vram_used = torch.cuda.memory_allocated(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory

            GPU_VRAM_USED.labels(device=self._device_name, pid=self._pid).set(vram_used)
            GPU_VRAM_TOTAL.labels(device=self._device_name, pid=self._pid).set(vram_total)

            utilization_proxy = (vram_used / vram_total * 100) if vram_total > 0 else 0
            GPU_UTILIZATION.labels(device=self._device_name, pid=self._pid).set(utilization_proxy)

        finally:
            self._timer = threading.Timer(self._interval, self._collect)
            self._timer.daemon = True
            self._timer.start()

    def stop(self):
        if self._timer:
            self._timer.cancel()
