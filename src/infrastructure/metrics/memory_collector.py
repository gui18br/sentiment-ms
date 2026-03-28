import os
import psutil
import threading
from prometheus_client import Gauge, REGISTRY
from prometheus_client.metrics import MetricWrapperBase


def _get_or_create(metric_type, name, *args, **kwargs) -> MetricWrapperBase:
    collector = REGISTRY._names_to_collectors.get(name)
    if collector:
        return collector
    return metric_type(name, *args, **kwargs)


PROCESS_RSS = _get_or_create(
    Gauge,
    "sentiment_ms_process_rss_bytes",
    "Memória RSS atual do processo worker",
    ["pid"],
    multiprocess_mode="liveall",
)

PROCESS_VMS = _get_or_create(
    Gauge,
    "sentiment_ms_process_vms_bytes",
    "Memória virtual atual do processo worker",
    ["pid"],
    multiprocess_mode="liveall",
)

PROCESS_CPU = _get_or_create(
    Gauge,
    "sentiment_ms_process_cpu_percent",
    "CPU % do processo worker",
    ["pid"],
    multiprocess_mode="liveall",
)


class MemoryCollector:
    def __init__(self, interval_seconds: float = 15.0):
        self._process = psutil.Process(os.getpid())
        self._pid = str(os.getpid())
        self._interval = interval_seconds
        self._timer: threading.Timer | None = None

    def start(self):
        self._process.cpu_percent(interval=None)  # primeira chamada sempre retorna 0, descarta
        self._collect()

    def _collect(self):
        try:
            mem = self._process.memory_info()
            PROCESS_RSS.labels(pid=self._pid).set(mem.rss)
            PROCESS_VMS.labels(pid=self._pid).set(mem.vms)

            cpu = self._process.cpu_percent(interval=None)
            PROCESS_CPU.labels(pid=self._pid).set(cpu)
        finally:
            self._timer = threading.Timer(self._interval, self._collect)
            self._timer.daemon = True
            self._timer.start()

    def stop(self):
        if self._timer:
            self._timer.cancel()