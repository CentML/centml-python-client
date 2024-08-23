import time

from prometheus_client import Gauge, start_http_server

from centml.compiler.config import settings

_gauge = None


def get_gauge():
    global _gauge
    if _gauge is None:
        _gauge = GaugeMetric()
    return _gauge


class GaugeMetric:
    def __init__(self):
        start_http_server(settings.CENTML_PROMETHEUS_PORT)
        self._gauge = Gauge('execution_time_microseconds', 'Kernel execution times by GPU', ['gpu', 'timestamp'])
        self._values = {}

    def increment(self, gpu_name, value):
        if gpu_name not in self._values:
            self._values[gpu_name] = 0
        self._values[gpu_name] += value

    def set_metric_value(self, gpu_name):
        self._gauge.labels(gpu=gpu_name, timestamp=time.time()).set(self._values[gpu_name])
        self._values[gpu_name] = 0
