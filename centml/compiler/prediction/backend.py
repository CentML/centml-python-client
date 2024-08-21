import time
from typing import List

import torch
from prometheus_client import Gauge, start_http_server
from torch._subclasses.fake_tensor import FakeTensorMode

from centml.compiler.config import settings
from centml.compiler.prediction.profiler import Profiler


class GaugeMetric:
    def __init__(self):
        start_http_server(settings.PROMETHEUS_PORT)
        self._gauge = Gauge('execution_time_microseconds', 'Kernel execution times by GPU', ['gpu', 'timestamp'])
        self._values = {}

    def increment(self, gpu_name, value):
        if gpu_name not in self._values:
            self._values[gpu_name] = 0
        self._values[gpu_name] += value

    def set_metric_value(self, gpu_name):
        self._gauge.labels(gpu=gpu_name, timestamp=time.time()).set(self._values[gpu_name])
        self._values[gpu_name] = 0


gauge = GaugeMetric()


def centml_prediction_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    profilers = []
    for gpu in settings.PREDICTION_GPUS.split(','):
        profilers.append(Profiler(gm, gpu))

    def forward(*args):
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        fake_args = [fake_mode.from_tensor(arg) if isinstance(arg, torch.Tensor) else arg for arg in args]
        with fake_mode:
            for prof in profilers:
                out, t = prof.propagate(*fake_args)
                gauge.increment(prof.gpu, t)
        return out

    return forward
