import time
from typing import List

import torch
from prometheus_client import Gauge, start_http_server
from torch._subclasses.fake_tensor import FakeTensorMode

from centml.compiler.config import settings
from centml.compiler.prediction.profiler import Profiler


class GaugeMetric:
    def __init__(self):
        start_http_server(8000)
        self._gauge = Gauge('execution_time_microseconds', 'Kernel execution times by GPU', ['gpu', 'time_stamp'])
        self._values = {}

    def increment(self, gpu_name, value):
        if gpu_name not in self._values:
            self._values[gpu_name] = 0
        self._values[gpu_name] += value

    def set_metric_value(self, gpu_name):
        self._gauge.labels(gpu=gpu_name, time_stamp=time.time()).set(self._values[gpu_name])
        self._values[gpu_name] = 0


gauge = GaugeMetric()


def centml_prediction_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    def forward(*args):
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        fake_args = [fake_mode.from_tensor(arg) if isinstance(arg, torch.Tensor) else arg for arg in args]
        with fake_mode:
            for gpu in settings.PREDICTION_GPUS.split(','):
                profiler = Profiler(gm, gpu)
                out, t = profiler.propagate(*fake_args)
                gauge.increment(gpu, t)
        return out

    return forward
