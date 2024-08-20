import builtins
import time
from typing import Callable, Dict, List, Optional, Union

import torch
from prometheus_client import Gauge, start_http_server
from torch._subclasses.fake_tensor import FakeTensorMode

from centml.compiler.backend import centml_dynamo_backend
from centml.compiler.config import OperationMode, settings
from centml.compiler.profiler import Profiler

start_http_server(8000)


class Gauge_Metric:
    def __init__(self):
        self._gauge = Gauge('execution_time_microseconds', 'Kernel execution times by GPU', ['gpu', 'time_stamp'])
        self._values = {}

    def increment(self, gpu_name, value):
        if gpu_name not in self._values:
            self._values[gpu_name] = 0
        self._values[gpu_name] += value

    def setMetricValue(self, gpu_name):
        self._gauge.labels(gpu=gpu_name, time_stamp=time.time()).set(self._values[gpu_name])
        self._values[gpu_name] = 0


def compile(
    model: Optional[Callable] = None,
    *,
    fullgraph: builtins.bool = False,
    dynamic: Optional[builtins.bool] = None,
    mode: Union[str, None] = None,
    options: Optional[Dict[str, Union[str, builtins.int, builtins.bool]]] = None,
    disable: builtins.bool = False,
) -> Callable:

    gauge = Gauge_Metric()

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

    if settings.MODE == OperationMode.REMOTE_COMPILATION:
        # Return the remote-compiled model as normal
        compiled_model = torch.compile(
            model,
            backend=centml_dynamo_backend,
            fullgraph=fullgraph,
            dynamic=dynamic,
            mode=mode,
            options=options,
            disable=disable,
        )
        return compiled_model
    else:
        # Proceed with prediction workflow
        compiled_model = torch.compile(
            model,
            backend=centml_prediction_backend,
            fullgraph=fullgraph,
            dynamic=dynamic,
            mode=mode,
            options=options,
            disable=disable,
        )

        def centml_wrapper(*args, **kwargs):
            out = compiled_model(*args, **kwargs)

            # Update the prometheus metrics with final values
            for gpu in settings.PREDICTION_GPUS.split(','):
                gauge.setMetricValue(gpu)

            # TODO: Do something with metrics

            return out

        return centml_wrapper
