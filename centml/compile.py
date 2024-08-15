import builtins
from typing import Callable, Dict, List, Optional, Union

import torch
from prometheus_client import Gauge, start_http_server
from torch._subclasses.fake_tensor import FakeTensorMode

from centml.compiler.backend import centml_dynamo_backend
from centml.compiler.config import OperationMode, settings
from centml.compiler.profiler import Profiler

start_http_server(8000)


class Metric:
    def __init__(self, gpu_name):
        self._time = 0
        self._metric = Gauge(f'{gpu_name}_metric', f'Sum of the kernel execution times on {gpu_name}')

    def increment(self, value):
        self._time += value

    def update_metric(self):
        self._metric.set(self._time)

    def reset(self):
        self._time = 0
        self._metric.set(0)

    def get(self):
        return self._time


def compile(
    model: Optional[Callable] = None,
    *,
    fullgraph: builtins.bool = False,
    dynamic: Optional[builtins.bool] = None,
    mode: Union[str, None] = None,
    options: Optional[Dict[str, Union[str, builtins.int, builtins.bool]]] = None,
    disable: builtins.bool = False,
) -> Callable:

    # Create a metric for each GPU
    GPU_METRICS = {gpu: Metric(gpu) for gpu in settings.PREDICTION_GPUS.split(',')}

    def centml_prediction_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        def forward(*args):
            fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
            fake_args = [fake_mode.from_tensor(arg) for arg in args]
            with fake_mode:
                for gpu, metric in GPU_METRICS.items():
                    profiler = Profiler(gm, gpu)
                    out, t = profiler.propagate(*fake_args)
                    metric.increment(t)
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
        # Proceed with production workflow
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
            for metric in GPU_METRICS.values():
                metric.update_metric()

            # TODO: Do something with metrics

            # Reset the metrics after the prediction
            for metric in GPU_METRICS.values():
                metric.reset()

            return out

        return centml_wrapper
