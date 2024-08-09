import builtins
from typing import Callable, Dict, List, Optional, Union

import torch
from prometheus_client import start_http_server
from torch._subclasses.fake_tensor import FakeTensorMode

import centml.compiler
from centml.compiler.backend import centml_dynamo_backend
from centml.compiler.config import settings
from centml.compiler.metrics import time_metric
from centml.compiler.profiler import Profiler

start_http_server(8000)


def compile(
    model: Optional[Callable] = None,
    *,
    fullgraph: builtins.bool = False,
    dynamic: Optional[builtins.bool] = None,
    mode: Union[str, None] = None,
    options: Optional[Dict[str, Union[str, builtins.int, builtins.bool]]] = None,
    disable: builtins.bool = False
) -> Callable:

    def centml_prediction_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):

        def forward(*args):
            fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
            fake_args = [fake_mode.from_tensor(arg) for arg in args]
            with fake_mode:
                profiler = Profiler(gm)
                out, t = profiler.propagate(*fake_args)

            # Increment the prometheus metric
            time_metric.inc(t)

            return out

        return forward

    if not settings.PREDICTING:
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
        # At this point the metric can be reset to 0
        # Need to do something with its value before resetting it
        time_metric.set(0)

        return out

    return centml_wrapper
