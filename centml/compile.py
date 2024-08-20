import builtins
from typing import Callable, Dict, Optional, Union

import torch

from centml.compiler.backend import centml_dynamo_backend
from centml.compiler.config import OperationMode, settings
from centml.compiler.prediction.backend import centml_prediction_backend, gauge


def compile(
    model: Optional[Callable] = None,
    *,
    fullgraph: builtins.bool = False,
    dynamic: Optional[builtins.bool] = None,
    mode: Union[str, None] = None,
    options: Optional[Dict[str, Union[str, builtins.int, builtins.bool]]] = None,
    disable: builtins.bool = False,
) -> Callable:

    if settings.MODE == OperationMode.REMOTE_COMPILATION:
        # Return the remote-compiled model
        compiled_model = torch.compile(
            model,
            backend=centml_dynamo_backend,  # Compilation backend
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
            backend=centml_prediction_backend,  # Prediction backend
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
                gauge.set_metric_value(gpu)

            return out

        return centml_wrapper
