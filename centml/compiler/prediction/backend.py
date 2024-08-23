from typing import List

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from centml.compiler.config import settings
from centml.compiler.prediction.kdtree import get_tree_db
from centml.compiler.prediction.metric import get_gauge
from centml.compiler.prediction.profiler import Profiler


def centml_prediction_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    profilers = []
    tree_db = get_tree_db()
    for gpu in settings.CENTML_PREDICTION_GPUS.split(','):
        profilers.append(Profiler(gm, gpu, tree_db))

    def forward(*args):
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        fake_args = [fake_mode.from_tensor(arg) if isinstance(arg, torch.Tensor) else arg for arg in args]
        with fake_mode:
            for prof in profilers:
                out, t = prof.propagate(*fake_args)
                gauge = get_gauge()
                gauge.increment(prof.gpu, t)
        return out

    return forward
