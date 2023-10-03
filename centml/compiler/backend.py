from typing import List

import torch

class Runner:
    def __init__(self, module, inputs):
        self._module = module
        self._inputs = inputs

    @property
    def module(self):
        return self._module

    def __call__(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)

def centml_dynamo_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    return Runner(gm, example_inputs)
