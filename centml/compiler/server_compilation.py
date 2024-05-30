from enum import Enum
from typing import List
import torch
from torch.fx import GraphModule
from hidet.graph.frontend import from_torch
from hidet.graph.frontend.torch.interpreter import Interpreter
from hidet.graph.frontend.torch.dynamo_backends import (
    get_flow_graph,
    get_compiled_graph,
    preprocess_inputs,
    HidetCompiledModel,
)


class CompilerType(Enum):
    HIDET = "hidet"


class BaseRCReturn:
    def __init__(self, compiler_type: CompilerType):
        self.compiler_type = compiler_type

    # Implement in child class
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class HidetRCReturn(BaseRCReturn):
    def __init__(self, hidet_compiled_model):
        super().__init__(CompilerType.HIDET)
        self.compiled_model_forward = hidet_compiled_model

    def __call__(self, *args, **kwargs):
        return self.compiled_model_forward(*args)


def hidet_backend_server(input_graph_module: GraphModule, example_inputs: List[torch.Tensor]):
    assert isinstance(input_graph_module, GraphModule)

    # Create hidet compiled graph
    interpreter: Interpreter = from_torch(input_graph_module)
    flow_graph, _, output_format = get_flow_graph(interpreter, example_inputs)
    cgraph = get_compiled_graph(flow_graph)

    # Perform inference using example inputs to get dispatch table
    hidet_inputs = preprocess_inputs(example_inputs)
    cgraph.run_async(hidet_inputs)

    # Get compiled forward function
    compiled_forward_function = HidetCompiledModel(cgraph, hidet_inputs, output_format)
    return HidetRCReturn(compiled_forward_function)
