import os
import pickle
import shutil
import logging
from enum import Enum
from typing import List, Callable
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
from centml.compiler.config import config_instance

storage_path = os.path.join(config_instance.CACHE_PATH, "server")
os.makedirs(storage_path, exist_ok=True)

logger = logging.getLogger(__name__)


# This function will delete the storage_path/{model_id} directory
def dir_cleanup(model_id: str):
    dir_path = os.path.join(storage_path, model_id)
    if not os.path.exists(dir_path):
        return  # Directory does not exist, return

    if not os.path.isdir(dir_path):
        raise Exception(f"'{dir_path}' is not a directory")

    try:
        shutil.rmtree(dir_path)
    except Exception as e:
        raise Exception("Failed to delete the directory") from e


class CompilerType(Enum):
    HIDET = "hidet"


class RootRCModule(Callable):
    def __init__(self, compiler_name: CompilerType):
        self.compiler_name = compiler_name

    # Implement in child class
    def __call__(self, *args, **kwargs):
        pass


class HidetRCModule(RootRCModule):
    def __init__(self, hidet_compiled_model):
        super().__init__(CompilerType.HIDET)
        self.compiled_model_forward = hidet_compiled_model

    def __call__(self, *args, **kwargs):
        return self.compiled_model_forward(*args)


def hidet_backend_server(input_graph_module: GraphModule, example_inputs: List[torch.Tensor], model_id: str):
    assert isinstance(input_graph_module, GraphModule)

    # Create hidet compiled graph
    interpreter: Interpreter = from_torch(input_graph_module)
    flow_graph, _, output_format = get_flow_graph(interpreter, example_inputs)
    cgraph = get_compiled_graph(flow_graph)

    # Perform inference using example inputs to get dispatch table
    hidet_inputs = preprocess_inputs(example_inputs)
    cgraph.run_async(hidet_inputs)

    # Get compiled forward function
    wrapper = HidetCompiledModel(cgraph, hidet_inputs, output_format)

    # Wrap the forward function in a torch.fx.GraphModule
    compiled_graph_module = HidetRCModule(wrapper)

    try:
        with open(os.path.join(storage_path, model_id, "graph_module.zip"), "wb") as f:
            pickle.dump(compiled_graph_module, f)
    except Exception as e:
        raise Exception("Saving graph module failed") from e
