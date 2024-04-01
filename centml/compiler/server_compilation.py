import os
import shutil
import logging
from typing import List
import torch
import pickle
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


# Calling the torch.fx.GraphModule will call RootModule.forward
# Due to limitations on how the GraphModule is constructed, args are passed as a tuple and later unpacked
class RootModule(torch.nn.Module):
    def __init__(self, callable):
        super().__init__()
        self.leaf_module = callable

    def forward(self, args):
        return self.leaf_module(args)


# We wrap the callable in a torch.nn.Module so that it can be a leaf module in the graph
# Leaf modules avoid being traced by the torch.fx.Tracer and are treated like black boxes
class ModuleWrapper(torch.nn.Module):
    def __init__(self, callable):
        super().__init__()
        self.callable = callable

    def forward(self, args):
        return self.callable(*args)


# Note: pickling a GraphModule won't save its Graph.
# Instead, it stores the Tracer and re-traces to recreate the Graph when deserializing.
class CustomTracer(torch.fx.Tracer):
    # Don't trace the ModuleWrapper
    def is_leaf_module(self, m, module_qualified_name):
        if isinstance(m, ModuleWrapper):
            return True
        return super().is_leaf_module(m, module_qualified_name)


# Create a torch.fx.GraphModule that wraps around `callable`.
# `callable` is a class whose forward function gets called when we call the GraphModule's forward function
def get_graph_module(callable):
    module = ModuleWrapper(callable)
    root = RootModule(module)
    tracer = CustomTracer()
    graph = tracer.trace(root)
    return GraphModule(root, graph)


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
    compiled_graph_module = get_graph_module(wrapper)

    try:
        with open(os.path.join(storage_path, model_id, "graph_module.zip"), "wb") as f:
            pickle.dump(compiled_graph_module, f)
    except Exception as e:
        raise Exception("Saving graph module failed") from e
