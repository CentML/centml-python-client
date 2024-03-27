import os
import shutil
import logging
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
from centml.compiler.config import config_instance

storage_path = os.path.join(config_instance.CACHE_PATH, "server")
os.makedirs(storage_path, exist_ok=True)

logger = logging.getLogger(__name__)


# Calling the torch.fx.GraphModule will call RootModule.forward
# Note that due to torch.fx.Tracer limitations, RootModule.forward can't have *args (or **kwargs) in its signature
# Therefore, args are passed as a tuple and later unpacked
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


# CustomTracer doesn't trace the callable (it treats it as a leaf module)
# However, the callable needs to be of class nn.Module for this to work
class CustomTracer(torch.fx.Tracer):
    def __init__(self, callable=None):
        self.callable_type = type(callable) if callable is not None else RootModule
        super().__init__()

    def is_leaf_module(self, m, module_qualified_name):
        if isinstance(m, self.callable_type):
            return True
        return super().is_leaf_module(m, module_qualified_name)


# Create a torch.fx.GraphModule that wraps around `callable`.
# `callable` is a class whose forward function gets called when we call the GraphModule's forward function
def get_graph_module(callable):
    module = ModuleWrapper(callable)
    root = RootModule(module)
    tracer = CustomTracer(module)
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
    wrapper = HidetCompiledModel(cgraph, hidet_inputs, output_format)

    # Wrap the forward function in a torch.fx.GraphModule
    compiled_graph_module = get_graph_module(wrapper)

    return compiled_graph_module
