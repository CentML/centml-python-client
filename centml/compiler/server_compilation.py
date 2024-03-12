import os
import shutil
import logging
from typing import List
import torch
from torch.fx import GraphModule, Graph
from hidet.graph.frontend import from_torch
from hidet.graph.frontend.torch.interpreter import Interpreter
from hidet.graph.frontend.torch.dynamo_backends import (
    get_flow_graph,
    get_compiled_graph,
    preprocess_inputs,
    CompiledForwardFunction,
)
from centml.compiler.config import config_instance
import pickle

storage_path = os.path.join(config_instance.CACHE_PATH, "server")
os.makedirs(storage_path, exist_ok=True)

logger = logging.getLogger(__name__)


# Create a torch.fx.GraphModule that wraps around `callable`
# graph_module(*inputs) will call callable.__call__(*inputs)
# In this function, `example_inputs`` is only used to determine the number of input nodes
def get_graph_module(callable, example_inputs):
    graph = Graph()
    input_nodes = [
        graph.create_node(op='placeholder', target=f'input_{i}', args=(), kwargs={}) for i in range(len(example_inputs))
    ]
    
    # Create a `get_attr` node that refers to `callable`
    callable_node = graph.create_node(op='get_attr', target='callable')

    # This should call "__call__" method of `callable` (the root)
    function_node = graph.create_node(op='call_method', target="__call__", args=(callable_node, *input_nodes), kwargs={})
    graph.output(function_node)
    
    # Make `callable` an attribute of a new parent module
    parent_module = torch.nn.Module()
    parent_module.callable = callable

    graph_module = GraphModule(parent_module, graph)

    return graph_module


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


def hidet_backend_server(graph_module: GraphModule, example_inputs: List[torch.Tensor], model_id: str):
    assert isinstance(graph_module, GraphModule)

    logger.info("received a subgraph with %d nodes to optimize", len(graph_module.graph.nodes))
    logger.debug("graph: %s", graph_module.graph)

    # Create hidet compiled graph
    interpreter: Interpreter = from_torch(graph_module)
    flow_graph, _, output_format = get_flow_graph(interpreter, example_inputs)
    cgraph = get_compiled_graph(flow_graph)

    # Perform inference using example inputs to get dispatch table
    hidet_inputs = preprocess_inputs(example_inputs)
    cgraph.run_async(hidet_inputs)

    # Get compiled forward function
    wrapper = CompiledForwardFunction(cgraph, example_inputs, output_format)

    # Wrap the forward function in a torch.fx.GraphModule
    graph_module = get_graph_module(wrapper, example_inputs)
    
    graph_module.graph.print_tabular()
    
    try:
        # This uses pickle to serialize to disk
        with open(os.path.join(storage_path, model_id, "graph_module.zip"), "wb") as f:
            pickle.dump(graph_module, f)
    except Exception as e:
        raise Exception("Saving graph module failed") from e
