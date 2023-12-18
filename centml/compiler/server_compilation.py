import os
import shutil
import logging
from enum import Enum
from typing import List
from torch import Tensor
from torch.fx import GraphModule
from hidet.runtime.compiled_graph import save_compiled_graph
from hidet.graph.frontend import from_torch
from hidet.graph.frontend.torch.interpreter import Interpreter
from hidet.graph.frontend.torch.dynamo_backends import get_flow_graph, get_compiled_graph, preprocess_inputs
from centml.compiler.config import config_instance


class CompilationStatus(Enum):
    NOT_FOUND = 1
    COMPILING = 2
    DONE = 3


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


def hidet_backend_server(graph_module: GraphModule, example_inputs: List[Tensor], model_id: str):
    assert isinstance(graph_module, GraphModule)

    logger.info("received a subgraph with %d nodes to optimize", len(graph_module.graph.nodes))
    logger.debug("graph: %s", graph_module.graph)

    interpreter: Interpreter = from_torch(graph_module)
    flow_graph, _, _ = get_flow_graph(interpreter, example_inputs)
    cgraph = get_compiled_graph(flow_graph)

    # perform inference using example inputs to get dispatch table
    hidet_inputs = preprocess_inputs(example_inputs)
    cgraph.run_async(hidet_inputs)

    try:
        save_compiled_graph(cgraph, os.path.join(storage_path, model_id, "cgraph.zip"), save_dispatch_table=True)
    except Exception as e:
        raise Exception("Saving compiled graph failed") from e
