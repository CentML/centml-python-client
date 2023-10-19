import os
import shutil
import logging
from typing import List, Callable, Sequence, Union
from enum import Enum
import hidet
import torch
from hidet import Tensor
from hidet.ir.expr import SymbolVar
from hidet.graph.flow_graph import FlowGraph
from hidet.graph.transforms import PassContext, optimize
from hidet.runtime import CompiledGraph
from hidet.ir import dtypes
from hidet.graph.frontend.torch.utils import serialize_output, resolve_save_dir_multigraph, symbol_like_torch
from hidet.graph.frontend.torch.dynamo_config import dynamo_config
from hidet.runtime.compiled_graph import save_compiled_graph
from hidet.graph.frontend.torch.interpreter import Interpreter


class CompilationStatus(Enum):
    NOT_FOUND = 1
    COMPILING = 2
    DONE = 3


storage_path = os.getenv("CENTML_SERVER_CACHE_DIR", default=os.path.expanduser("~/.cache/centml-server"))
os.makedirs(storage_path, exist_ok=True)

logger = logging.getLogger(__name__)


# This function will delete the compiler/pickled_objects_server/{model_id} directory
def dir_cleanup(model_id):
    dir_path = os.path.join(storage_path, model_id)
    if not os.path.exists(dir_path):
        return  # Directory does not exist, return

    if not os.path.isdir(dir_path):
        raise Exception(f"'{dir_path}' is not a directory")

    try:
        shutil.rmtree(dir_path)
    except Exception as e:
        raise Exception("Failed to delete the directory") from e


def preprocess_inputs(inputs: Sequence[torch.Tensor]) -> List[hidet.Tensor]:
    torch_inputs: List[torch.Tensor] = []
    for x in inputs:
        if not x.is_contiguous():
            # warnings.warn_once('Hidet received a non-contiguous torch input tensor, converting it to contiguous')
            x = x.contiguous()
        torch_inputs.append(x)
    hidet_inputs: List[hidet.Tensor] = [hidet.from_torch(tensor) for tensor in torch_inputs]
    return hidet_inputs


# taken from hidet.graph.frontend.torch.generate_executor
def generate_executor(flow_graph: FlowGraph, model_id: str) -> Callable:
    use_fp16 = dynamo_config["use_fp16"]
    use_fp16_reduction = dynamo_config["use_fp16_reduction"]
    use_attention = dynamo_config["use_attention"]
    search_space = dynamo_config["search_space"]
    parallel_k = dynamo_config["parallel_k"]
    tensor_core = dynamo_config["use_tensor_core"]
    save_dir = dynamo_config["dump_graph_ir"]

    with PassContext() as ctx:
        if use_fp16:
            ctx.set_precision("float16")
        if use_fp16 and use_fp16_reduction:
            ctx.set_reduce_precision("float16")
        ctx.set_use_attention(use_attention)
        if save_dir:
            graph_dir = resolve_save_dir_multigraph(save_dir)
            ctx.save_graph_instrument(graph_dir)
        if tensor_core:
            ctx.set_mma("mma" if tensor_core else "simt")
        ctx.set_parallel_k(disabled=parallel_k == "disabled", search=parallel_k == "search")
        logger.info("start to optimize the flow graph")
        graph_opt: FlowGraph = optimize(flow_graph)
        logger.info("finish optimizing the flow graph")

    logger.info("schedule search space: %d", search_space)
    logger.info("start to build the optimized computation graph")
    cgraph: CompiledGraph = graph_opt.build(space=search_space)
    logger.info("finish building computation graph")

    try:
        save_compiled_graph(cgraph, os.path.join(storage_path, model_id, "cgraph.pkl"))
    except Exception as e:
        raise Exception("Saving compiled graph failed") from e


# taken from hidet.graph.frontend.torch.hidet_backend
def get_flow_graph(tfx_graph, example_inputs) -> FlowGraph:
    interpreter: Interpreter = hidet.frontend.from_torch(tfx_graph)
    inputs: List[Union[Tensor, SymbolVar, int, bool, float]] = []  # for flow graph construction
    for example_input in example_inputs:
        if isinstance(example_input, torch.Tensor):
            symbolic_input = symbol_like_torch(example_input)
            inputs.append(symbolic_input)
        elif isinstance(example_input, (int, bool, float)):
            inputs.append(example_input)
        elif isinstance(example_input, torch.SymInt):
            from torch.fx.experimental.symbolic_shapes import SymNode

            node: SymNode = example_input.node
            try:
                inputs.append(node.pytype(example_input))
            except RuntimeError:
                # is a symbolic scalar input
                pytype2dtype = {int: dtypes.int32, float: dtypes.float32, bool: dtypes.boolean}
                inputs.append(hidet.symbol_var(name=str(example_input), dtype=pytype2dtype[node.pytype]))
        else:
            raise ValueError(f"hidet_backend: unexpected example input {example_input}, type {type(example_input)}")

    output = interpreter(*inputs)
    output_format, output_tensors = serialize_output(output)
    input_tensors = [x for x in inputs if isinstance(x, hidet.Tensor)]

    return (hidet.trace_from(output_tensors, inputs=input_tensors), inputs, output_format)


# taken from hidet.graph.frontend.torch.hidet_backend
def hidet_backend_server(graph_module, example_inputs, model_id):
    assert isinstance(graph_module, torch.fx.GraphModule)

    logger.info("received a subgraph with %d nodes to optimize", len(graph_module.graph.nodes))
    logger.debug("graph: %s", graph_module.graph)

    if dynamo_config["print_input_graph"]:
        graph_module.print_readable()
        print("---")
        graph_module.graph.print_tabular()

    flow_graph, _, _ = get_flow_graph(graph_module, example_inputs)
    generate_executor(flow_graph, model_id)
