import os
import pickle
import hashlib
import time
import tempfile
from http import HTTPStatus
from typing import List, Sequence
import requests
import hidet
import torch
from hidet import Tensor
from hidet.ir.type import DataType
from hidet.ir.expr import SymbolVar
from hidet.graph.frontend.torch.utils import deserialize_output
from hidet.runtime.compiled_graph import load_compiled_graph
from centml.compiler import config_instance
from centml.compiler.server_compilation import CompilationStatus, get_flow_graph, preprocess_inputs, dir_cleanup

base_path = os.path.join(config_instance.CACHE_PATH, "compiler")
os.makedirs(base_path, exist_ok=True)
server_url = f"http://{config_instance.SERVER_IP}:{config_instance.SERVER_PORT}"


class Runner:
    def __init__(self, module, inputs):
        self._module = module
        self._inputs = inputs
        self.compiled_forward_function = None
        # to be used with the non-blocking version
        self.compiled = False

        self.remote_compilation()

    @property
    def module(self):
        return self._module

    @property
    def inputs(self):
        return self._inputs

    def __get_model_id(self, flow_graph):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            try:
                hidet.save_graph(flow_graph, temp_file.name)
            except Exception as e:
                raise Exception("Failed to save FlowGraph for hashing") from e

            with open(temp_file.name, "rb") as f:
                flow_graph_hash = hashlib.md5(f.read()).hexdigest()

        return flow_graph_hash

    def __download_model(self, model_id):
        download_response = requests.get(url=f"{server_url}/download/{model_id}", timeout=config_instance.TIMEOUT)
        if download_response.status_code != HTTPStatus.OK:
            raise Exception("Download request failed")

        download_path = os.path.join(base_path, f"cgraph_{model_id}.temp")
        with open(download_path, "wb") as f:
            f.write(download_response.content)
        cgraph = load_compiled_graph(download_path)

        return cgraph

    def __compile_model(self, model_id):
        compile_response = requests.post(
            url=f"{server_url}/compile_model/{model_id}",
            files={"model": pickle.dumps(self.module), "inputs": pickle.dumps(self.inputs)},
            timeout=config_instance.TIMEOUT_COMPILE,
        )
        return compile_response

    def __wait_for_status(self, model_id):
        failed_tries = 0
        while True:
            status_response = requests.get(f"{server_url}/status/{model_id}", timeout=config_instance.TIMEOUT)
            if status_response.status_code != HTTPStatus.OK:
                raise Exception("Status check failed")
            status = status_response.json()["status"]
            if status == CompilationStatus.DONE.value:
                break
            if status == CompilationStatus.COMPILING.value:
                time.sleep(1)
                continue
            if status == CompilationStatus.NOT_FOUND.value or compiled_response.status_code != HTTPStatus.OK:
                # If model isn't compiling after requesting compilation, retry up to 3 times
                dir_cleanup(model_id)
                compiled_response = self.__compile_model(model_id)
                failed_tries += 1
                if failed_tries > config_instance.MAX_RETRIES:
                    raise Exception("Compilation failed too many times")

    def remote_compilation(self):
        flow_graph, inputs, output_format = get_flow_graph(self.module, self.inputs)

        model_id = self.__get_model_id(flow_graph)

        # check if the corresponding CompiledGraph is saved locally
        cgraph_path = os.path.join(base_path, f"cgraph_{model_id}.temp")
        if os.path.isfile(cgraph_path):
            cgraph = load_compiled_graph(cgraph_path)
        else:
            # check if the corresponding CompiledGraph is cached on the server
            cache_response = requests.get(f"{server_url}/status/{model_id}", timeout=config_instance.TIMEOUT)
            if cache_response.status_code != HTTPStatus.OK:
                raise Exception("Status check failed")

            status = cache_response.json()["status"]
            if status != CompilationStatus.DONE.value:
                self.__wait_for_status(model_id)

            cgraph = self.__download_model(model_id)

        def run_executor(*inputs: torch.Tensor):
            hidet_inputs = preprocess_inputs(inputs)
            hidet_outputs: List[hidet.Tensor] = cgraph.run_async(hidet_inputs)
            torch_outputs: List[torch.Tensor] = [tensor.torch() for tensor in hidet_outputs]
            return torch_outputs

        def wrapper(*args: Tensor):
            tensor_args = []
            for param, arg in zip(inputs, args):
                if isinstance(param, Tensor):
                    tensor_args.append(arg)
                elif isinstance(param, SymbolVar):
                    dtype = param.type
                    assert isinstance(dtype, DataType)
                    if dtype.name == "int32":
                        from hidet.ffi import runtime_api

                        runtime_api.set_symbol_value(param.name, int(arg))
                    else:
                        raise ValueError(
                            f"hidet_backend: unsupported symbolic dtype {dtype}. We only support int32 now."
                        )
                else:
                    # ignore constant
                    pass
            outputs: Sequence[torch.Tensor] = run_executor(*tensor_args)
            ret = deserialize_output(output_format, outputs)
            return ret

        self.compiled_forward_function = wrapper
        self.compiled = True

    def __call__(self, *args, **kwargs):
        # If model is currently compiling, return the uncompiled forward function
        if not self.compiled_forward_function:
            return self.module(*args, **kwargs)

        forward_function = self.compiled_forward_function(*args)
        return forward_function


def centml_dynamo_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    return Runner(gm, example_inputs)
