import os
import pickle
import hashlib
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
from centml.compiler.dynamo_server import CompilationStatus, get_flow_graph, preprocess_inputs, dir_cleanup

base_path = os.getenv("CENTML_CACHE_DIR", default=os.path.expanduser("~/.cache/centml/compiler"))
os.makedirs(base_path, exist_ok=True)
server_IP = os.getenv("CENTML_SERVER_IP", default="0.0.0.0")
server_port = os.getenv("CENTML_SERVER_PORT", default="8080")
server_url = f"http://{server_IP}:{server_port}"

TIMEOUT = 10
TIMEOUT_COMPILE = 100
MAX_RETRIES = 3


class Runner:
    def __init__(self, module, inputs):
        self._module = module
        self._inputs = inputs
        self.compiled_forward_function = None
        # to be used with the non-blocking version
        self.compiled = False

        self.remote_compilation()

        # mp.set_start_method("forkserver", force=True)
        # self.child_process = mp.Process(
        #     target=self.remote_compilation, args=(self.compiled,)
        # )
        # self.child_process.start()

    @property
    def module(self):
        return self._module

    @property
    def inputs(self):
        return self._inputs

    def download_model(self, model_id):
        download_response = requests.get(url=f"{server_url}/download/{model_id}", timeout=TIMEOUT)
        if download_response.status_code != HTTPStatus.OK:
            raise Exception("Download request failed")

        # TODO: see if load_compiled_graph can be rewritten to save
        # the file object to a cgraph object without using the disk.
        download_path = os.path.join(base_path, f"cgraph_{model_id}.temp")
        with open(download_path, "wb") as f:
            f.write(download_response.content)
        cgraph = load_compiled_graph(download_path)

        # delete downloaded CompiledGraph file
        os.unlink(download_path)
        return cgraph

    def remote_compilation(self):
        flow_graph, inputs, output_format = get_flow_graph(self.module, self._inputs)
        fg_file_path = os.path.join(base_path, "pickled_flowgraph.temp")

        try:
            hidet.save_graph(flow_graph, fg_file_path)
        except Exception as e:
            raise Exception("Failed to save FlowGraph for hashing") from e

        with open(fg_file_path, "rb") as f:
            flow_graph_hash = hashlib.md5(f.read()).hexdigest()

        # delete the FlowGraph now that it's been hashed
        os.unlink(fg_file_path)

        # check if the corresponding CompiledGraph is cached on the server
        cache_response = requests.get(f"{server_url}/status/{flow_graph_hash}", timeout=TIMEOUT)

        if cache_response.status_code != HTTPStatus.OK:
            raise Exception("Status check failed")

        status = cache_response.json()["status"]

        if status == CompilationStatus.DONE.value:
            # model has already been compiled, so we just need to download
            cgraph = self.download_model(flow_graph_hash)
        elif status == CompilationStatus.COMPILING.value:
            # returning none will tell self.__call__ to return the uncompiled forward function
            return None
        elif status == CompilationStatus.NOT_FOUND.value:  # NOT_FOUND

            def compile_model():
                # Is it really needed to dump to disk, instead of just sending a pickled object directly?
                # allegedly, sending over pickled objects is dangerous. Maybe we should try to send as JSON.
                tfx_file_path = os.path.join(base_path, "pickled_tfx_graph.temp")
                with open(tfx_file_path, "wb") as f:
                    pickle.dump(self.module, f)

                example_inputs_path = os.path.join(base_path, "pickled_example_inputs.temp")
                with open(example_inputs_path, "wb") as f:
                    pickle.dump(self.inputs, f)

                with open(tfx_file_path, "rb") as tfx_f, open(example_inputs_path, "rb") as ei_f:
                    compile_response = requests.post(
                        url=f"{server_url}/compile_model/",
                        data={"model_id": flow_graph_hash},
                        files={"serialized_model": tfx_f, "serialized_example_inputs": ei_f},
                        timeout=TIMEOUT_COMPILE,
                    )

                # delete the tfx_graph and example_inputs now that they have been sent over
                os.unlink(tfx_file_path)
                os.unlink(example_inputs_path)

                return compile_response

            compiled_response = compile_model()
            failed_tries = 0

            while True:
                status_response = requests.get(f"{server_url}/status/{flow_graph_hash}", timeout=TIMEOUT)
                if status_response.status_code != HTTPStatus.OK:
                    raise Exception("Status check failed")

                status = status_response.json()["status"]
                if status == CompilationStatus.DONE.value:
                    break
                if status == CompilationStatus.COMPILING.value:
                    continue
                if status == CompilationStatus.NOT_FOUND.value or compiled_response.status_code != HTTPStatus.OK:
                    # If model isn't compiling after requesting compilation, retry up to 3 times
                    dir_cleanup(flow_graph_hash)
                    compiled_response = compile_model()
                    if failed_tries > MAX_RETRIES:
                        raise Exception("Compilation failed too many times")
                    failed_tries += 1

            cgraph = self.download_model(flow_graph_hash)

        else:
            raise Exception("Invalid status returned from server")

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
