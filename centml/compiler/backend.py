import os
import gc
import pickle
import hashlib
import time
import tempfile
import logging
import weakref
import threading as th
from http import HTTPStatus
from typing import List, Callable
import requests
import torch
from torch.fx import GraphModule
import hidet
from hidet.graph.frontend.torch.interpreter import Interpreter
from hidet.graph.frontend.torch.dynamo_backends import get_flow_graph
from hidet.runtime.compiled_graph import CompiledGraph
from centml.compiler.config import config_instance, CompilationStatus


hidet.option.imperative(False)
base_path = os.path.join(config_instance.CACHE_PATH, "compiler")
os.makedirs(base_path, exist_ok=True)
server_url = f"http://{config_instance.SERVER_IP}:{config_instance.SERVER_PORT}"

logger = logging.getLogger(__name__)


class Runner:
    def __init__(self, module: GraphModule, inputs: List[torch.Tensor]):
        self._module: GraphModule = weakref.ref(module)
        self._inputs: List[torch.Tensor] = inputs
        self.compiled_forward_function: Callable[[torch.Tensor], tuple] = None
        self.lock = th.Lock()
        self.child_thread = th.Thread(target=self.remote_compilation)

        try:
            self.child_thread.start()
        except Exception:
            logger.exception("Remote compilation failed with the following exception: \n")

    @property
    def module(self):
        return self._module()

    @property
    def inputs(self):
        return self._inputs

    @module.deleter
    def module(self):
        self._module().graph.owning_module = None
        self._module = None

    @inputs.deleter
    def inputs(self):
        self._inputs = None

    def _get_model_id(self, flow_graph: hidet.FlowGraph) -> str:
        if not flow_graph:
            raise Exception("Getting model id: flow graph is None.")

        with tempfile.NamedTemporaryFile() as temp_file:
            try:
                hidet.save_graph(flow_graph, temp_file.name)
            except Exception as e:
                raise Exception(f"Getting model id: failed to save FlowGraph. {e}\n") from e

            with open(temp_file.name, "rb") as f:
                flow_graph_hash = hashlib.md5(f.read()).hexdigest()

        return flow_graph_hash

    def _download_model(self, model_id: str) -> CompiledGraph:
        download_response = requests.get(url=f"{server_url}/download/{model_id}", timeout=config_instance.TIMEOUT)
        if download_response.status_code != HTTPStatus.OK:
            raise Exception(
                f"Download: request failed, exception from server:\n{download_response.json().get('detail')}"
            )
        download_dir = os.path.join(base_path, model_id)
        os.makedirs(download_dir, exist_ok=True)
        download_path = os.path.join(download_dir, "graph_module.zip")
        with open(download_path, "wb") as f:
            f.write(download_response.content)
            return pickle.loads(download_response.content)

    def _compile_model(self, model_id: str):
        compile_response = requests.post(
            url=f"{server_url}/submit/{model_id}",
            files={"model": pickle.dumps(self.module), "inputs": pickle.dumps(self.inputs)},
            timeout=config_instance.TIMEOUT,
        )
        if compile_response.status_code != HTTPStatus.OK:
            raise Exception(
                f"Compile model: request failed, exception from server:\n{compile_response.json().get('detail')}\n"
            )

    def _wait_for_status(self, model_id: str) -> bool:
        tries = 0
        while True:
            # get server compilation status
            status_response = requests.get(f"{server_url}/status/{model_id}", timeout=config_instance.TIMEOUT)
            if status_response.status_code != HTTPStatus.OK:
                raise Exception(
                    f"Status check: request failed, exception from server:\n{status_response.json().get('detail')}"
                )

            status = status_response.json().get("status")

            if status == CompilationStatus.DONE.value:
                return True
            elif status == CompilationStatus.COMPILING.value:
                pass
            elif status == CompilationStatus.NOT_FOUND.value:
                tries += 1
                self._compile_model(model_id)
            else:
                tries += 1

            if tries > config_instance.MAX_RETRIES:
                raise Exception("Waiting for status: compilation failed too many times.\n")

            time.sleep(config_instance.COMPILING_SLEEP_TIME)

    def remote_compilation(self):
        # start by getting the model_id
        interpreter: Interpreter = hidet.frontend.from_torch(self.module)
        flow_graph, _, _ = get_flow_graph(interpreter, self.inputs)

        model_id = self._get_model_id(flow_graph)

        # check if graph module is saved locally
        graph_module_path = os.path.join(base_path, model_id, "graph_module.zip")
        if os.path.isfile(graph_module_path):
            with open(graph_module_path, "rb") as f:
                graph_module = pickle.load(f)
        else:
            self._wait_for_status(model_id)
            graph_module = self._download_model(model_id)

        self.compiled_forward_function = graph_module

        # Let garbage collector free the memory used by the uncompiled model
        with self.lock:
            interpreter = None
            del self.inputs
            del self.module
            gc.collect()
            torch.cuda.empty_cache()

    def __call__(self, *args, **kwargs):
        # If model is currently compiling, return the uncompiled forward function
        with self.lock:
            if not self.compiled_forward_function:
                return self.module.forward(*args, **kwargs)

        return self.compiled_forward_function(*args)


def centml_dynamo_backend(gm: GraphModule, example_inputs: List[torch.Tensor]):
    return Runner(gm, example_inputs)
