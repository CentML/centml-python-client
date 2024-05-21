import os
import gc
import json
import time
import pickle
import hashlib
import logging
import weakref
import warnings
import tempfile
import requests
import threading as th
from http import HTTPStatus
from typing import List, Callable
from torch.fx import GraphModule
import torch
from torch._dynamo.output_graph import GraphCompileReason
from hidet.graph.frontend.torch.interpreter import Interpreter
from hidet.graph.frontend.torch.dynamo_backends import get_flow_graph
from hidet.runtime.compiled_graph import CompiledGraph
from centml.compiler.config import config_instance, CompilationStatus
from centml.compiler.utils import get_backend_compiled_forward_path

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
            logging.getLogger(__name__).exception("Remote compilation failed with the following exception: \n")

    @property
    def module(self) -> GraphModule:
        return self._module()

    @property
    def inputs(self) -> List[torch.Tensor]:
        return self._inputs

    @module.deleter
    def module(self):
        self._module().graph.owning_module = None
        self._module = None

    @inputs.deleter
    def inputs(self):
        self._inputs = None

    def _get_model_id(self) -> str:
        # We use to_folder to save the GraphModule's:
        # - state dict (weights and more) in pickled form (using torch.save)
        # - submodules (layers, activation functions, etc.), usally as pickled files 
        # - parameters and buffers (in the state dict)
        # the GraphModule's Graph is not saved since the code generated from it is

        with tempfile.TemporaryDirectory() as tempdir:
            with warnings.catch_warnings():
                # to_folder gives a ignorable warning when it needs to pickle submodules
                warnings.filterwarnings("ignore")
                self.module.to_folder(tempdir)

            # The module.py file will contain the tempdir's path. Since tempfile's name change, 
            # we remove occurances to this path string to keep the hash consistent
            module_file = os.path.join(tempdir, "module.py")
            with open(module_file, 'r') as file:
                file_data = file.read()
            
            tempdir_name = tempdir.split("/")[-1]
            file_data = file_data.replace(tempdir_name, 'path')
            
            with open(module_file, 'w') as file:
                file.write(file_data)

            sha_hash = hashlib.sha256()
            
            for root, _, files in os.walk(tempdir):
                files.sort() # Enforce consistent order of files
                for file in files:
                    file_path = os.path.join(root, file)
                    with open(file_path, 'rb') as f:
                        # Read in chunks to avoid loading too much into memory
                        for block in iter(lambda: f.read(4096), b""):
                            sha_hash.update(block)

            # Hash the metadata since it's not saved with to_folder
            if self.module.meta:
                json_metadata: str = json.dumps(self.module.meta, sort_keys=True)
                sha_hash.update(json_metadata.encode())

        return sha_hash.hexdigest()

    def _download_model(self, model_id: str) -> CompiledGraph:
        download_response = requests.get(
            url=f"{config_instance.SERVER_URL}/download/{model_id}", timeout=config_instance.TIMEOUT
        )
        if download_response.status_code != HTTPStatus.OK:
            raise Exception(
                f"Download: request failed, exception from server:\n{download_response.json().get('detail')}"
            )
        download_path = get_backend_compiled_forward_path(model_id)
        with open(download_path, "wb") as f:
            f.write(download_response.content)
            return pickle.loads(download_response.content)

    def _compile_model(self, model_id: str):
        compile_response = requests.post(
            url=f"{config_instance.SERVER_URL}/submit/{model_id}",
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
            status_response = requests.get(
                f"{config_instance.SERVER_URL}/status/{model_id}", timeout=config_instance.TIMEOUT
            )
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
        model_id = self._get_model_id()
        print(model_id)

        # check if compiled forward is saved locally
        compiled_forward_path = get_backend_compiled_forward_path(model_id)
        if os.path.isfile(compiled_forward_path):
            with open(compiled_forward_path, "rb") as f:
                compiled_forward = pickle.load(f)
        else:
            self._wait_for_status(model_id)
            compiled_forward = self._download_model(model_id)

        self.compiled_forward_function = compiled_forward

        # Let garbage collector free the memory used by the uncompiled model
        with self.lock:
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
