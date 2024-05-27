import io
import os
import gc
import time
import shutil
import hashlib
import logging
import weakref
import tempfile
import threading as th
from http import HTTPStatus
from typing import List, Callable
import requests
from torch.fx import GraphModule
import torch
from hidet.runtime.compiled_graph import CompiledGraph
from centml.compiler.config import config_instance, CompilationStatus
from centml.compiler.utils import get_backend_compiled_forward_path


class Runner:
    def __init__(self, module: GraphModule, inputs: List[torch.Tensor]):
        if not module:
            raise Exception("No module provided")
        
        self._module: GraphModule = weakref.ref(module)
        self._inputs: List[torch.Tensor] = inputs
        self.compiled_forward_function: Callable[[torch.Tensor], tuple] = None
        self.lock = th.Lock()
        self.child_thread = th.Thread(target=self.remote_compilation)

        self.serialized_model_dir = tempfile.mkdtemp()
        self.seralized_model_path = os.path.join(self.serialized_model_dir, config_instance.SERIALIZED_MODEL_FILE)

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

    def __del__(self):
        
        if os.path.exists(self.serialized_model_dir):
            shutil.rmtree(self.serialized_model_dir)

    def _get_model_id(self) -> str:
        # The GraphModule can be large, so lets serialize it to disk
        # This saves a zip file full of pickled files.
        try:
            torch.save(self.module, self.seralized_model_path, pickle_protocol=config_instance.PICKLE_PROTOCOL)
        except Exception as e:
            raise Exception(f"Failed to save module with torch.save: {e}") from e

        sha_hash = hashlib.sha256()
        with open(self.seralized_model_path, "rb") as serialized_model_file:
            # Read in chunks to not load too much into memory
            for block in iter(lambda: serialized_model_file.read(4096), b""):
                sha_hash.update(block)

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
        return torch.load(download_path)

    def _compile_model(self, model_id: str):
        # The model should have been saved using torch.save when we found the model_id
        if not os.path.isfile(self.seralized_model_path):
            raise Exception(f"Model not saved at path {self.seralized_model_path}")

        # Inputs should not be too large, so we can serialize them in memory
        serialized_inputs = io.BytesIO()
        torch.save(self.inputs, serialized_inputs, pickle_protocol=config_instance.PICKLE_PROTOCOL)
        serialized_inputs.seek(0)  # seek since serialized_inputs is still in memory

        with open(self.seralized_model_path, 'rb') as model_file:
            compile_response = requests.post(
                url=f"{config_instance.SERVER_URL}/submit/{model_id}",
                files={"model": model_file, "inputs": serialized_inputs},
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

        # check if compiled forward is saved locally
        compiled_forward_path = get_backend_compiled_forward_path(model_id)
        if os.path.isfile(compiled_forward_path):
            compiled_forward = torch.load(compiled_forward_path)
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
