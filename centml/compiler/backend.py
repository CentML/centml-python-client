import os
import gc
import time
import hashlib
import logging
import threading as th
from http import HTTPStatus
from weakref import ReferenceType, ref
from tempfile import TemporaryDirectory
from typing import List, Callable, Optional
import requests
import torch
from torch.fx import GraphModule
from centml.compiler.config import settings, CompilationStatus
from centml.compiler.utils import get_backend_compiled_forward_path


class Runner:
    def __init__(self, module: GraphModule, inputs: List[torch.Tensor]):
        if not module:
            raise Exception("No module provided")

        self._module: ReferenceType[GraphModule] = ref(module)
        self._inputs: List[torch.Tensor] = inputs
        self.compiled_forward_function: Optional[Callable[[torch.Tensor], tuple]] = None
        self.lock = th.Lock()
        self.child_thread = th.Thread(target=self.remote_compilation_starter)

        self.serialized_model_dir: Optional[TemporaryDirectory] = None
        self.serialized_model_path: Optional[str] = None
        self.serialized_input_path: Optional[str] = None

        try:
            self.child_thread.start()
        except Exception as e:
            logging.getLogger(__name__).exception(f"Failed to start compilation thread\n{e}")

    @property
    def module(self) -> Optional[GraphModule]:
        return self._module()

    @module.deleter
    def module(self):
        self._module().graph.owning_module = None
        self._module = None

    @property
    def inputs(self) -> List[torch.Tensor]:
        return self._inputs

    @inputs.deleter
    def inputs(self):
        self._inputs = None

    def _serialize_model_and_inputs(self):
        self.serialized_model_dir = TemporaryDirectory()  # pylint: disable=consider-using-with
        self.serialized_model_path = os.path.join(self.serialized_model_dir.name, settings.CENTML_SERIALIZED_MODEL_FILE)
        self.serialized_input_path = os.path.join(self.serialized_model_dir.name, settings.CENTML_SERIALIZED_INPUT_FILE)

        # torch.save saves a zip file full of pickled files with the model's states.
        try:
            torch.save(self.module, self.serialized_model_path, pickle_protocol=settings.CENTML_PICKLE_PROTOCOL)
            torch.save(self.inputs, self.serialized_input_path, pickle_protocol=settings.CENTML_PICKLE_PROTOCOL)
        except Exception as e:
            raise Exception(f"Failed to save module or inputs with torch.save: {e}") from e

    def _get_model_id(self) -> str:
        if not self.serialized_model_path or not os.path.isfile(self.serialized_model_path):
            raise Exception(f"Model not saved at path {self.serialized_model_path}")

        sha_hash = hashlib.sha256()
        with open(self.serialized_model_path, "rb") as serialized_model_file:
            # Read in chunks to not load too much into memory
            for block in iter(lambda: serialized_model_file.read(settings.CENTML_HASH_CHUNK_SIZE), b""):
                sha_hash.update(block)

        model_id = sha_hash.hexdigest()
        logging.info(f"Model has id {model_id}")
        return model_id

    def _download_model(self, model_id: str):
        download_response = requests.get(
            url=f"{settings.CENTML_SERVER_URL}/download/{model_id}", timeout=settings.CENTML_COMPILER_TIMEOUT
        )
        if download_response.status_code != HTTPStatus.OK:
            raise Exception(
                f"Download: request failed, exception from server:\n{download_response.json().get('detail')}"
            )
        if download_response.content == b"":
            raise Exception("Download: empty response from server")
        download_path = get_backend_compiled_forward_path(model_id)
        with open(download_path, "wb") as f:
            f.write(download_response.content)
        return torch.load(download_path)

    def _compile_model(self, model_id: str):
        # The model should have been saved using torch.save when we found the model_id
        if not self.serialized_model_path or not self.serialized_input_path:
            raise Exception("Model or inputs not serialized")
        if not os.path.isfile(self.serialized_model_path):
            raise Exception(f"Model not saved at path {self.serialized_model_path}")
        if not os.path.isfile(self.serialized_input_path):
            raise Exception(f"Inputs not saved at path {self.serialized_input_path}")

        with open(self.serialized_model_path, 'rb') as model_file, open(self.serialized_input_path, 'rb') as input_file:
            compile_response = requests.post(
                url=f"{settings.CENTML_SERVER_URL}/submit/{model_id}",
                files={"model": model_file, "inputs": input_file},
                timeout=settings.CENTML_COMPILER_TIMEOUT,
            )
        if compile_response.status_code != HTTPStatus.OK:
            raise Exception(
                f"Compile model: request failed, exception from server:\n{compile_response.json().get('detail')}\n"
            )

    def _wait_for_status(self, model_id: str) -> bool:
        tries = 0
        while True:
            # get server compilation status
            status = None
            try:
                status_response = requests.get(
                    f"{settings.CENTML_SERVER_URL}/status/{model_id}", timeout=settings.CENTML_COMPILER_TIMEOUT
                )
                if status_response.status_code != HTTPStatus.OK:
                    raise Exception(
                        f"Status check: request failed, exception from server:\n{status_response.json().get('detail')}"
                    )
                status = status_response.json().get("status")
            except Exception as e:
                logging.getLogger(__name__).exception(f"Status check failed:\n{e}")

            if status == CompilationStatus.DONE.value:
                return True
            elif status == CompilationStatus.COMPILING.value:
                pass
            elif status == CompilationStatus.NOT_FOUND.value:
                logging.info("Submitting model to server for compilation.")
                try:
                    self._compile_model(model_id)
                except Exception as e:
                    logging.getLogger(__name__).exception(f"Submitting compilation failed:\n{e}")
                tries += 1
            else:
                tries += 1

            if tries > settings.CENTML_COMPILER_MAX_RETRIES:
                raise Exception("Waiting for status: compilation failed too many times.\n")

            time.sleep(settings.CENTML_COMPILER_SLEEP_TIME)

    def remote_compilation_starter(self):
        try:
            self.remote_compilation()
        except Exception as e:
            logging.getLogger(__name__).exception(f"Compilation thread failed:\n{e}")

    def remote_compilation(self):
        self._serialize_model_and_inputs()

        model_id = self._get_model_id()

        # check if compiled forward is saved locally
        compiled_forward_path = get_backend_compiled_forward_path(model_id)
        if os.path.isfile(compiled_forward_path):
            logging.info("Compiled model found in local cache. Not submitting to server.")
            compiled_forward = torch.load(compiled_forward_path)
        else:
            self._wait_for_status(model_id)
            compiled_forward = self._download_model(model_id)

        self.compiled_forward_function = compiled_forward

        logging.info("Compilation successful.")

        # Let garbage collector free the memory used by the uncompiled model
        with self.lock:
            del self.inputs
            if self.module:
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
