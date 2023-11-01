import os
import pickle
import hashlib
import time
import tempfile
from http import HTTPStatus
from typing import List
import requests
import hidet
import torch
from hidet.graph.frontend.torch.interpreter import Interpreter
from hidet.graph.frontend.torch.dynamo_backends import get_flow_graph, get_wrapper
from hidet.runtime.compiled_graph import load_compiled_graph
from centml.compiler import config_instance
from centml.compiler.server_compilation import CompilationStatus

base_path = os.path.join(config_instance.CACHE_PATH, "compiler")
os.makedirs(base_path, exist_ok=True)
server_url = f"http://{config_instance.SERVER_IP}:{config_instance.SERVER_PORT}"


class Runner:
    def __init__(self, module, inputs):
        self._module = module
        self._inputs = inputs
        self.compiled_forward_function = None

        self.remote_compilation()

    @property
    def module(self):
        return self._module

    @property
    def inputs(self):
        return self._inputs

    def __get_model_id(self, flow_graph):
        with tempfile.NamedTemporaryFile() as temp_file:
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
            raise Exception(f"Download request failed, exception from server: {download_response.json().get('detail')}")

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

    # 
    def __wait_for_status(self, model_id):
        failed_tries = 0

        while True:
            #get server compilation status
            status_response = requests.get(f"{server_url}/status/{model_id}", timeout=config_instance.TIMEOUT)
            if status_response.status_code != HTTPStatus.OK:
                raise Exception(f"Status check failed, exception from server: {status_response.json().get('detail')}")
            status = status_response.json().get("status")
            
            if   status == CompilationStatus.DONE.value:
                return True
            elif status == CompilationStatus.COMPILING.value:
                print("COMPILING")
                continue
            elif status == CompilationStatus.NOT_FOUND.value or compiled_response.status_code != HTTPStatus.OK:
                failed_tries += 1
            else:
                raise Exception("Server returned invalid status response")

            if failed_tries > config_instance.MAX_RETRIES:
                return False
                # failure_reason = (
                #     f"Most recent server exception: {compiled_response.json().get('detail')}."
                #     if compiled_response.status_code != HTTPStatus.OK
                #     else "Compilation not found on server."
                # )
                # raise Exception("Compilation failed too many times. " + failure_reason)
            else:
                compiled_response = self.__compile_model(model_id)
                # time.sleep(config_instance.COMPILING_SLEEP_TIME)

    def remote_compilation(self):
        #start by getting the model_id
        interpreter: Interpreter = hidet.frontend.from_torch(self.module)
        flow_graph, inputs, output_format = get_flow_graph(interpreter, self.inputs)
        model_id = self.__get_model_id(flow_graph)

        cgraph_path = os.path.join(base_path, f"cgraph_{model_id}.temp")
        if os.path.isfile(cgraph_path): # cgraph is saved locally
            cgraph = load_compiled_graph(cgraph_path)
        else:
            status_return = self.__wait_for_status(model_id)

            if not status_return:
                # 
                raise Exception("status check failed")
                
        cgraph = self.__download_model(model_id)

        wrapper = get_wrapper(cgraph, inputs, output_format)

        self.compiled_forward_function = wrapper

    def __call__(self, *args, **kwargs):
        # If model is currently compiling, return the uncompiled forward function
        if not self.compiled_forward_function:
            return self.module(*args, **kwargs)

        forward_function = self.compiled_forward_function(*args)
        return forward_function


def centml_dynamo_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    return Runner(gm, example_inputs)
