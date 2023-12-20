import os
import warnings
from http import HTTPStatus
from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch, MagicMock
import torch
from parameterized import parameterized_class
from transformers import BertForPreTraining, AutoTokenizer
from torch.fx import GraphModule
import hidet
from hidet.graph.frontend.torch.dynamo_backends import get_flow_graph
from centml.compiler.backend import Runner
from centml.compiler.server_compilation import CompilationStatus
from centml.compiler import config_instance
from .test_helpers import model_suite

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@parameterized_class([inputs for inputs in model_suite.values()])
class TestGetModelId(TestCase):
    def graph_module_to_flow_graph(self, gm: GraphModule):
        interpreter = hidet.frontend.from_torch(gm)
        return get_flow_graph(interpreter, [self.inputs])[0]

    def assert_flow_graph_id_equal(self, flow_graph_1, flow_graph_2):
        model_id1 = self.runner._get_model_id(flow_graph_1)
        model_id2 = self.runner._get_model_id(flow_graph_2)
        self.assertEqual(model_id1, model_id2)

    @patch('threading.Thread.start')
    def setUp(self, mock_thread) -> None:
        warnings.filterwarnings("ignore", category=UserWarning)

        # self.graph_module = None
        # def custom_backend(gm, inputs):
        #     self.graph_module = gm
        #     return lambda x: [torch.zeros(1)] # don't actually compile anything

        # need to copy self.model to invoke compilation on each test
        # model_compiled = torch.compile(deepcopy(self.model), backend=custom_backend)
        # model_compiled(self.inputs)

        # self.flow_graph = self.graph_module_to_flow_graph(self.graph_module)
        # self.runner = Runner(self.graph_module, None)

    # @patch('hidet.save_graph')
    # def test_none_flow_graph(self, mock_save_graph):
    #     with self.assertRaises(Exception) as context:
    #         self.runner._get_model_id(None)

    #     self.assertIn("Getting model id: flow graph is None.", str(context.exception))
    #     mock_save_graph.assert_not_called()

    # @patch('hidet.save_graph')
    # def test_exception_on_save_graph_failure(self, mock_save_graph):
    #     mock_save_graph.side_effect = Exception("Test Exception")
    #     with self.assertRaises(Exception) as context:
    #         self.runner._get_model_id(self.flow_graph)

    #     self.assertIn("Getting model id: failed to save FlowGraph.", str(context.exception))
    #     mock_save_graph.assert_called_once()

    # Given the same model graph, the model id should be the same
    @patch('os.path.isfile')
    @patch('centml.compiler.backend.Runner._wait_for_status')
    def test_model_id_consistency(self, mock_wait_for_status, mock_is_file):
        mock_is_file.return_value = False
        mock_wait_for_status.side_effect = Exception("Test Exception")
        
        model_compiled = torch.compile(self.model, backend="centml")
        model_compiled(self.inputs)

        hash_1 = mock_wait_for_status.call_args
        print("HASH IS", hash_1)


    # Given two different models, the model id should be different
    # Change an operator's name to make the flow graph different
    # def test_output_uniqueness(self):
    #     different_flow_graph = deepcopy(self.flow_graph)
    #     different_flow_graph.outputs[0].op.name = "different_name"

    #     model_id1 = self.runner._get_model_id(self.flow_graph)
    #     model_id2 = self.runner._get_model_id(different_flow_graph)
    #     self.assertNotEqual(model_id1, model_id2)


class TestDownloadModel(TestCase):
    @patch('threading.Thread.start')
    def setUp(self, mock_thread) -> None:
        model = MagicMock(spec=GraphModule)
        self.runner = Runner(model, None)

    @patch('os.makedirs')
    @patch('centml.compiler.backend.requests')
    def test_failed_download(self, mock_requests, mock_makedirs):
        # Mock the response from the requests library
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.NOT_FOUND
        mock_requests.get.return_value = mock_response

        model_id = "download_fail"
        with self.assertRaises(Exception) as context:
            self.runner._download_model(model_id)

        mock_requests.get.assert_called_once()
        self.assertIn("Download: request failed, exception from server", str(context.exception))
        mock_makedirs.assert_not_called()

    @patch('os.makedirs')
    @patch('builtins.open')
    @patch('centml.compiler.backend.load_compiled_graph')
    @patch('centml.compiler.backend.requests')
    def test_successful_download(self, mock_requests, mock_load, mock_open, mock_makedirs):
        # Mock the response from the requests library
        mock_response = MagicMock(spec=bytes)
        mock_response.status_code = HTTPStatus.OK
        mock_response.content = b'model_data'
        mock_requests.get.return_value = mock_response

        # Call the _download_model function
        model_id = "download_success"
        self.runner._download_model(model_id)

        mock_requests.get.assert_called_once()
        mock_load.assert_called_once()
        mock_open.assert_called_once()
        mock_makedirs.assert_called_once()


class TestWaitForStatus(TestCase):
    @patch('threading.Thread.start')
    def setUp(self, mock_thread) -> None:
        model = MagicMock(spec=GraphModule)
        self.runner = Runner(model, None)

    @patch('centml.compiler.backend.requests')
    def test_invalid_status(self, mock_requests):
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.BAD_REQUEST
        mock_requests.get.return_value = mock_response

        model_id = "invalid_status"
        with self.assertRaises(Exception) as context:
            self.runner._wait_for_status(model_id)

        mock_requests.get.assert_called_once()
        self.assertIn("Status check: request failed, exception from server", str(context.exception))

    @patch('centml.compiler.config.Config.COMPILING_SLEEP_TIME', new=0)
    @patch('centml.compiler.backend.Runner._compile_model')
    @patch('centml.compiler.backend.requests')
    def test_max_tries(self, mock_requests, mock_compile):
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.json.return_value = {"status": CompilationStatus.NOT_FOUND.value}
        mock_requests.get.return_value = mock_response

        model_id = "max_tries"
        with self.assertRaises(Exception) as context:
            self.runner._wait_for_status(model_id)

        self.assertEqual(mock_compile.call_count, config_instance.MAX_RETRIES + 1)
        self.assertIn("Waiting for status: compilation failed too many times", str(context.exception))

    @patch('centml.compiler.config.Config.COMPILING_SLEEP_TIME', new=0)
    @patch('centml.compiler.backend.requests')
    def test_wait_on_compilation(self, mock_requests):
        COMPILATION_STEPS = 10
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.json.side_effect = [{"status": CompilationStatus.COMPILING.value}] * COMPILATION_STEPS + [
            {"status": CompilationStatus.DONE.value}
        ]
        mock_requests.get.return_value = mock_response

        model_id = "compilation_done"
        self.runner._wait_for_status(model_id)

    @patch('centml.compiler.backend.requests')
    def test_compilation_done(self, mock_requests):
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.json.return_value = {"status": CompilationStatus.DONE.value}
        mock_requests.get.return_value = mock_response

        model_id = "compilation_done"
        self.runner._wait_for_status(model_id)


class TestRemoteCompilation(TestCase):
    @patch('threading.Thread.start')
    def setUp(self, mock_thread):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True).eval()
        graph_module = torch.fx.symbolic_trace(model)
        self.runner = Runner(graph_module, [torch.zeros(1, 3, 224, 224)])

    @patch('os.path.isfile')
    @patch('centml.compiler.backend.load_compiled_graph')
    def test_cgraph_saved(self, mock_load, mock_isfile):
        mock_isfile.return_value = True
        mock_load.return_value = MagicMock()

        self.runner.remote_compilation()
        mock_load.assert_called_once()

    @patch('os.path.isfile')
    @patch('centml.compiler.backend.Runner._download_model')
    @patch('centml.compiler.backend.Runner._wait_for_status')
    def test_cgraph_not_saved(self, mock_status, mock_download, mock_isfile):
        mock_isfile.return_value = False
        mock_status.return_value = True
        mock_download.return_value = MagicMock()

        self.runner.remote_compilation()
        mock_status.assert_called_once()
        mock_download.assert_called_once()

import unittest
if __name__ == '__main__':
    t = TestGetModelId_0()
    t.setUp()