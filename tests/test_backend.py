import os
import warnings
from http import HTTPStatus
from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch, MagicMock
import torch
from parameterized import parameterized_class
from torch.fx import GraphModule
import hidet
from hidet.graph.frontend.torch.dynamo_backends import get_flow_graph
from centml.compiler.backend import Runner
from centml.compiler.config import CompilationStatus, config_instance
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
        model = MagicMock(spec=GraphModule)
        self.flowgraph = MagicMock(spec=hidet.FlowGraph)
        self.runner = Runner(model, None)

    @patch('hidet.save_graph')
    def test_none_flow_graph(self, mock_save_graph):
        with self.assertRaises(Exception) as context:
            self.runner._get_model_id(None)

        self.assertIn("Getting model id: flow graph is None.", str(context.exception))
        mock_save_graph.assert_not_called()

    @patch('hidet.save_graph')
    def test_exception_on_save_graph_failure(self, mock_save_graph):
        mock_save_graph.side_effect = Exception("Test Exception")
        with self.assertRaises(Exception) as context:
            self.runner._get_model_id(self.flowgraph)

        self.assertIn("Getting model id: failed to save FlowGraph.", str(context.exception))
        mock_save_graph.assert_called_once()

    # Stop remote_compilation after grabbing model_id from _wait_for_status
    def exit_early(self, *args):
        raise Exception("Exiting early")

    # Don't run remote_compile in a seperate thread
    def start_func(thread_self):
        thread_self._target()

    def tearDown(self) -> None:
        torch._dynamo.reset()

    # Given the same model graph, the model id should be the same
    # Check this by grabbing the model_id passed to _wait_for_status
    @patch("os.path.isfile", new=lambda x: False)
    @patch("threading.Thread.start", new=start_func)
    @patch("centml.compiler.backend.Runner._wait_for_status", side_effect=exit_early)
    def test_model_id_consistency(self, mock_wait):
        model_compiled_1 = torch.compile(self.model, backend="centml")
        model_compiled_1(self.inputs)
        hash_1 = mock_wait.call_args[0][0]

        # Reset the dynamo cache to force recompilation
        torch._dynamo.reset()

        model_compiled_2 = torch.compile(self.model, backend="centml")
        model_compiled_2(self.inputs)
        hash_2 = mock_wait.call_args[0][0]

        self.assertEqual(hash_1, hash_2)
        torch._dynamo.reset()

    # Given two different models, the model ids should be different
    @patch("os.path.isfile", new=lambda x: False)
    @patch("threading.Thread.start", new=start_func)
    @patch("centml.compiler.backend.Runner._wait_for_status", side_effect=exit_early)
    def test_model_id_uniqueness(self, mock_wait):
        def get_modified_model(model):
            modified = deepcopy(model)
            for param in modified.parameters():
                param.data.add_(1)
                break
            return modified

        model_2 = get_modified_model(self.model)

        model_compiled_1 = torch.compile(self.model, backend="centml")
        model_compiled_1(self.inputs)
        hash_1 = mock_wait.call_args[0][0]

        model_compiled_2 = torch.compile(model_2, backend="centml")
        model_compiled_2(self.inputs)
        hash_2 = mock_wait.call_args[0][0]

        self.assertNotEqual(hash_1, hash_2)


class TestDownloadModel(TestCase):
    @patch("threading.Thread.start")
    def setUp(self, mock_thread) -> None:
        model = MagicMock(spec=GraphModule)
        self.runner = Runner(model, None)

    @patch("os.makedirs")
    @patch("centml.compiler.backend.requests")
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

    @patch("os.makedirs")
    @patch("builtins.open")
    @patch("centml.compiler.backend.load_compiled_graph")
    @patch("centml.compiler.backend.requests")
    def test_successful_download(self, mock_requests, mock_load, mock_open, mock_makedirs):
        # Mock the response from the requests library
        mock_response = MagicMock(spec=bytes)
        mock_response.status_code = HTTPStatus.OK
        mock_response.content = b"model_data"
        mock_requests.get.return_value = mock_response

        # Call the _download_model function
        model_id = "download_success"
        self.runner._download_model(model_id)

        mock_requests.get.assert_called_once()
        mock_load.assert_called_once()
        mock_open.assert_called_once()
        mock_makedirs.assert_called_once()


class TestWaitForStatus(TestCase):
    @patch("threading.Thread.start")
    def setUp(self, mock_thread) -> None:
        model = MagicMock(spec=GraphModule)
        self.runner = Runner(model, None)

    @patch("centml.compiler.backend.requests")
    def test_invalid_status(self, mock_requests):
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.BAD_REQUEST
        mock_requests.get.return_value = mock_response

        model_id = "invalid_status"
        with self.assertRaises(Exception) as context:
            self.runner._wait_for_status(model_id)

        mock_requests.get.assert_called_once()
        self.assertIn("Status check: request failed, exception from server", str(context.exception))

    @patch("centml.compiler.config.Config.COMPILING_SLEEP_TIME", new=0)
    @patch("centml.compiler.backend.Runner._compile_model")
    @patch("centml.compiler.backend.requests")
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

    @patch("centml.compiler.config.Config.COMPILING_SLEEP_TIME", new=0)
    @patch("centml.compiler.backend.requests")
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

    @patch("centml.compiler.backend.requests")
    def test_compilation_done(self, mock_requests):
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.json.return_value = {"status": CompilationStatus.DONE.value}
        mock_requests.get.return_value = mock_response

        model_id = "compilation_done"
        self.runner._wait_for_status(model_id)


class TestRemoteCompilation(TestCase):
    @patch("threading.Thread.start")
    def setUp(self, mock_thread):
        model = torch.hub.load("pytorch/vision:v0.10.0", "resnet34", pretrained=True).eval()
        graph_module = torch.fx.symbolic_trace(model)
        self.runner = Runner(graph_module, [torch.zeros(1, 3, 224, 224)])

    @patch("os.path.isfile", new=lambda x: True)
    @patch("centml.compiler.backend.load_compiled_graph")
    def test_cgraph_saved(self, mock_load):
        mock_load.return_value = MagicMock()

        self.runner.remote_compilation()
        mock_load.assert_called_once()

    @patch('os.path.isfile', new=lambda x: False)
    @patch('centml.compiler.backend.Runner._download_model')
    @patch('centml.compiler.backend.Runner._wait_for_status')
    def test_cgraph_not_saved(self, mock_status, mock_download):
        mock_status.return_value = True
        mock_download.return_value = MagicMock()

        self.runner.remote_compilation()
        mock_status.assert_called_once()
        mock_download.assert_called_once()


import unittest

if __name__ == "__main__":
    unittest.main()
