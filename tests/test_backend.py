from http import HTTPStatus
from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch, MagicMock
import torch
from parameterized import parameterized_class
from torch.fx import GraphModule
import hidet
from centml.compiler.backend import Runner
from centml.compiler.config import CompilationStatus, config_instance
from .test_helpers import MODEL_SUITE


class SetUpGraphModule(TestCase):
    @patch('threading.Thread.start', new=lambda x: None)
    def setUp(self) -> None:
        model = MagicMock(spec=GraphModule)
        self.runner = Runner(model, None)


@parameterized_class(list(MODEL_SUITE.values()))
class TestGetModelId(SetUpGraphModule):
    # Reset the dynamo cache to force recompilation
    def tearDown(self) -> None:
        torch._dynamo.reset()

    @patch('hidet.save_graph')
    def test_none_flow_graph(self, mock_save_graph):
        with self.assertRaises(Exception) as context:
            self.runner._get_model_id(None)

        self.assertIn("Getting model id: flow graph is None.", str(context.exception))
        mock_save_graph.assert_not_called()

    @patch('hidet.save_graph')
    def test_exception_on_save_graph_failure(self, mock_save_graph):
        mock_save_graph.side_effect = Exception("Test Exception")
        flowgraph = MagicMock(spec=hidet.FlowGraph)

        with self.assertRaises(Exception) as context:
            self.runner._get_model_id(flowgraph)

        self.assertIn("Getting model id: failed to save FlowGraph.", str(context.exception))
        mock_save_graph.assert_called_once()

    # Don't run remote_compile in a seperate thread
    def start_func(self):
        self._target()

    # Given the same model graph, the model id should be the same
    # Check this by grabbing the model_id passed to _wait_for_status
    @patch("os.path.isfile", new=lambda x: False)
    @patch("threading.Thread.start", new=start_func)
    @patch("centml.compiler.backend.Runner._wait_for_status", side_effect=Exception("Exiting early"))
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
    @patch("centml.compiler.backend.Runner._wait_for_status", side_effect=Exception("Exiting early"))
    def test_model_id_uniqueness(self, mock_wait):
        def get_modified_model(model):
            modified = deepcopy(model)
            next(modified.parameters()).data.add_(1)
            return modified

        model_compiled_1 = torch.compile(self.model, backend="centml")
        model_compiled_1(self.inputs)
        hash_1 = mock_wait.call_args[0][0]

        model_2 = get_modified_model(self.model)
        model_compiled_2 = torch.compile(model_2, backend="centml")
        model_compiled_2(self.inputs)
        hash_2 = mock_wait.call_args[0][0]

        self.assertNotEqual(hash_1, hash_2)


class TestDownloadModel(SetUpGraphModule):
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


class TestWaitForStatus(SetUpGraphModule):
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


@parameterized_class(list(MODEL_SUITE.values()))
class TestRemoteCompilation(TestCase):
    def call_remote_compilation(self):
        # Ensure remote_compilation is called in the same thread
        def start_func(thread_self):
            thread_self._target()

        # Ensure the default forward function is called
        @staticmethod
        def call_default_forward(*args, **kwargs):
            return self.model.forward(*args, **kwargs)

        with patch("threading.Thread.start", new=start_func), patch(
            "centml.compiler.backend.Runner.__call__", new=call_default_forward
        ):
            compiled_model = torch.compile(self.model, backend="centml")
            compiled_model(self.inputs)

        torch._dynamo.reset()

    @patch("os.path.isfile", new=lambda x: True)
    @patch("centml.compiler.backend.load_compiled_graph")
    def test_cgraph_saved(self, mock_load):
        mock_load.return_value = MagicMock()

        self.call_remote_compilation()
        mock_load.assert_called_once()

    @patch('os.path.isfile', new=lambda x: False)
    @patch('centml.compiler.backend.Runner._download_model')
    @patch('centml.compiler.backend.Runner._wait_for_status')
    def test_cgraph_not_saved(self, mock_status, mock_download):
        mock_status.return_value = True
        mock_download.return_value = MagicMock()

        self.call_remote_compilation()
        mock_status.assert_called_once()
        mock_download.assert_called_once()
