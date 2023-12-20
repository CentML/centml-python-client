from http import HTTPStatus
from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch, MagicMock
import torch
from torch.fx import GraphModule
import hidet
from hidet.graph.frontend.torch.dynamo_backends import get_flow_graph
from centml.compiler.backend import Runner
from centml.compiler.config import config_instance, CompilationStatus


class TestGetModelId(TestCase):
    def graph_module_to_flow_graph(self, gm: GraphModule):
        interpreter = hidet.frontend.from_torch(gm)
        return get_flow_graph(interpreter, self.inputs)[0]

    @patch('threading.Thread.start')
    def setUp(self, mock_thread) -> None:
        self.inputs = [torch.zeros(1, 3, 224, 224)]
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True, verbose=False).eval()
        self.graph_module = torch.fx.symbolic_trace(self.model)
        self.flow_graph = self.graph_module_to_flow_graph(self.graph_module)
        self.runner = Runner(self.graph_module, None)

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
            self.runner._get_model_id(self.flow_graph)

        self.assertIn("Getting model id: failed to save FlowGraph.", str(context.exception))
        mock_save_graph.assert_called_once()

    # Given the same flow graph, the model id should be the same
    def test_flow_graph_id_consistency(self):
        model_id1 = self.runner._get_model_id(self.flow_graph)
        model_id2 = self.runner._get_model_id(self.flow_graph)
        self.assertEqual(model_id1, model_id2)

    # Given the same torch fx graph, the model id should be the same
    def test_torch_fx_graph_id_consistency(self):
        graph_module_2 = deepcopy(self.graph_module)
        flow_graph_2 = self.graph_module_to_flow_graph(graph_module_2)

        model_id_1 = self.runner._get_model_id(self.flow_graph)
        model_id_2 = self.runner._get_model_id(flow_graph_2)
        self.assertEqual(model_id_1, model_id_2)

    # Given the same torch fx graph, the model id should be the same
    def test_model_id_consistency(self):
        model_2 = deepcopy(self.model)
        graph_module_2 = torch.fx.symbolic_trace(model_2)
        flow_graph_2 = self.graph_module_to_flow_graph(graph_module_2)

        model_id_1 = self.runner._get_model_id(self.flow_graph)
        model_id_2 = self.runner._get_model_id(flow_graph_2)
        self.assertEqual(model_id_1, model_id_2)

    # Given two different models, the model id should be different
    def test_output_uniqueness(self):
        model_34 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True, verbose=False).eval()
        graph_module_34 = torch.fx.symbolic_trace(model_34)
        flow_graph_34 = self.graph_module_to_flow_graph(graph_module_34)

        model_id1 = self.runner._get_model_id(self.flow_graph)
        model_id2 = self.runner._get_model_id(flow_graph_34)
        self.assertNotEqual(model_id1, model_id2)


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
    def setUp(self, mock_thread) -> None:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True, verbose=False).eval()
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
