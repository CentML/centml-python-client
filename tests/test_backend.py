from http import HTTPStatus
from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch, MagicMock
import torch
from parameterized import parameterized_class
from torch.fx import GraphModule
import centml
from centml.compiler.backend import Runner
from centml.compiler.config import CompilationStatus, settings
from .test_helpers import MODEL_SUITE


# Ensure remote_compilation is called in the same thread
def start_func(thread_self):
    thread_self._target()


class SetUpGraphModule(TestCase):
    @patch('threading.Thread.start', new=lambda x: None)
    def setUp(self) -> None:
        model = MagicMock(spec=GraphModule)
        inputs = [torch.tensor([1.0])]
        self.runner = Runner(model, inputs)


@parameterized_class(list(MODEL_SUITE.values()))
class TestGetModelId(SetUpGraphModule):
    # Reset the dynamo cache to force recompilation
    def tearDown(self) -> None:
        torch._dynamo.reset()

    @patch("centml.compiler.backend.os.path.isfile", new=lambda x: False)
    def test_no_serialized_model(self):
        with self.assertRaises(Exception) as context:
            self.runner._get_model_id()

        self.assertIn("Model not saved at path", str(context.exception))

    # Given the same model graph, the model id should be the same
    # Grab the model_id's passed to get_backend_compiled_forward_path
    @patch("threading.Thread.start", new=start_func)
    @patch("centml.compiler.backend.get_backend_compiled_forward_path", side_effect=Exception("Exiting early"))
    def test_model_id_consistency(self, mock_get_path):
        # self.model and self.inputs come from @parameterized_class
        model_compiled_1 = centml.compile(self.model)
        model_compiled_1(self.inputs)
        hash_1 = mock_get_path.call_args[0][0]
        torch._dynamo.reset()  # Reset the dynamo cache to force recompilation

        model_compiled_2 = centml.compile(self.model)
        model_compiled_2(self.inputs)
        hash_2 = mock_get_path.call_args[0][0]
        torch._dynamo.reset()

        self.assertEqual(hash_1, hash_2)

    # Given two different models, the model ids should be different
    # We made the models different by adding 1 to the first value in some layer's
    # Grab the model_id's passed to get_backend_compiled_forward_path
    @patch("threading.Thread.start", new=start_func)
    @patch("centml.compiler.backend.get_backend_compiled_forward_path", side_effect=Exception("Exiting early"))
    def test_model_id_uniqueness(self, mock_get_path):
        def get_modified_model(model):
            modified = deepcopy(model)
            state_dict = modified.state_dict()
            some_layer = list(state_dict.values())[0]
            some_layer.view(-1)[0] += 1
            return modified

        # self.model and self.inputs come from @parameterized_class
        model_compiled_1 = centml.compile(self.model)
        model_compiled_1(self.inputs)
        hash_1 = mock_get_path.call_args[0][0]

        model_2 = get_modified_model(self.model)
        model_compiled_2 = centml.compile(model_2)
        model_compiled_2(self.inputs)
        hash_2 = mock_get_path.call_args[0][0]

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
    @patch("centml.compiler.backend.torch.load")
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
    @patch("centml.compiler.config.settings.CENTML_COMPILER_SLEEP_TIME", new=0)
    @patch("centml.compiler.backend.requests")
    @patch("logging.Logger.exception")
    def test_invalid_status(self, mock_logger, mock_requests):
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.BAD_REQUEST
        mock_requests.get.return_value = mock_response

        model_id = "invalid_status"
        with self.assertRaises(Exception) as context:
            self.runner._wait_for_status(model_id)

        mock_requests.get.assert_called()
        assert mock_requests.get.call_count == settings.CENTML_COMPILER_MAX_RETRIES + 1
        assert len(mock_logger.call_args_list) == settings.CENTML_COMPILER_MAX_RETRIES + 1
        print(mock_logger.call_args_list)
        assert mock_logger.call_args_list[0].startswith("Status check failed:")
        assert "Waiting for status: compilation failed too many times.\n" == str(context.exception)

    @patch("centml.compiler.config.settings.CENTML_COMPILER_SLEEP_TIME", new=0)
    @patch("centml.compiler.backend.requests")
    @patch("logging.Logger.exception")
    def test_exception_in_status(self, mock_logger, mock_requests):
        exception_message = "Exiting early"
        mock_requests.get.side_effect = Exception(exception_message)

        model_id = "exception_in_status"
        with self.assertRaises(Exception) as context:
            self.runner._wait_for_status(model_id)

        mock_requests.get.assert_called()
        assert mock_requests.get.call_count == settings.CENTML_COMPILER_MAX_RETRIES + 1
        mock_logger.assert_called_with(f"Status check failed:\n{exception_message}")
        assert str(context.exception) == "Waiting for status: compilation failed too many times.\n"

    @patch("centml.compiler.config.settings.CENTML_COMPILER_SLEEP_TIME", new=0)
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

        self.assertEqual(mock_compile.call_count, settings.CENTML_COMPILER_MAX_RETRIES + 1)
        self.assertIn("Waiting for status: compilation failed too many times", str(context.exception))

    @patch("centml.compiler.config.settings.CENTML_COMPILER_SLEEP_TIME", new=0)
    @patch("centml.compiler.backend.requests")
    def test_wait_on_compilation(self, mock_requests):
        # Mock the status check
        COMPILATION_STEPS = 10
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.json.side_effect = [{"status": CompilationStatus.COMPILING.value}] * COMPILATION_STEPS + [
            {"status": CompilationStatus.DONE.value}
        ]
        mock_requests.get.return_value = mock_response

        model_id = "compilation_done"
        # _wait_for_status should return True when compilation DONE
        assert self.runner._wait_for_status(model_id)

    @patch("centml.compiler.config.settings.CENTML_COMPILER_SLEEP_TIME", new=0)
    @patch("centml.compiler.backend.requests")
    @patch("centml.compiler.backend.Runner._compile_model")
    @patch("logging.Logger.exception")
    def test_exception_in_compilation(self, mock_logger, mock_compile, mock_requests):
        # Mock the status check
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.json.return_value = {"status": CompilationStatus.NOT_FOUND.value}
        mock_requests.get.return_value = mock_response

        # Mock the compile model function
        exception_message = "Exiting early"
        mock_compile.side_effect = Exception(exception_message)

        model_id = "exception_in_compilation"
        with self.assertRaises(Exception) as context:
            self.runner._wait_for_status(model_id)

        mock_requests.get.assert_called()
        assert mock_requests.get.call_count == settings.CENTML_COMPILER_MAX_RETRIES + 1

        mock_compile.assert_called()
        assert mock_compile.call_count == settings.CENTML_COMPILER_MAX_RETRIES + 1

        mock_logger.assert_called_with(f"Submitting compilation failed:\n{exception_message}")
        assert str(context.exception) == "Waiting for status: compilation failed too many times.\n"

    @patch("centml.compiler.backend.requests")
    def test_compilation_done(self, mock_requests):
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.json.return_value = {"status": CompilationStatus.DONE.value}
        mock_requests.get.return_value = mock_response

        model_id = "compilation_done"
        # _wait_for_status should return True when compilation DONE
        assert self.runner._wait_for_status(model_id)


@parameterized_class(list(MODEL_SUITE.values()))
class TestRemoteCompilation(TestCase):
    def call_remote_compilation(self):
        with patch("threading.Thread.start", new=start_func), patch(
            "centml.compiler.backend.Runner.__call__", new=self.model.forward
        ):
            compiled_model = centml.compile(self.model)
            compiled_model(self.inputs)

        torch._dynamo.reset()

    @patch("os.path.isfile", new=lambda x: True)
    @patch("centml.compiler.backend.Runner._get_model_id", new=lambda x: "1234")
    @patch("centml.compiler.backend.torch.load")
    def test_compiled_cached(self, mock_load):
        mock_load.return_value = MagicMock()
        self.call_remote_compilation()
        mock_load.assert_called_once()

    @patch('os.path.isfile', new=lambda x: False)
    @patch("centml.compiler.backend.Runner._get_model_id", new=lambda x: "1234")
    @patch('centml.compiler.backend.Runner._download_model')
    @patch('centml.compiler.backend.Runner._wait_for_status')
    def test_compiled_return_not_cached(self, mock_status, mock_download):
        mock_status.return_value = True
        mock_download.return_value = MagicMock()

        self.call_remote_compilation()

        mock_status.assert_called_once()
        mock_download.assert_called_once()
