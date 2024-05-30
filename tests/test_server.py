from unittest import TestCase
from unittest.mock import MagicMock, patch
from http import HTTPStatus
import pytest
import torch
import hidet
from fastapi import UploadFile, HTTPException
from fastapi.testclient import TestClient
from parameterized import parameterized_class
from centml.compiler.server import app, background_compile, read_upload_files
from centml.compiler.config import CompilationStatus
from tests.test_helpers import MODEL_SUITE, get_dummy_model_and_inputs

client = TestClient(app)


class TestStatusHandler(TestCase):
    def test_empty_request(self):
        response = client.get("/status/")
        self.assertEqual(response.status_code, HTTPStatus.NOT_FOUND)

    @patch("os.path.isdir", new=lambda x: False)
    def test_model_not_found(self):
        model_id = "nonexistent_model"
        response = client.get(f"/status/{model_id}")
        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertEqual(response.json(), {"status": CompilationStatus.NOT_FOUND.value})

    @patch("os.path.isfile", new=lambda x: False)
    @patch("os.path.isdir", new=lambda x: True)
    def test_model_compiling(self):
        model_id = "compiling_model"
        response = client.get(f"/status/{model_id}")
        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertEqual(response.json(), {"status": CompilationStatus.COMPILING.value})

    @patch("os.path.isfile", new=lambda x: True)
    @patch("os.path.isdir", new=lambda x: True)
    def test_model_done(self):
        model_id = "completed_model"
        response = client.get(f"/status/{model_id}")
        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertEqual(response.json(), {"status": CompilationStatus.DONE.value})


@parameterized_class(list(MODEL_SUITE.values()))
class TestBackgroundCompile(TestCase):
    @pytest.mark.gpu
    @patch("os.rename")
    @patch("logging.Logger.exception")
    @patch("centml.compiler.server.torch.save")
    def test_successful_compilation(self, mock_save, mock_logger, mock_rename):
        # For some reason there is a deadlock with parallel builds
        hidet.option.parallel_build(False)

        # Get the graph_module and example inputs that would be passed to background compile
        class MockRunner:
            def __init__(self):
                self.graph_module = None
                self.example_inputs = None

            def __call__(self, module, inputs):
                self.graph_module, self.example_inputs = module, inputs
                return module.forward

        mock_init = MockRunner()

        # self.model and self.inputs come from @parameterized_class
        model, inputs = self.model.cuda(), self.inputs.cuda()
        model_compiled = torch.compile(model, backend=mock_init)
        model_compiled(inputs)

        model_id = "successful_model"
        background_compile(model_id, mock_init.graph_module, mock_init.example_inputs)

        mock_rename.assert_called_once()
        mock_save.assert_called_once()
        mock_logger.assert_not_called()


class TestReadUploadFiles(TestCase):
    def test_mock_cant_read(self):
        model_id = "file_cant_be_read"

        mock_file = MagicMock()
        mock_file.file.read.side_effect = Exception("an exception occurred")

        with self.assertRaises(HTTPException) as excinfo:
            read_upload_files(model_id, mock_file, mock_file)

        self.assertEqual(excinfo.exception.status_code, HTTPStatus.BAD_REQUEST)
        self.assertIn("Compilation: error reading serialized content", str(excinfo.exception))

    @patch("torch.load", side_effect=Exception("an exception occurred"))
    def test_cant_load(self, mock_load):
        model_id = "file_cant_be_unpickled"

        # Create file-like objects with test data
        model_file, input_file = get_dummy_model_and_inputs("model", "inputs")
        model = UploadFile(filename="model", file=model_file)
        inputs = UploadFile(filename="inputs", file=input_file)

        with self.assertRaises(HTTPException) as excinfo:
            read_upload_files(model_id, model, inputs)

        mock_load.assert_called_once()
        self.assertEqual(excinfo.exception.status_code, HTTPStatus.BAD_REQUEST)
        self.assertIn("Compilation: error loading content with torch.load:", str(excinfo.exception))

    def test_proper_read(self):
        model_id = "test_model_id"

        model_data, inputs_data = "model", "inputs"

        # Create file-like objects with test data
        model_file, input_file = get_dummy_model_and_inputs("model", "inputs")
        model = UploadFile(filename="model", file=model_file)
        inputs = UploadFile(filename="inputs", file=input_file)

        tfx_graph, example_inputs = read_upload_files(model_id, model, inputs)

        self.assertEqual(tfx_graph, model_data)
        self.assertEqual(example_inputs, inputs_data)


class TestCompileHandler(TestCase):
    @patch("centml.compiler.server.get_status")
    def test_model_compiling(self, mock_status):
        model_id = "compiling_model"
        mock_status.return_value = CompilationStatus.COMPILING

        model_file, input_file = get_dummy_model_and_inputs("model", "inputs")
        response = client.post(f"/submit/{model_id}", files={"model": model_file, "inputs": input_file})
        self.assertEqual(response.status_code, HTTPStatus.OK)

    @patch("os.makedirs")
    @patch("centml.compiler.server.background_compile")
    @patch("centml.compiler.server.get_status")
    def test_model_not_compiled(self, mock_status, mock_compile, mock_mkdir):
        model_id = "compiling_model"
        mock_status.return_value = CompilationStatus.NOT_FOUND
        # Stop compilation from happening
        mock_compile.new = lambda x, y, z: None

        model_file, input_file = get_dummy_model_and_inputs("model", "inputs")
        response = client.post(f"/submit/{model_id}", files={"model": model_file, "inputs": input_file})

        self.assertEqual(response.status_code, HTTPStatus.OK)
        mock_mkdir.assert_called_once()
        mock_compile.assert_called_once()


class TestDownloadHandler(TestCase):
    @patch("os.path.isfile", new=lambda x: False)
    def test_download_handler_invalid_model_id(self):
        model_id = "invalid_model_id"

        response = client.get(f"/download/{model_id}")
        self.assertEqual(response.status_code, HTTPStatus.NOT_FOUND)

    @patch("os.path.isfile", new=lambda x: True)
    @patch("centml.compiler.server.FileResponse")
    def test_download_handler_success(self, mock_file_response):
        model_id = "valid_model_id"

        response = client.get(f"/download/{model_id}")

        self.assertEqual(response.status_code, HTTPStatus.OK)
