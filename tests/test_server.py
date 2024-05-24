import shutil
import tempfile
import warnings
from io import BytesIO
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
from centml.compiler.config import CompilationStatus, config_instance
from tests.test_helpers import MODEL_SUITE

client = TestClient(app=app)


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
    @patch("logging.Logger.exception")
    @patch("centml.compiler.server.torch.save")
    @patch("threading.Thread.start", new=lambda x: None)
    def test_successful_compilation(self, mock_save, mock_logger):
        # For some reason there is a deadlock with parallel builds
        hidet.option.parallel_build(False)
        warnings.filterwarnings("ignore", category=UserWarning)

        model = self.model.cuda()
        inputs = self.inputs.cuda()
        
        # Get the graph_module and example inputs that would be passed to background compile
        def mock_init(self, module, inputs):
            # set this so Runner.__del__ doesn't throw an exception
            self.serialized_model_dir = "fake_path"
            global graph_module
            global example_inputs
            graph_module, example_inputs = module, inputs

        with patch('centml.compiler.backend.Runner.__init__', new=mock_init), \
        patch('centml.compiler.backend.Runner.__call__', new=model.forward):
            model_compiled = torch.compile(model, backend="centml")
            model_compiled(inputs)

        model_id = "successful_model"
        background_compile(model_id, graph_module, example_inputs)

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
        model_file, inputs_file = BytesIO(), BytesIO()
        torch.save("model", model_file)
        torch.save("inputs", inputs_file)
        model_file.seek(0)
        inputs_file.seek(0)

        model = UploadFile(filename="model", file=model_file)
        inputs = UploadFile(filename="inputs", file=inputs_file)

        with self.assertRaises(HTTPException) as excinfo:
            read_upload_files(model_id, model, inputs)

        mock_load.assert_called_once()
        self.assertEqual(excinfo.exception.status_code, HTTPStatus.BAD_REQUEST)
        self.assertIn("Compilation: error loading content with torch.load:", str(excinfo.exception))

    def test_proper_read(self):
        model_id = "test_model_id"

        model_data, inputs_data = "model", "inputs"

        # Create file-like objects with test data
        model_file, inputs_file = BytesIO(), BytesIO()
        torch.save(model_data, model_file)
        torch.save(inputs_data, inputs_file)
        model_file.seek(0)
        inputs_file.seek(0)

        model = UploadFile(filename="model", file=model_file)
        inputs = UploadFile(filename="inputs", file=inputs_file)

        tfx_graph, example_inputs = read_upload_files(model_id, model, inputs)

        self.assertEqual(tfx_graph, model_data)
        self.assertEqual(example_inputs, inputs_data)


class TestCompileHandler(TestCase):
    def setUp(self) -> None:
        self.model = BytesIO()
        self.inputs = BytesIO()
        torch.save("model", self.model)
        torch.save("inputs", self.inputs)
        self.model.seek(0)
        self.inputs.seek(0)

    @patch("centml.compiler.server.get_status")
    def test_model_compiling(self, mock_status):
        model_id = "compiling_model"
        mock_status.return_value = CompilationStatus.COMPILING

        response = client.post(f"/submit/{model_id}", files={"model": self.model, "inputs": self.inputs})
        self.assertEqual(response.status_code, HTTPStatus.OK)

    @patch("os.makedirs")
    @patch("centml.compiler.server.background_compile")
    @patch("centml.compiler.server.get_status")
    def test_model_not_compiled(self, mock_status, mock_compile, mock_mkdir):
        model_id = "compiling_model"
        mock_status.return_value = CompilationStatus.NOT_FOUND
        # Stop compilation from happening
        mock_compile.new = lambda x, y, z: None

        response = client.post(f"/submit/{model_id}", files={"model": self.model, "inputs": self.inputs})

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

# if __name__ == "__main__":
#     t = TestBackgroundCompile_0()
#     t.test_successful_compilation()