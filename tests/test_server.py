import pickle
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
from centml.compiler.config import CompilationStatus
from .test_helpers import MODEL_SUITE

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
    @patch("centml.compiler.server_compilation.save_compiled_graph")
    @patch("logging.Logger.exception")
    @patch("threading.Thread.start", new=lambda x: None)
    def test_successful_compilation(self, mock_logger, mock_save_cgraph):
        # For some reason there is a deadlock with parallel builds
        hidet.option.parallel_build(False)
        warnings.filterwarnings("ignore", category=UserWarning)

        model = self.model.cuda()
        inputs = self.inputs.cuda()

        with patch('centml.compiler.backend.Runner.__init__', return_value=None) as mock_init, patch(
            'centml.compiler.backend.Runner.__call__', new=model.forward
        ):
            model_compiled = torch.compile(model, backend="centml")
            model_compiled(inputs)
            graph_module = mock_init.call_args[0][0]
            example_inputs = mock_init.call_args[0][1]

        model_id = "successful_model"
        background_compile(model_id, graph_module, example_inputs)

        mock_logger.assert_not_called()
        mock_save_cgraph.assert_called_once()


class TestReadUploadFiles(TestCase):
    def test_mock_cant_read(self):
        model_id = "file_cant_be_read"

        mock_file = MagicMock()
        mock_file.file.read.side_effect = Exception("an exception occurred")

        with self.assertRaises(HTTPException) as excinfo:
            read_upload_files(model_id, mock_file, mock_file)

        self.assertEqual(excinfo.exception.status_code, HTTPStatus.BAD_REQUEST)
        self.assertIn("Compilation: error reading serialized content", str(excinfo.exception))

    @patch("pickle.loads", side_effect=Exception("an exception occurred"))
    def test_cant_unpickle(self, mock_pickle_loads):
        model_id = "file_cant_be_unpickled"

        model_data = "model"
        inputs_data = "inputs"

        # Create file-like objects with any data
        model_file = BytesIO(pickle.dumps(model_data))
        inputs_file = BytesIO(pickle.dumps(inputs_data))

        model = UploadFile(filename="model", file=model_file)
        inputs = UploadFile(filename="inputs", file=inputs_file)

        with self.assertRaises(HTTPException) as excinfo:
            read_upload_files(model_id, model, inputs)

        mock_pickle_loads.assert_called_once()
        self.assertEqual(excinfo.exception.status_code, HTTPStatus.BAD_REQUEST)
        self.assertIn("error loading pickled content", str(excinfo.exception))

    def test_proper_read(self):
        model_id = "test_model_id"

        # Create file-like objects with pickleable data
        model_data = "model"
        inputs_data = "inputs"

        model_file = BytesIO(pickle.dumps(model_data))
        inputs_file = BytesIO(pickle.dumps(inputs_data))

        model = UploadFile(filename="model", file=model_file)
        inputs = UploadFile(filename="inputs", file=inputs_file)

        tfx_graph, example_inputs = read_upload_files(model_id, model, inputs)

        self.assertEqual(tfx_graph, model_data)
        self.assertEqual(example_inputs, inputs_data)


class TestCompileHandler(TestCase):
    def setUp(self) -> None:
        self.model = pickle.dumps("model")
        self.inputs = pickle.dumps("inputs")

    @patch("os.path.isdir", new=lambda x: True)
    def test_model_compiling(self):
        model_id = "compiling_model"

        response = client.post(f"/submit/{model_id}", files={"model": self.model, "inputs": self.inputs})
        self.assertEqual(response.status_code, HTTPStatus.OK)

    @patch("os.makedirs")
    @patch("centml.compiler.server.background_compile")
    @patch("os.path.isdir", new=lambda x: False)
    def test_model_not_compiled(self, mock_compile, mock_mkdir):
        model_id = "compiling_model"
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
