import os
import tempfile
import pickle
import warnings
from unittest import TestCase
from unittest.mock import MagicMock, patch
from http import HTTPStatus
import pytest
import torch
import hidet
from fastapi import UploadFile
from fastapi.testclient import TestClient
from parameterized import parameterized_class
from centml.compiler.server import app, background_compile
from centml.compiler.config import CompilationStatus
from .test_helpers import model_suite

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


@parameterized_class(list(model_suite.values()))
class TestBackgroundCompile(TestCase):
    @patch("logging.Logger.exception")
    def test_mock_cant_read(self, mock_logger):
        model_id = "file_cant_be_read"

        mock_file_content = MagicMock()
        mock_file_content.read.side_effect = Exception("an exception occurred")

        mock_upload_file = MagicMock(spec=UploadFile)
        mock_upload_file.file = mock_file_content

        background_compile(model_id, mock_upload_file, mock_upload_file)

        mock_logger.assert_called_once()
        log_message = mock_logger.call_args[0][0]

        self.assertIn("error reading serialized content", log_message)

    @patch("logging.Logger.exception")
    def test_model_empty_file(self, mock_logger):
        model_id = "empty_model"

        with tempfile.NamedTemporaryFile() as zero_byte_file:
            background_compile(model_id, zero_byte_file, zero_byte_file)

            mock_logger.assert_called_once()
            log_message = mock_logger.call_args[0][0]

            self.assertIn("error loading pickled content", log_message)

    @pytest.mark.gpu
    @patch("centml.compiler.server_compilation.save_compiled_graph")
    @patch("logging.Logger.exception")
    @patch("threading.Thread.start", new=lambda x: None)
    def test_successful_compilation(self, mock_logger, mock_save_cgraph):
        # For some reason there is a deadlock with parallel builds
        hidet.option.parallel_build(False)
        warnings.filterwarnings("ignore", category=UserWarning)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        model = self.model.cuda()
        inputs = self.inputs.cuda()

        def call_default_forward(_self, *args, **kwargs):
            return model.forward(*args, **kwargs)

        with patch('centml.compiler.backend.Runner.__init__', return_value=None) as mock_init, patch(
            'centml.compiler.backend.Runner.__call__', new=call_default_forward
        ):
            model_compiled = torch.compile(model, backend="centml")
            model_compiled(inputs)
            gm = mock_init.call_args[0][0]

        model_id = "successful_model"
        with tempfile.NamedTemporaryFile() as model_file, tempfile.NamedTemporaryFile() as input_file:
            pickle.dump(gm, model_file)
            model_file.flush()
            model_file.seek(0)

            pickle.dump([inputs], input_file)
            input_file.flush()
            input_file.seek(0)

            background_compile(model_id, model_file, input_file)

        mock_logger.assert_not_called()
        mock_save_cgraph.assert_called_once()


class TestCompileHandler(TestCase):
    def setUp(self) -> None:
        self.model_id = "compiling_model"
        self.model = tempfile.NamedTemporaryFile()  # pylint: disable=consider-using-with
        self.inputs = tempfile.NamedTemporaryFile()  # pylint: disable=consider-using-with
        self.model.write(b"model")
        self.inputs.write(b"inputs")
        self.model.seek(0)
        self.inputs.seek(0)

    def tearDown(self) -> None:
        self.model.close()
        self.inputs.close()

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
