import os
import tempfile
import pickle
from unittest import TestCase
from unittest.mock import MagicMock, patch
from http import HTTPStatus
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
from torch.fx import GraphModule
from fastapi import UploadFile
from fastapi.testclient import TestClient
from centml.compiler.server import app, background_compile
from centml.compiler.server_compilation import CompilationStatus

client = TestClient(app=app)


class TestStatusHandler(TestCase):
    def test_empty_request(self):
        response = client.get("/status/")
        self.assertEqual(response.status_code, HTTPStatus.NOT_FOUND)

    @patch("os.path.isdir")
    def test_model_not_found(self, mock_dir):
        model_id = "nonexistent_model"
        mock_dir.return_value = False

        response = client.get(f"/status/{model_id}")
        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertEqual(response.json(), {"status": CompilationStatus.NOT_FOUND.value})

    @patch("os.path.isfile")
    @patch("os.path.isdir")
    def test_model_compiling(self, mock_dir, mock_file):
        model_id = "compiling_model"
        mock_dir.return_value = True
        mock_file.return_value = False

        response = client.get(f"/status/{model_id}")
        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertEqual(response.json(), {"status": CompilationStatus.COMPILING.value})

    @patch("os.path.isfile")
    @patch("os.path.isdir")
    def test_model_done(self, mock_dir, mock_file):
        model_id = "completed_model"
        mock_dir.return_value = True
        mock_file.return_value = True

        response = client.get(f"/status/{model_id}")
        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertEqual(response.json(), {"status": CompilationStatus.DONE.value})


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

    @patch("logging.Logger.exception")
    def test_successful_compilation_resnet(self, mock_logger):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True, verbose=False).eval()
        graph_module: GraphModule = torch.fx.symbolic_trace(model)

        inputs = [torch.zeros(1, 3, 224, 224)]
        model_id = "successful_model_resnet"

        with tempfile.NamedTemporaryFile() as model_file, tempfile.NamedTemporaryFile() as input_file:
            # testFxGraph is a saved torch.fx.GraphModule representation of resnet18
            pickle.dump(graph_module, model_file)
            model_file.flush()
            model_file.seek(0)

            pickle.dump(inputs, input_file)
            input_file.flush()
            input_file.seek(0)

            background_compile(model_id, model_file, input_file)

        mock_logger.assert_not_called()

    @patch("logging.Logger.exception")
    def test_successful_compilation_roberta(self, mock_logger):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Use wrapper to specify to tracer that model uses input_ids and not input_embeds
        class RobertaWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids):
                return self.model(input_ids=input_ids, attention_mask=None)

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = (
            BertForSequenceClassification.from_pretrained(
                "bert-base-uncased", max_position_embeddings=8192, ignore_mismatched_sizes=True
            )
            .eval()
            .to('cuda')
        )
        wrapper_model = RobertaWrapper(model)

        # Example input for tracing
        inputs = tokenizer("Hello, my dog is cute", padding='max_length', max_length=4096, return_tensors="pt")
        inputs = {'input_ids': inputs['input_ids'].cuda()}

        # Create the GraphModule
        tracer = torch.fx.Tracer()
        graph = tracer.trace(wrapper_model, concrete_args=inputs)
        graph_module = torch.fx.GraphModule(wrapper_model, graph)

        # Formatting for hidet
        inputs = [i.clone().cuda() for i in inputs.values()]
        model_id = "successful_model_roberta"

        # Save model and inputs to files
        with tempfile.NamedTemporaryFile() as model_file, tempfile.NamedTemporaryFile() as input_file:
            pickle.dump(graph_module, model_file)
            model_file.flush()
            model_file.seek(0)

            pickle.dump(inputs, input_file)
            input_file.flush()
            input_file.seek(0)

            background_compile(model_id, model_file, input_file)

        mock_logger.assert_not_called()


class TestCompileHandler(TestCase):
    def setUp(self) -> None:
        self.model_id = "compiling_model"
        self.model = tempfile.NamedTemporaryFile()
        self.inputs = tempfile.NamedTemporaryFile()
        self.model.write(b"model")
        self.inputs.write(b"inputs")
        self.model.seek(0)
        self.inputs.seek(0)

    @patch("os.path.isdir")
    def test_model_compiling(self, mock_path):
        model_id = "compiling_model"
        mock_path.return_value = True

        response = client.post(f"/submit/{model_id}", files={"model": self.model, "inputs": self.inputs})
        self.assertEqual(response.status_code, HTTPStatus.OK)

    @patch("os.makedirs")
    @patch("centml.compiler.server.background_compile")
    @patch("os.path.isdir")
    def test_model_not_compiled(self, mock_path, mock_compile, mock_mkdir):
        model_id = "compiling_model"
        mock_path.return_value = False
        mock_compile.return_value = None

        response = client.post(f"/submit/{model_id}", files={"model": self.model, "inputs": self.inputs})

        self.assertEqual(response.status_code, HTTPStatus.OK)

        mock_mkdir.assert_called_once()
        mock_compile.assert_called_once()


class TestDownloadHandler(TestCase):
    @patch("os.path.isfile")
    def test_download_handler_invalid_model_id(self, mock_isfile):
        model_id = "invalid_model_id"

        mock_isfile.return_value = False

        response = client.get(f"/download/{model_id}")
        self.assertEqual(response.status_code, HTTPStatus.NOT_FOUND)

    @patch("os.path.isfile")
    @patch("centml.compiler.server.FileResponse")
    def test_download_handler_success(self, mock_file_response, mock_isfile):
        model_id = "valid_model_id"

        mock_isfile.return_value = True

        response = client.get(f"/download/{model_id}")

        self.assertEqual(response.status_code, HTTPStatus.OK)
