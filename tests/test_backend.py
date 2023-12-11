from unittest import TestCase
from unittest.mock import patch
import torch
from torch.fx import GraphModule
import hidet
from hidet.graph.frontend.torch.dynamo_backends import get_flow_graph
from centml.compiler.backend import Runner


class TestGetModelId(TestCase):
    def get_graph_module(self, name, device):
        model = torch.hub.load('pytorch/vision:v0.10.0', name, pretrained=True, verbose=False).eval().to(device)
        graph_module: GraphModule = torch.fx.symbolic_trace(model)
        interpreter = hidet.frontend.from_torch(graph_module)
        return graph_module, interpreter

    @patch('threading.Thread.start')
    def setUp(self, mock_thread) -> None:
        graph_module, interpreter = self.get_graph_module('resnet18', 'cpu')
        graph_module_34, interpreter_34 = self.get_graph_module('resnet34', 'cuda')
        self.inputs = [torch.zeros(1, 3, 224, 224)]
        self.inputs_34 = [torch.zeros(1, 3, 224, 224).cuda()]
        self.flow_graph, _, _ = get_flow_graph(interpreter, self.inputs)
        self.flow_graph_34, _, _ = get_flow_graph(interpreter_34, self.inputs_34)
        self.runner = Runner(graph_module, None)
        self.runner_34 = Runner(graph_module_34, None)

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

    @patch('hidet.save_graph')
    def test_output_consistency(self, mock_save_graph):
        model_id1 = self.runner._get_model_id(self.flow_graph)
        model_id2 = self.runner._get_model_id(self.flow_graph)
        self.assertEqual(model_id1, model_id2)
        self.assertEqual(mock_save_graph.call_count, 2)

    @patch('hidet.save_graph')
    def test_output_uniqueness(self, mock_save_graph):
        model_id1 = self.runner._get_model_id(self.flow_graph)
        model_id2 = self.runner_34._get_model_id(self.flow_graph_34)
        self.assertNotEqual(model_id1, model_id2)
        self.assertEqual(mock_save_graph.call_count, 2)


if __name__ == '__main__':
    t = TestGetModelId()
    t.setUp()
