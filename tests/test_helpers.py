import torch
from transformers import BertForPreTraining, AutoTokenizer

@staticmethod
def get_graph_module_for_conv_model(model, inputs):
    return torch.fx.symbolic_trace(model)


@staticmethod
def get_graph_module_for_llm(model, inputs):
    # Use wrapper to specify to tracer that model uses input_ids and not input_embeds
    class RobertaWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids):
            return self.model(input_ids=input_ids, attention_mask=None)

    wrapper_model = RobertaWrapper(model)

    bert_inputs = {'input_ids': inputs}

    # Create the GraphModule
    tracer = torch.fx.Tracer()
    graph = tracer.trace(wrapper_model, concrete_args=bert_inputs)
    return torch.fx.GraphModule(wrapper_model, graph)


model_suite = {
    "resnet18":{
        "model": torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True, verbose=False).eval(),
        "inputs": [torch.zeros(1, 3, 224, 224)],
        "get_graph_module": get_graph_module_for_conv_model,
    },
    "bert-base-uncased": {
        "model": BertForPreTraining.from_pretrained("bert-base-uncased", ignore_mismatched_sizes=True).eval(),
        "inputs": AutoTokenizer.from_pretrained("bert-base-uncased")("Hello, my dog is cute", padding='max_length', return_tensors="pt")['input_ids'],
        "get_graph_module": get_graph_module_for_llm,
    },
}