import torch


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
