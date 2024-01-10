import torch
from transformers import BertForPreTraining, AutoTokenizer

MODEL_SUITE = {
    "resnet18": {
        "model": torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True, verbose=False).eval(),
        "inputs": torch.zeros(1, 3, 224, 224),
    },
    "bert-base-uncased": {
        "model": BertForPreTraining.from_pretrained("bert-base-uncased", ignore_mismatched_sizes=True).eval(),
        "inputs": AutoTokenizer.from_pretrained("bert-base-uncased")(
            "Hello, my dog is cute", padding='max_length', return_tensors="pt"
        )['input_ids'],
    },
}
