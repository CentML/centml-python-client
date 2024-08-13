import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import centml.compile

torch.set_default_device('cpu')
torch.set_float32_matmul_precision('high')
torch.set_default_dtype(torch.float16)

model_name = "gpt2-xl"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

inputs = torch.randint(low=0, high=tokenizer.vocab_size, size=(1, 512), dtype=torch.int64, device='cpu')

compiled_model = centml.compile(model)
output = compiled_model(inputs)

while True:
    time.sleep(1)
