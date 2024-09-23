import csv
import gc
import json

import torch
import torchvision.models as models
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from centml.compiler.prediction.kdtree import KDTreeWithValues
from centml.compiler.prediction.profiler import Profiler
from scripts.timer import timed

torch.set_float32_matmul_precision('high')
torch.set_default_device('cuda')
torch.set_default_dtype(torch.float16)

CURR_GPU = "A10G"
OUTPUT_FILE = 'data.csv'

# Different HuggingFace Models + Different Input Sizes
hf_model_tests = [
    ("EleutherAI/gpt-neo-2.7B", (1, 512)),
    ("gpt2-xl", (1, 1024)),
    ("gpt2-large", (1, 1024)),
    ("gpt2-xl", (1, 512)),
    ("google-bert/bert-large-uncased", (8, 512)),
    ("google-bert/bert-large-uncased", (16, 512)),
    ("meta-llama/Meta-Llama-3.1-8B", (1, 256)),
    ("gpt2-medium", (1, 1024)),
    ("facebook/bart-large", (1, 1024)),
    ("google/pegasus-cnn_dailymail", (1, 1024)),
]

# GPU bound tests for A100 80GB
# A10080GB_hf_model_tests = [
#     ("EleutherAI/gpt-neo-2.7B", (1, 1024)),
#     ("gpt2-xl", (2, 1024)),
#     ("meta-llama/Meta-Llama-3.1-8B", (1, 1024)),
#     ("google/pegasus-cnn_dailymail", (4, 1024)),
#     ("facebook/bart-large", (4, 1024)),
#     ("google-bert/bert-large-uncased", (16, 512)),
#     ("gpt2-medium", (2, 1024)),
#     ("gpt2-large", (2, 1024)),
# ]

# Different Batch Sizes for each ResNet Model (torchvision)
resnet_tests = [512, 720]


def percent_error(observed, true):
    return abs((observed - true) / true) * 100


class DataCollectionTreeDB:
    def __init__(self):
        self.db = {}

    def add(self, key, point, time):
        if key not in self.db:
            self.db[key] = KDTreeWithValues()

        self.db[key].add(point, time)

    def get(self, key, inp):
        if key not in self.db:
            # print("New Key")
            return None

        dist, val = self.db[key].query(inp)

        if dist > 0:
            # print("Distance too large ", dist)
            return None
        return val


db = DataCollectionTreeDB()
cuda_kernel_time = 0
actual_time = 0


def custom_backend(gm: torch.fx.GraphModule, inps):
    print("Compiling")
    profiler = Profiler(mod=gm, gpu=CURR_GPU, treeDB=db, data_collection_mode=True)

    def forward(*args):
        global cuda_kernel_time
        global actual_time
        out, t, actual_t = profiler.propagate(*args)
        cuda_kernel_time += t
        actual_time += actual_t
        return out

    return forward


def hf_model_test(model_name, input_size, custom_backend):
    global cuda_kernel_time
    global actual_time
    models_without_tokenizer = {"google/pegasus-cnn_dailymail"}

    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda:0")
    if model_name not in models_without_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    inp = torch.randint(
        low=0,
        high=tokenizer.vocab_size if model_name not in models_without_tokenizer else 50265,
        size=input_size,
        dtype=torch.int64,
        device='cuda:0',
    )

    with torch.inference_mode():
        for _ in range(10):
            _, t = timed(lambda: model(inp))
            print(t)

    compiled_model = torch.compile(model, backend=custom_backend)
    compiled_model(inp)

    cuda_kernel_time /= 1000000

    print(f"{model_name}, {input_size}")
    print("Real time: ", actual_time)
    print("Kernel execution time: ", cuda_kernel_time)
    print("Error: ", percent_error(cuda_kernel_time, actual_time))

    cuda_kernel_time = 0
    actual_time = 0
    del model, inp, compiled_model
    gc.collect()
    torch.cuda.empty_cache()


def resnet_test(batch_size, custom_backend):
    global cuda_kernel_time
    global actual_time
    model = models.resnet50(weights=True, num_classes=1000).cuda()
    model.eval()
    inp = torch.randn(batch_size, 3, 128, 128).cuda(0)

    with torch.inference_mode():
        for _ in range(10):
            _, t = timed(lambda: model(inp))
            print(t)

    compiled_model = torch.compile(model, backend=custom_backend)
    compiled_model(inp)

    cuda_kernel_time /= 1000000

    print(f"resnet, ({batch_size}, 3, 128, 128)")
    print("Real time: ", actual_time)
    print("TOTAL TIME: ", cuda_kernel_time)
    print("Error: ", percent_error(cuda_kernel_time, actual_time))

    cuda_kernel_time = 0
    actual_time = 0
    del model, inp, compiled_model
    gc.collect()
    torch.cuda.empty_cache()



for batch_size in resnet_tests:
    resnet_test(batch_size, custom_backend)

# Write to CSV
with open(OUTPUT_FILE, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    csvwriter.writerow(['op', 'dim', 'inp_dtypes', 'out_dtypes', 'gpu', 'points', 'values'])

    for key, tree in db.db.items():
        op, dim, inp_dtypes, out_dtype, gpu = key
        points_str = json.dumps(tree.points)
        values_str = json.dumps(tree.values)
        csvwriter.writerow([op, dim, inp_dtypes, out_dtype, gpu, points_str, values_str])
