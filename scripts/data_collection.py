import csv
import gc
import json
import os
import random
import statistics
import time

import numpy as np
import torch
import torchvision.models as models
from sklearn.neighbors import KDTree
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    GPT2ForSequenceClassification,
    PegasusConfig,
    PegasusForCausalLM,
)

torch.set_float32_matmul_precision('high')
torch.set_default_device('cuda')
torch.set_default_dtype(torch.float16)

CURR_GPU = "A10G"


def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


def percent_error(observed, true):
    return abs((observed - true) / true) * 100


class KDTreeWithValues:
    def __init__(self, points=None, values=None):
        self.points = points if points else []
        self.values = values if values else []
        if self.points:
            self.tree = KDTree(self.points)
        else:
            self.tree = None

    def add(self, point, value):
        self.points.append(point)
        self.values.append(value)
        self.tree = KDTree(self.points)

    def query(self, point):
        if self.tree is None:
            return None, None

        dist, idx = self.tree.query([point], k=1)
        return dist[0][0], self.values[idx[0][0]]


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


class ShapeProp:
    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())

    def propagate(self, *args):
        global db

        total_time = 0

        args_iter = iter(args)
        env: Dict[str, Node] = {}

        dtypeMap = {
            torch.float32: 'f32',
            torch.float: 'f32',
            torch.float64: 'f64',
            torch.double: 'f64',
            torch.float16: 'f16',
            torch.half: 'f16',
            torch.bfloat16: 'bf16',
            torch.complex32: 'c32',
            torch.chalf: 'c32',
            torch.complex64: 'c64',
            torch.cfloat: 'c64',
            torch.complex128: 'c128',
            torch.cdouble: 'c128',
            torch.uint8: 'u8',
            torch.uint16: 'u16',
            torch.uint32: 'u32',
            torch.uint64: 'u64',
            torch.int8: 'i8',
            torch.int16: 'i16',
            torch.short: 'i16',
            torch.int32: 'i32',
            torch.int: 'i32',
            torch.int64: 'i64',
            torch.long: 'i64',
            torch.bool: 'bool',
            torch.quint8: 'qu8',
            torch.qint8: 'qi8',
            torch.qint32: 'qi32',
            torch.quint4x2: 'qu4x2',
        }

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target: str):
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        def get_flattened_shapes(args):
            flattened_shapes = []
            dtypes = []

            for arg in args:
                if isinstance(arg, (tuple, list)):
                    if len(arg) > 0 and isinstance(arg[0], (tuple, list, torch.Tensor)):
                        nested_shapes, nested_dtypes = get_flattened_shapes(arg[0])
                        shape = [len(arg)] + nested_shapes
                        dtypes.extend(nested_dtypes.split(','))
                    else:
                        shape = [len(arg)]
                elif isinstance(arg, torch.Tensor):
                    shape = list(arg.shape)
                    dtypes.append(dtypeMap[arg.dtype])
                elif isinstance(arg, bool):
                    shape = [1 if arg == True else 0]
                elif isinstance(arg, (int, float)):
                    shape = [arg]
                else:
                    shape = [1]
                flattened_shapes.extend(shape)

            if len(flattened_shapes) < 2:
                flattened_shapes.extend([1])

            input_dtypes = ','.join(dtypes) if dtypes else 'N/A'

            return flattened_shapes, input_dtypes

        def get_output_dtypes(results):
            def find_dtypes(results):
                if isinstance(results, torch.Tensor):
                    return [dtypeMap[results.dtype]]
                elif isinstance(results, (list, tuple)):
                    dtypes = []
                    for item in results:
                        dtypes.extend(find_dtypes(item))
                    return dtypes
                return []

            types = find_dtypes(results)

            if types:
                return ','.join(types)
            return 'N/A'

        for node in self.graph.nodes:
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':
                result = fetch_attr(node.target)
            elif node.op == 'call_function':
                args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                inp_shapes, input_dtypes = get_flattened_shapes(args)
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    result = node.target(*args, **kwargs)
                output_dtypes = get_output_dtypes(result)

                key = (node.target.__name__, len(inp_shapes), input_dtypes, output_dtypes, CURR_GPU)

                t = db.get(key, inp_shapes)

                if t is None:
                    new_time = 0
                    for x in prof.key_averages():
                        new_time += x.cuda_time_total
                    total_time += new_time
                    db.add(key, inp_shapes, new_time)
                else:
                    total_time += t

            elif node.op == 'call_method':
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                inp_shapes, input_dtypes = get_flattened_shapes(args)
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    result = getattr(self_obj, node.target)(*args, **kwargs)
                output_dtypes = get_output_dtypes(result)

                key = (node.target, len(inp_shapes), input_dtypes, output_dtypes, CURR_GPU)

                t = db.get(key, inp_shapes)

                if t is None:
                    new_time = 0
                    for x in prof.key_averages():
                        new_time += x.cuda_time_total
                    total_time += new_time
                    db.add(key, inp_shapes, new_time)
                else:
                    total_time += t

            elif node.op == 'call_module':
                mod = self.modules[node.target]
                args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                inp_shapes, input_dtypes = get_flattened_shapes(args)
                param_shapes = [param.shape for name, param in mod.named_parameters()]
                param_dtypes = [dtypeMap[param.dtype] for name, param in mod.named_parameters()]
                flattened_params = [dim for shape in param_shapes for dim in shape]
                inp_shapes = inp_shapes + flattened_params
                input_dtypes = input_dtypes + ',' + ','.join(param_dtypes)
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    result = mod(*args, **kwargs)

                output_dtypes = get_output_dtypes(result)

                key = (mod._get_name(), len(inp_shapes), input_dtypes, output_dtypes, CURR_GPU)

                t = db.get(key, inp_shapes)

                if t is None:
                    new_time = 0
                    for x in prof.key_averages():
                        new_time += x.cuda_time_total
                    total_time += new_time
                    db.add(key, inp_shapes, new_time)
                else:
                    total_time += t

            if isinstance(result, torch.Tensor):
                node.shape = result.shape
                node.dtype = result.dtype

            env[node.name] = result

        return total_time / 1000000


added_time = 0


def custom_backend(gm: torch.fx.GraphModule, inps):
    print("Compiling")
    shape = ShapeProp(gm)
    t = shape.propagate(*inps)

    def forward(*args):
        global added_time
        added_time += t
        return gm.forward(*args)

    return forward


def model_test(model_name, input_size, custom_backend):
    global added_time
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda:0")
    if model_name not in {"google/pegasus-cnn_dailymail"}:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    inp = torch.randint(
        low=0,
        high=tokenizer.vocab_size if model_name not in {"google/pegasus-cnn_dailymail"} else 50265,
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
    print(f"{model_name}, {input_size}")
    print("Real time: ", t)
    print("TOTAL TIME: ", added_time)
    print("Error: ", percent_error(added_time, t))

    added_time = 0
    del model, inp, compiled_model
    gc.collect()
    torch.cuda.empty_cache()


def resnet_test(batch_size, custom_backend):
    global added_time
    model = models.resnet50(weights=True, num_classes=1000).cuda()
    model.eval()
    inp = torch.randn(batch_size, 3, 128, 128).cuda(0)

    with torch.inference_mode():
        for _ in range(10):
            _, t = timed(lambda: model(inp))
            print(t)

    compiled_model = torch.compile(model, backend=custom_backend)
    compiled_model(inp)
    print(f"resnet, ({batch_size}, 3, 128, 128)")
    print("Real time: ", t)
    print("TOTAL TIME: ", added_time)
    print("Error: ", percent_error(added_time, t))

    added_time = 0
    del model, inp, compiled_model
    gc.collect()
    torch.cuda.empty_cache()


model_tests = [
    ("EleutherAI/gpt-neo-2.7B", (1, 512)),
    ("gpt2-xl", (1, 1024)),
    ("gpt2-large", (1, 1024)),
    ("gpt2-xl", (1, 512)),
    ("google-bert/bert-large-uncased", (8, 512)),
    ("google-bert/bert-large-uncased", (16, 512)),
    ("meta-llama/Meta-Llama-3.1-8B", (1, 512)),
    ("meta-llama/Meta-Llama-3.1-8B", (1, 256)),
    ("gpt2-medium", (1, 1024)),
    ("facebook/bart-large", (1, 1024)),
    ("google/pegasus-cnn_dailymail", (1, 1024)),
]

for model_name, input_size in model_tests:
    model_test(model_name, input_size, custom_backend)

resnet_tests = [1024, 720, 1440]
for batch_size in resnet_tests:
    resnet_test(batch_size, custom_backend)


with open('data.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    csvwriter.writerow(['op', 'dim', 'inp_dtypes', 'out_dtypes', 'gpu', 'points', 'values'])

    for key, tree in db.db.items():
        op, dim, inp_dtypes, out_dtype, gpu = key
        points_str = json.dumps(tree.points)
        values_str = json.dumps(tree.values)
        csvwriter.writerow([op, dim, inp_dtypes, out_dtype, gpu, points_str, values_str])
