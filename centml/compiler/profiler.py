import ast
import csv
import logging
from typing import Dict

import torch
import torch.fx
from sklearn.neighbors import KDTree  # type: ignore
from torch.fx.node import Node

from centml.compiler.config import settings


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


class TreeDB:
    def __init__(self):
        self.db = {}

    def add_from_db(self, key, points, times):
        if key not in self.db:
            self.db[key] = {}
        self.db[key] = KDTreeWithValues(points, times)

    def get(self, key, inp):
        if key not in self.db:
            logging.getLogger(__name__).warning(f"Key {key} not found in database")
            return None

        _, val = self.db[key].query(inp)
        return val


def populate_db(csv_file, database):
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = (row['op'], int(row['dim']), row['inp_dtypes'], row['out_dtypes'], row['gpu'])
                points = ast.literal_eval(row['points'])
                values = ast.literal_eval(row['values'])
                database.add_from_db(key, points, values)
            except ValueError as e:
                logging.getLogger(__name__).exception(f"Error parsing row: {row}\n{e}")


class Profiler:
    def __init__(self, mod, gpu):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())
        self.total_time = 0
        self.TreeDB = TreeDB()
        self.gpu = gpu
        populate_db(settings.PREDICTION_DATA_DIR, self.TreeDB)

    def propagate(self, *args):
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
                    shape = [1 if arg is True else 0]
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
            result = None
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':
                result = fetch_attr(node.target)
            elif node.op == 'call_function':
                args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = node.target(*args, **kwargs)

                inp_shapes, input_dtypes = get_flattened_shapes(args)
                output_dtypes = get_output_dtypes(result)

                key = (node.target.__name__, len(inp_shapes), input_dtypes, output_dtypes, self.gpu)

                t = self.TreeDB.get(key, inp_shapes)
                if t is not None:
                    self.total_time += t
            elif node.op == 'call_method':
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)

                inp_shapes, input_dtypes = get_flattened_shapes(args)
                output_dtypes = get_output_dtypes(result)

                key = (node.target, len(inp_shapes), input_dtypes, output_dtypes, self.gpu)

                t = self.TreeDB.get(key, inp_shapes)
                if t is not None:
                    self.total_time += t
            elif node.op == 'call_module':
                mod = self.modules[node.target]
                args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = mod(*args, **kwargs)

                inp_shapes, input_dtypes = get_flattened_shapes(args)
                param_shapes = [param.shape for name, param in mod.named_parameters()]
                param_dtypes = [dtypeMap[param.dtype] for name, param in mod.named_parameters()]
                flattened_params = [dim for shape in param_shapes for dim in shape]
                inp_shapes = inp_shapes + flattened_params
                input_dtypes = input_dtypes + ',' + ','.join(param_dtypes)
                output_dtypes = get_output_dtypes(result)

                key = (mod._get_name(), len(inp_shapes), input_dtypes, output_dtypes, self.gpu)
                t = self.TreeDB.get(key, inp_shapes)
                if t is not None:
                    self.total_time += t
            elif node.op == 'output':
                args = load_arg(node.args)
                return args[0], self.total_time

            env[node.name] = result
