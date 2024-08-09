import ast
import csv
from typing import Dict

import torch
import torch.fx
from sklearn.neighbors import KDTree  # type: ignore
from torch.fx.node import Node


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
            print("Key not found")
            return None

        _, val = self.db[key].query(inp)
        return val


def populate_db(csv_file, database):
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = (row['op'], int(row['dim']), row['gpu'], row['dtype'])
                points = ast.literal_eval(row['points'])
                values = ast.literal_eval(row['values'])
                database.add_from_db(key, points, values)
            except ValueError as e:
                print(f"Error parsing row: {row}")
                print(e)


data = './sample_data.csv'


class Profiler:
    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())
        self.total_time = 0
        self.TreeDB = TreeDB()
        populate_db(data, self.TreeDB)

    def propagate(self, *args):
        args_iter = iter(args)
        env: Dict[str, Node] = {}

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

            for arg in args:
                if isinstance(arg, (tuple, list)):
                    if len(arg) > 0 and isinstance(arg[0], (tuple, list, torch.Tensor)):
                        shape = [len(arg)] + get_flattened_shapes(arg[0])
                    else:
                        shape = [len(arg)]
                elif isinstance(arg, torch.Tensor):
                    shape = list(arg.shape)
                elif isinstance(arg, bool):
                    shape = [1 if arg is True else 0]
                elif isinstance(arg, (int, float)):
                    shape = [arg]
                else:
                    shape = [1]
                flattened_shapes.extend(shape)

            if len(flattened_shapes) < 2:
                flattened_shapes.extend([1])

            return flattened_shapes

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
                inp_shapes = get_flattened_shapes(args)
                key = (node.target.__name__, len(inp_shapes), 'A10G', 'f16')
                t = self.TreeDB.get(key, inp_shapes)
                if t is not None:
                    self.total_time += t
            elif node.op == 'call_method':
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
                inp_shapes = get_flattened_shapes(args)
                key = (node.target, len(inp_shapes), 'A10G', 'f16')
                t = self.TreeDB.get(key, inp_shapes)
                if t is not None:
                    self.total_time += t
            elif node.op == 'call_module':
                mod = self.modules[node.target]
                args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = mod(*args, **kwargs)

                inp_shapes = get_flattened_shapes(args)
                param_shapes = [param.shape for name, param in mod.named_parameters()]
                flattened_params = [dim for shape in param_shapes for dim in shape]
                inp_shapes = inp_shapes + flattened_params
                key = (mod._get_name(), len(inp_shapes), 'A10G', 'f16')
                t = self.TreeDB.get(key, inp_shapes)
                if t is not None:
                    self.total_time += t
            elif node.op == 'output':
                args = load_arg(node.args)
                return args[0], self.total_time

            env[node.name] = result
