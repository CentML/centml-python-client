from typing import Dict

import torch
import torch.fx
from torch.fx.node import Node


class Profiler:
    def __init__(self, mod, gpu, treeDB, data_collection_mode=False):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())
        self.tree_db = treeDB
        self.gpu = gpu
        self.data_collection_mode = data_collection_mode

    def propagate(self, *args):
        args_iter = iter(args)
        env: Dict[str, Node] = {}
        total_time = 0

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
                    dtypes.append(str(arg.dtype))
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
                    return [str(results.dtype)]
                if isinstance(results, (list, tuple)):
                    dtypes = []
                    for item in results:
                        dtypes.extend(find_dtypes(item))
                    return dtypes
                return []

            types = find_dtypes(results)

            if types:
                return ','.join(types)
            return 'N/A'

        def get_time_or_profile(key, inp_shapes, operation, *args, **kwargs):
            t = self.tree_db.get(key, inp_shapes)

            if self.data_collection_mode and t is None:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                    operation(*args, **kwargs)
                event_time_total = 0
                for event in prof.key_averages():
                    event_time_total += event.cuda_time_total
                t = event_time_total
                self.tree_db.add(key, inp_shapes, t)

            return t

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

                t = get_time_or_profile(key, inp_shapes, node.target, *args, **kwargs)

                total_time += t
            elif node.op == 'call_method':
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)

                inp_shapes, input_dtypes = get_flattened_shapes(args)
                output_dtypes = get_output_dtypes(result)

                key = (node.target, len(inp_shapes), input_dtypes, output_dtypes, self.gpu)

                t = get_time_or_profile(key, inp_shapes, getattr(self_obj, node.target), *args, **kwargs)

                total_time += t
            elif node.op == 'call_module':
                mod = self.modules[node.target]
                args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = mod(*args, **kwargs)

                inp_shapes, input_dtypes = get_flattened_shapes(args)

                param_shapes = [param.shape for name, param in mod.named_parameters()]
                param_dtypes = [str(param.dtype) for name, param in mod.named_parameters()]
                flattened_params = [dim for shape in param_shapes for dim in shape]

                inp_shapes = inp_shapes + flattened_params
                input_dtypes = input_dtypes + ',' + ','.join(param_dtypes)

                output_dtypes = get_output_dtypes(result)

                key = (mod._get_name(), len(inp_shapes), input_dtypes, output_dtypes, self.gpu)

                t = get_time_or_profile(key, inp_shapes, mod, *args, **kwargs)

                total_time += t
            elif node.op == 'output':
                args = load_arg(node.args)
                return args[0], total_time

            env[node.name] = result
