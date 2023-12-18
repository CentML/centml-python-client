import hidet
import torch._dynamo
from .compiler import backend

hidet.option.imperative(False)
hidet.option.debug_enable_var_id(False)

# Register centml compiler backend to torch dynamo
if "centml" not in torch._dynamo.list_backends():
    torch._dynamo.register_backend(compiler_fn=backend.centml_dynamo_backend, name="centml")
