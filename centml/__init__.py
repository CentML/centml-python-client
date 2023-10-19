import torch._dynamo

from .compiler import backend

# Register centml compiler backend to torch dynamo
torch._dynamo.register_backend(compiler_fn=backend.centml_dynamo_backend, name="centml")
