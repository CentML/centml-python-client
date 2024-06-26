import torch._dynamo

from centml.compiler.backend import centml_dynamo_backend


# Register centml compiler backend to torch dynamo
if "centml" not in torch._dynamo.list_backends():
    torch._dynamo.register_backend(compiler_fn=centml_dynamo_backend, name="centml")
