import torch._dynamo
import hidet
from .compiler import backend

# Configured to have server calculate and send dispatch table
hidet.option.store_dispatch_table(True)

# Register centml compiler backend to torch dynamo
if "centml" not in torch._dynamo.list_backends():
    torch._dynamo.register_backend(compiler_fn=backend.centml_dynamo_backend, name="centml")
