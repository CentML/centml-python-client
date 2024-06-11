# centml-python-client
![](https://github.com/CentML/centml-python-client/actions/workflows/unit_tests.yml/badge.svg)

### Installation
First, ensure you meet the requirements for  [Hidet](https://github.com/hidet-org/hidet), namely:
- CUDA Toolkit 11.6+
- Python 3.8+

Inside the project's base directory, run the following commands:
```bash
pip install . 
```

### Compilation

To use the compilation feature, make sure to install Hidet:
```bash
pip install hidet
```

Then, within your python script include the following:
```python
import torch
# This will import the "centml" torch.compile backend
import centml.compiler  

# Define these yourself
model = ...
inputs = ...

# Pass the "centml" backend
compiled_model = torch.compile(model, backend="centml")
# Since torch.compile is JIT, compilation is only triggered when you call the model
output = compiled_model(inputs)
```

By default, the centml compiler will send the model to a server at the URL `http://0.0.0.0:8090`. \
You can change by setting the environment variables `CENTML_SERVER_IP` and `CENTML_SERVER_PORT`

### Tests
To run tests, first install required packages:
```bash
pip install -r requirements-dev.txt
cd tests
```

When running on a local machine, it is recommended to run tests with the following command. This skips tests that require a GPU.
```bash
pytest --sanity
```

To run all the tests, use:
```bash
pytest
```
