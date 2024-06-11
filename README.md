# centml-python-client
![](https://github.com/CentML/centml-python-client/actions/workflows/unit_tests.yml/badge.svg)

### Installation
First, ensure you meet the requirements for  [Hidet](https://github.com/hidet-org/hidet), namely:
- CUDA Toolkit 11.6+
- Python 3.8+

To install without cloning, run the following command:
```bash
pip install git+https://github.com/CentML/centml-python-client.git@main
```

Alternatively to build from source, clone this repo then inside the project's base directory, run the following command:
```bash
pip install . 
```

### CLI
Once installed, use the centml CLI tool with the following command:
```bash
centml 
```

### Compilation

The centml remote compiler allows you to compile your ML model on an remote server using the [hidet](https://hidet.org/docs/stable/index.html) compiler. \
To run the server locally, you can use the following CLI command:
```bash
centml server
```
By default, the server will run at the URL `http://0.0.0.0:8090`. \
You can change this by setting the environment variables `CENTML_SERVER_IP` and `CENTML_SERVER_PORT`


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
# Since torch.compile is JIT, compilation is only triggered when you first call the model
output = compiled_model(inputs)
```
Make sure to set the environment variables `CENTML_SERVER_IP` and `CENTML_SERVER_PORT` to communicate with the desired server


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
