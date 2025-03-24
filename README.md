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

### Un-installation

To uninstall `centml`, simply do:
```bash
pip uninstall centml
```

### CLI
Once installed, use the centml CLI tool with the following command:
```bash
centml 
```

If you want tab completion, run
```bash
source scripts/completions/completion.<shell language>
```
Shell language can be: bash, zsh, fish
(Hint: add `source /path/to/completions/completion.<shell language>` to your `~/.bashrc`, `~/.zshrc` or `~/.config/fish/completions/centml.fish`)

### Compilation

centml-python-client's compiler feature allows you to compile your ML model remotely using the [hidet](https://hidet.org/docs/stable/index.html) backend. \
Thus, use the compilation feature, make sure to run:
```bash
pip install hidet
```

To run the server locally, you can use the following CLI command:
```bash
centml server
```
By default, the server will run at the URL `http://0.0.0.0:8090`. \
You can change this by setting the environment variable `CENTML_SERVER_URL`


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
Note that the centml backend compiler is non-blocking. This means it that until the server returns the compiled model, your python script will use the uncompiled model to generate the output.

Again, make sure your script's environment sets `CENTML_SERVER_URL` to communicate with the desired server.

To see logs, add this to your script before triggering compilation:
```python
logging.basicConfig(level=logging.INFO)
```

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

### Common Issues

- **`SSL` certificate on `MacOS`**

    Sometimes, you will see issues when using command like `centml cluster [CMD]`, where the output might look like:

    ```logs

    File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/urllib3/util/retry.py", line 519, in increment

    raise MaxRetryError(_pool, url, reason) from reason  # type: ignore[arg-type]

    urllib3.exceptions.MaxRetryError: HTTPSConnectionPool(host='api.centml.com', port=443):

    Max retries exceeded with url: /deployments

    (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1007)')))
    ```

    **Solution**:
    To fix this issue, navigate to your `python` installation directory and run the `Install Certificates.command` file located there.

    For example, if you are using `python3.10`, the file path would be:
    `
    /Applications/Python 3.10/Install Certificates.command
    `
