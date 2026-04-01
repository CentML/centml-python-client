# centml-python-client
![](https://github.com/CentML/centml-python-client/actions/workflows/unit_tests.yml/badge.svg)

### Installation

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
