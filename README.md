# centml-python-client

### Installation
Inside the project's base directory, run the following commands:
```bash
pip install . 
pip install --pre --extra-index-url https://download.hidet.org/whl hidet
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
