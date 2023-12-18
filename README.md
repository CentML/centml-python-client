# centml-python-client

### Installation
First, ensure you meet the requirements for  [Hidet](https://github.com/hidet-org/hidet), namely:
- CUDA Toolkit 11.6+
- Python 3.8+

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
