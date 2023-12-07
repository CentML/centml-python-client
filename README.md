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
```
Then, run the tests with
```bash
cd tests
pytest
```