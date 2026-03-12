import os
import pytest

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Tests that require PyTorch at import time -- skip during sanity runs
# where PyTorch is not installed.
_PYTORCH_TEST_FILES = ["test_backend.py", "test_helpers.py", "test_server.py"]

collect_ignore = []


def pytest_addoption(parser):
    parser.addoption("--sanity", action="store_true", help="Run sanity tests (exclude 'gpu' tests)")


def pytest_configure(config):
    if config.getoption("--sanity", default=False):
        collect_ignore.extend(_PYTORCH_TEST_FILES)


def pytest_collection_modifyitems(config, items):
    if config.getoption("--sanity"):
        skip_gpu = pytest.mark.skip(reason="Skipping GPU tests for sanity run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
