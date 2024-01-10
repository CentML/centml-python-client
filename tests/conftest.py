import os
import pytest

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pytest_addoption(parser):
    parser.addoption("--sanity", action="store_true", help="Run sanity tests (exclude 'gpu' tests)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--sanity"):
        skip_gpu = pytest.mark.skip(reason="Skipping GPU tests for sanity run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
