import pytest

def pytest_addoption(parser):
    parser.addoption("--sanity", action="store_true", help="Run sanity tests (exclude 'gpu' tests)")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--sanity"):
        skip_gpu = pytest.mark.skip(reason="skipping gpu tests for sanity run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)