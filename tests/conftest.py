"""Test policy: local runs must be cheap (mocked models only). Tests that load a real model are
marked `model` and skipped unless RUN_MODEL_TESTS=1 (intended for Kaggle/Colab)."""
import os

import pandas  # noqa: F401  Windows DLL load-order workaround (pyarrow vs torch)
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "model: loads a real model; needs RUN_MODEL_TESTS=1")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_MODEL_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="loads a real model; set RUN_MODEL_TESTS=1 (Kaggle only)")
    for item in items:
        if "model" in item.keywords:
            item.add_marker(skip)
