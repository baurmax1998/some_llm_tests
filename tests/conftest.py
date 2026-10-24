"""Gemeinsame Fixtures für lokale Modelltests."""
import os
import time

import pytest

from src.llm import DEFAULT_MODEL, MLXModelBackend


@pytest.fixture(scope="session")
def backend():
    """Das große Modell nur einmal pro Testlauf laden."""
    started = time.perf_counter()
    instance = MLXModelBackend(os.getenv("TEST_MODEL_ID", DEFAULT_MODEL))
    print(f"\nModell laden: {time.perf_counter() - started:.3f} s")
    return instance
