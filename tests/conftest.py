"""Gemeinsame Fixtures für lokale Modelltests."""
import os

import pytest

from src.llm import DEFAULT_MODEL, MLXModelBackend


@pytest.fixture(scope="session")
def backend():
    """Das große Modell nur einmal pro Testlauf laden."""
    return MLXModelBackend(os.getenv("TEST_MODEL_ID", DEFAULT_MODEL))
