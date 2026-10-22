"""Modellzugriff und Token-Generierung."""

from .backend import DEFAULT_MODEL, MLXModelBackend
from .generation import generate

__all__ = ["DEFAULT_MODEL", "MLXModelBackend", "generate"]
