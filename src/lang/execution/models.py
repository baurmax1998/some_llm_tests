"""Werte einer SIC-Ausführung."""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExecutionResult:
    goal: str
    value: Any
