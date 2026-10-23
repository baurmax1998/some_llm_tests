"""Register für externe, vollständig qualifizierte Funktionen."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
from .errors import LangNameError
from .types import FunctionType


@dataclass(frozen=True)
class ExternalFunction:
    symbol: str
    origin: tuple[str, ...]
    type: FunctionType
    implementation: Callable[..., Any]

    @property
    def path(self) -> tuple[str, ...]:
        return (self.symbol, *self.origin)


class ModuleRegistry:
    def __init__(self) -> None:
        self._functions: dict[tuple[str, ...], ExternalFunction] = {}

    def register(
        self, symbol: str, origin: tuple[str, ...], type_: FunctionType,
        implementation: Callable[..., Any],
    ) -> None:
        external = ExternalFunction(symbol, tuple(origin), type_, implementation)
        if external.path in self._functions:
            raise ValueError(f"Externe Funktion {'::'.join(external.path)} ist bereits registriert")
        self._functions[external.path] = external

    def resolve(self, symbol: str, origin: tuple[str, ...]) -> ExternalFunction:
        path = (symbol, *origin)
        try:
            return self._functions[path]
        except KeyError as exc:
            raise LangNameError(f"Unbekannte externe Funktion {'::'.join(path)}") from exc
