"""Register für externe, vollständig qualifizierte Funktionen."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
from .calculation import CalculationSpec, ErrorSpec
from .errors import LangNameError
from .types import FunctionType, clone_type


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
        self._calculations: dict[tuple[str, ...], CalculationSpec] = {}

    def register(
        self, symbol: str, origin: tuple[str, ...], type_: FunctionType,
        implementation: Callable[..., Any],
    ) -> None:
        external = ExternalFunction(symbol, tuple(origin), type_, implementation)
        if external.path in self._functions or external.path in self._calculations:
            raise ValueError(f"Externe Funktion {'::'.join(external.path)} ist bereits registriert")
        self._functions[external.path] = external

    def resolve(self, symbol: str, origin: tuple[str, ...]) -> ExternalFunction:
        path = (symbol, *origin)
        try:
            external = self._functions[path]
        except KeyError as exc:
            raise LangNameError(f"Unbekannte externe Funktion {'::'.join(path)}") from exc
        # Generische Typvariablen einer Signatur dürfen sich nicht zwischen
        # unabhängigen Aufrufen gegenseitig festlegen.
        return ExternalFunction(
            external.symbol,
            external.origin,
            clone_type(external.type),
            external.implementation,
        )

    def register_source(self, module: str, package: str, source: str) -> None:
        """Kompiliert eine Datei und exportiert ihr Goal sowie sämtliche Lets."""
        from .checker import check
        from .parser import parse

        checked = check(parse(source), self)
        dependencies = {
            binding.name: _all_expression_names(binding.value)
            for binding in checked.program.bindings
        }
        origin = (module, package)
        for binding in checked.program.bindings:
            # Direkte importierte Fehler behalten ihre ursprüngliche Identität.
            direct = tuple(
                original if original.identity[:1] != ("__local__",) else
                ErrorSpec(original.name, (*origin, original.name), original.payload)
                for original in checked.direct_errors.get(binding.name, ())
            )
            checked.direct_errors[binding.name] = direct
        for binding in checked.program.bindings:
            spec = CalculationSpec(
                binding.name, origin, dict(checked.bindings), dict(checked.inputs),
                dependencies, dict(checked.direct_errors), checked,
            )
            path = (binding.name, *origin)
            if path in self._functions or path in self._calculations:
                raise ValueError(f"Export {'::'.join(path)} ist bereits registriert")
            self._calculations[path] = spec

    # Verständlicher Alias für Projektintegrationen.
    register_module = register_source

    def resolve_calculation(self, symbol: str, origin: tuple[str, ...]) -> CalculationSpec:
        path = (symbol, *origin)
        try:
            calculation = self._calculations[path]
        except KeyError as exc:
            raise LangNameError(f"Unbekannte Berechnung {'::'.join(path)}") from exc
        variables = {}
        return CalculationSpec(
            calculation.symbol,
            calculation.origin,
            {name: clone_type(type_, variables) for name, type_ in calculation.output_types.items()},
            {name: clone_type(type_, variables) for name, type_ in calculation.input_types.items()},
            calculation.dependencies,
            {
                name: tuple(ErrorSpec(error.name, error.identity,
                                      clone_type(error.payload, variables) if error.payload else None)
                            for error in errors)
                for name, errors in calculation.direct_errors.items()
            },
            calculation.checked,
        )

    def has_calculation(self, symbol: str, origin: tuple[str, ...]) -> bool:
        return (symbol, *origin) in self._calculations

    def paths(self) -> tuple[tuple[str, ...], ...]:
        return tuple(sorted((*self._functions, *self._calculations)))


def _all_expression_names(expression):
    """Freie Namen einschließlich noch nicht definierter Eingaben."""
    from .dependencies import expression_names
    return expression_names(expression)
