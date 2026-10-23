"""Abhängigkeitsgetriebene Ausführung geprüfter Programme."""
from __future__ import annotations
from dataclasses import dataclass
from functools import reduce
from typing import Any, Callable
from . import ast
from .checker import CheckedProgram
from .errors import LangCalculationError, LangRuntimeError
from .registry import ModuleRegistry
from .types import (
    FunctionType, ListType, MapType, NullableType, Primitive, RecordType, Type,
    TypeVariable, format_type, prune,
)


@dataclass(frozen=True)
class ExecutionResult:
    goal: str
    value: Any


class Evaluator:
    def __init__(
        self, checked: CheckedProgram, registry: ModuleRegistry | None = None,
        overrides: dict[str, Any] | None = None,
        identity_origin: tuple[str, ...] = (),
    ):
        self.checked = checked
        self.registry = registry or ModuleRegistry()
        self.definitions = {binding.name: binding.value for binding in checked.program.bindings}
        self.values: dict[str, Any] = {}
        self.active: set[str] = set()
        self.overrides = dict(overrides or {})
        self.identity_origin = identity_origin
        self.captured_values: dict[int, dict[str, Any]] = {}

    def run(self) -> ExecutionResult:
        goal = self.checked.program.goal.name
        return ExecutionResult(goal, self.resolve(goal))

    def call(self, name: str, **arguments: Any) -> Any:
        """Ruft ein Goal oder Let von Python aus mit denselben Override-Regeln auf."""
        nested = Evaluator(self.checked, self.registry, arguments, self.identity_origin)
        return nested.resolve(name)

    def resolve(self, name: str) -> Any:
        if name in self.overrides:
            expected = self.checked.bindings.get(name, self.checked.inputs.get(name))
            if expected is not None:
                validate_runtime_value(self.overrides[name], expected, name)
            return self.overrides[name]
        if name in self.values:
            return self.values[name]
        if name in self.active:
            raise LangRuntimeError(f"Unzulässiger Abhängigkeitszyklus bei {name!r}")
        if name not in self.definitions:
            if name in self.checked.inputs:
                raise LangRuntimeError(f"Erforderliche Eingabe {name!r} wurde nicht übergeben")
            raise LangRuntimeError(f"Name {name!r} besitzt keine Definition")
        self.active.add(name)
        try:
            value = self.evaluate(self.definitions[name], {})
            validate_runtime_value(value, self.checked.bindings[name], name)
            self.values[name] = value
            return value
        finally:
            self.active.remove(name)

    def evaluate(self, expression: ast.Expr, local: dict[str, Any]) -> Any:
        if isinstance(expression, ast.Literal): return expression.value
        if isinstance(expression, ast.Name):
            if expression.origin:
                return self.registry.resolve(expression.name, expression.origin).implementation
            return local[expression.name] if expression.name in local else self.resolve(expression.name)
        if isinstance(expression, ast.Annotation): return self.evaluate(expression.value, local)
        if isinstance(expression, ast.ErrorExpr):
            payload = self.evaluate(expression.payload, local) if expression.payload is not None else None
            identity = (*self.identity_origin, expression.name) if self.identity_origin else (expression.name,)
            raise LangCalculationError(expression.name, payload, identity)
        if isinstance(expression, ast.OutputAccess):
            key = id(expression.captured)
            if key not in self.captured_values:
                self.captured_values[key] = self.evaluate_capture(expression.captured, local)
            return self.captured_values[key][expression.name]
        if isinstance(expression, ast.ListExpr):
            return [self.evaluate(item, local) for item in expression.items]
        if isinstance(expression, ast.MapExpr):
            return {
                self.evaluate(key, local): self.evaluate(value, local)
                for key, value in expression.entries
            }
        if isinstance(expression, ast.RecordExpr):
            return {name: self.evaluate(value, local) for name, value in expression.fields}
        if isinstance(expression, ast.Unary):
            value = self.evaluate(expression.operand, local)
            return not value if expression.operator == "not" else -value
        if isinstance(expression, ast.Binary):
            if expression.operator == "and":
                return self.evaluate(expression.left, local) and self.evaluate(expression.right, local)
            if expression.operator == "or":
                return self.evaluate(expression.left, local) or self.evaluate(expression.right, local)
            if expression.operator == "??":
                left = self.evaluate(expression.left, local)
                return left if left is not None else self.evaluate(expression.right, local)
            left, right = self.evaluate(expression.left, local), self.evaluate(expression.right, local)
            operations: dict[str, Callable[[Any, Any], Any]] = {
                "+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
                "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
                "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
                ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
            }
            operations.update({
                "/": lambda a, b: a / b,
                "%": lambda a, b: a % b,
            })
            return operations[expression.operator](left, right)
        if isinstance(expression, ast.IfExpr):
            branch = expression.then_branch if self.evaluate(expression.condition, local) else expression.else_branch
            return self.evaluate(branch, local)
        if isinstance(expression, ast.MatchExpr):
            value = self.evaluate(expression.value, local)
            for case in expression.cases:
                pattern = case.pattern
                matched = False
                case_local = local
                if isinstance(pattern, ast.NullPattern):
                    matched = value is None
                elif isinstance(pattern, ast.LiteralPattern):
                    matched = value == pattern.value
                elif isinstance(pattern, ast.BindingPattern):
                    matched = True
                    case_local = {**local, pattern.name: value}
                elif isinstance(pattern, ast.WildcardPattern):
                    matched = True
                if matched:
                    return self.evaluate(case.body, case_local)
            raise LangRuntimeError("Kein match-Fall passt zum Wert")
        if isinstance(expression, ast.LocalLet):
            value = self.evaluate(expression.value, local)
            return self.evaluate(expression.body, {**local, expression.name: value})
        if isinstance(expression, ast.Index):
            try:
                return self.evaluate(expression.value, local)[self.evaluate(expression.index, local)]
            except (KeyError, IndexError) as exc:
                raise LangRuntimeError("Index oder Map-Schlüssel ist nicht vorhanden") from exc
        if isinstance(expression, ast.SafeField):
            value = self.evaluate(expression.value, local)
            if value is None:
                return None
            try:
                return value[expression.name] if isinstance(value, dict) else getattr(value, expression.name)
            except (KeyError, AttributeError) as exc:
                raise LangRuntimeError(f"Feld {expression.name!r} fehlt zur Laufzeit") from exc
        if isinstance(expression, ast.Field):
            value = self.evaluate(expression.value, local)
            try:
                return value[expression.name] if isinstance(value, dict) else getattr(value, expression.name)
            except (KeyError, AttributeError) as exc:
                raise LangRuntimeError(f"Feld {expression.name!r} fehlt zur Laufzeit") from exc
        if isinstance(expression, ast.Lambda):
            captured = dict(local)
            def function(*arguments: Any) -> Any:
                if len(arguments) != len(expression.parameters):
                    raise LangRuntimeError("Falsche Parameteranzahl")
                return self.evaluate(expression.body, {**captured, **dict(zip(expression.parameters, arguments))})
            return function
        if isinstance(expression, ast.Call):
            spec = self.checked.call_specs.get(id(expression))
            if spec is not None:
                arguments = {
                    argument.name: self.evaluate(argument.value, local)
                    for argument in expression.arguments
                    if isinstance(argument, ast.NamedArgument)
                }
                return self.invoke_calculation(spec, arguments).resolve(spec.symbol)
            function = self.evaluate(expression.function, local)
            return function(*(self.evaluate(argument, local) for argument in expression.arguments))
        if isinstance(expression, ast.SafeMethodCall):
            values = self.evaluate(expression.value, local)
            if values is None:
                return None
            # Argumente müssen weiterhin im ursprünglichen Scope ausgewertet werden.
            arguments = [self.evaluate(argument, local) for argument in expression.arguments]
            return self.evaluate_method(values, expression.name, arguments)
        if isinstance(expression, ast.MethodCall):
            values = self.evaluate(expression.value, local)
            arguments = [self.evaluate(argument, local) for argument in expression.arguments]
            return self.evaluate_method(values, expression.name, arguments)
        raise AssertionError(type(expression))  # pragma: no cover

    def invoke_calculation(self, spec, arguments: dict[str, Any]) -> "Evaluator":
        checked = spec.checked or self.checked
        return Evaluator(checked, self.registry, arguments, spec.origin)

    def evaluate_capture(self, captured: ast.CapturedCall, local: dict[str, Any]) -> dict[str, Any]:
        info = self.checked.captures[id(captured)]
        arguments = {
            argument.name: self.evaluate(argument.value, local)
            for argument in captured.call.arguments
            if isinstance(argument, ast.NamedArgument)
        }
        evaluator = self.invoke_calculation(info.spec, arguments)
        normal = [name for name in captured.outputs if name not in info.caught]
        values = {name: None for name in captured.outputs}
        try:
            # Das Hauptergebnis bestimmt Erfolg oder Fehler. Erst danach werden
            # angeforderte Zwischenwerte aus derselben Auswertung entnommen.
            evaluator.resolve(info.spec.symbol)
            for name in normal:
                values[name] = evaluator.resolve(name)
            return values
        except LangCalculationError as error:
            match = next(
                (name for name, declared in info.caught.items()
                 if declared.name == error.name and
                 (declared.identity == error.identity or declared.identity[:1] == ("__local__",))),
                None,
            )
            if match is None:
                raise
            values[match] = error.payload if info.caught[match].payload is not None else True
            return values

    def evaluate_method(self, values: Any, name: str, arguments: list[Any]) -> Any:
        if name == "keys": return list(values.keys())
        if name == "values": return list(values.values())
        if name == "get": return values.get(arguments[0])
        if name == "contains_key": return arguments[0] in values
        if name == "filter": return [item for item in values if arguments[0](item)]
        if name == "map": return [arguments[0](item) for item in values]
        if name == "flat_map": return [mapped for item in values for mapped in arguments[0](item)]
        if name == "sum": return sum(values)
        if name == "fold": return reduce(arguments[1], values, arguments[0])
        if name == "any": return any(arguments[0](item) for item in values)
        if name == "all": return all(arguments[0](item) for item in values)
        if name == "count": return len(values)
        if name == "take": return values[:arguments[0]]
        if name == "drop": return values[arguments[0]:]
        if name == "first": return values[0] if values else None
        if name == "last": return values[-1] if values else None
        if name == "find": return next((item for item in values if arguments[0](item)), None)
        if name == "sort": return sorted(values)
        if name == "group_by":
            grouped = {}
            for item in values:
                grouped.setdefault(arguments[0](item), []).append(item)
            return grouped
        raise LangRuntimeError(f"Unbekannte Methode {name!r}")


def validate_runtime_value(value: Any, expected: Type, context: str = "value") -> None:
    """Prüft insbesondere untypisierte Werte aus externen Funktionen."""
    expected = prune(expected)
    valid = True
    if isinstance(expected, Primitive):
        if expected.name == "Bool":
            valid = type(value) is bool
        elif expected.name == "Int":
            valid = type(value) is int
        elif expected.name == "Float":
            valid = type(value) is float
        elif expected.name == "Null":
            valid = value is None
        elif expected.name == "String":
            valid = isinstance(value, str)
    elif isinstance(expected, NullableType):
        if value is None:
            return
        validate_runtime_value(value, expected.inner, context)
        return
    elif isinstance(expected, ListType):
        valid = isinstance(value, list)
        if valid:
            for index, item in enumerate(value):
                validate_runtime_value(item, expected.element, f"{context}[{index}]")
    elif isinstance(expected, MapType):
        valid = isinstance(value, dict)
        if valid:
            for key, item in value.items():
                validate_runtime_value(key, expected.key, f"{context}.key")
                validate_runtime_value(item, expected.value, f"{context}[{key!r}]")
    elif isinstance(expected, RecordType):
        valid = isinstance(value, dict)
        if valid:
            for name, field_type in expected.fields.items():
                if name not in value:
                    raise LangRuntimeError(f"{context}: erforderliches Feld {name!r} fehlt")
                validate_runtime_value(value[name], field_type, f"{context}.{name}")
    elif isinstance(expected, FunctionType):
        valid = callable(value)
    elif isinstance(expected, TypeVariable):  # defensiv; der Checker lehnt dies vorher ab
        valid = True
    if not valid:
        raise LangRuntimeError(
            f"{context}: Laufzeitwert passt nicht zu {format_type(expected)}"
        )


def execute(
    checked: CheckedProgram, registry: ModuleRegistry | None = None,
    arguments: dict[str, Any] | None = None,
) -> ExecutionResult:
    return Evaluator(checked, registry, arguments).run()
