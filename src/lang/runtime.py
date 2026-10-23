"""Abhängigkeitsgetriebene Ausführung geprüfter Programme."""
from __future__ import annotations
from dataclasses import dataclass
from functools import reduce
from typing import Any, Callable
from . import ast
from .checker import CheckedProgram
from .errors import LangRuntimeError
from .registry import ModuleRegistry
from .types import FunctionType, ListType, Primitive, RecordType, Type, TypeVariable, format_type, prune


@dataclass(frozen=True)
class ExecutionResult:
    goal: str
    value: Any


class Evaluator:
    def __init__(self, checked: CheckedProgram, registry: ModuleRegistry | None = None):
        self.checked = checked
        self.registry = registry or ModuleRegistry()
        self.definitions = {binding.name: binding.value for binding in checked.program.bindings}
        self.values: dict[str, Any] = {}
        self.active: set[str] = set()

    def run(self) -> ExecutionResult:
        goal = self.checked.program.goal.name
        return ExecutionResult(goal, self.resolve(goal))

    def resolve(self, name: str) -> Any:
        if name in self.values:
            return self.values[name]
        if name in self.active:
            raise LangRuntimeError(f"Unzulässiger Abhängigkeitszyklus bei {name!r}")
        if name not in self.definitions:
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
        if isinstance(expression, ast.ListExpr):
            return [self.evaluate(item, local) for item in expression.items]
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
            left, right = self.evaluate(expression.left, local), self.evaluate(expression.right, local)
            operations: dict[str, Callable[[Any, Any], Any]] = {
                "+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
                "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
                "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
                ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
            }
            return operations[expression.operator](left, right)
        if isinstance(expression, ast.IfExpr):
            branch = expression.then_branch if self.evaluate(expression.condition, local) else expression.else_branch
            return self.evaluate(branch, local)
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
            function = self.evaluate(expression.function, local)
            return function(*(self.evaluate(argument, local) for argument in expression.arguments))
        if isinstance(expression, ast.MethodCall):
            values = self.evaluate(expression.value, local)
            arguments = [self.evaluate(argument, local) for argument in expression.arguments]
            if expression.name == "filter": return [item for item in values if arguments[0](item)]
            if expression.name == "map": return [arguments[0](item) for item in values]
            if expression.name == "sum": return sum(values)
            if expression.name == "fold": return reduce(arguments[1], values, arguments[0])
            raise LangRuntimeError(f"Unbekannte Methode {expression.name!r}")
        raise AssertionError(type(expression))  # pragma: no cover


def validate_runtime_value(value: Any, expected: Type, context: str = "value") -> None:
    """Prüft insbesondere untypisierte Werte aus externen Funktionen."""
    expected = prune(expected)
    valid = True
    if isinstance(expected, Primitive):
        if expected.name == "Bool":
            valid = type(value) is bool
        elif expected.name == "Int":
            valid = type(value) is int
        elif expected.name == "String":
            valid = isinstance(value, str)
    elif isinstance(expected, ListType):
        valid = isinstance(value, list)
        if valid:
            for index, item in enumerate(value):
                validate_runtime_value(item, expected.element, f"{context}[{index}]")
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


def execute(checked: CheckedProgram, registry: ModuleRegistry | None = None) -> ExecutionResult:
    return Evaluator(checked, registry).run()
