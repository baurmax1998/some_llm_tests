"""Statische Analyse der Abhängigkeiten zwischen Top-Level-Bindings."""
from __future__ import annotations
from dataclasses import fields, is_dataclass
from typing import Iterable

from . import ast
from .errors import LangError


class LangDependencyError(LangError):
    pass


class LangDefinitionOrderError(LangError):
    pass


def expression_names(expression: ast.Expr, local: frozenset[str] = frozenset()) -> set[str]:
    """Sammelt unqualifizierte freie Namen eines Ausdrucks."""
    if isinstance(expression, ast.Name):
        return set() if expression.origin or expression.name in local else {expression.name}
    if isinstance(expression, ast.Lambda):
        return expression_names(expression.body, local | frozenset(expression.parameters))
    result: set[str] = set()
    if is_dataclass(expression):
        for descriptor in fields(expression):
            value = getattr(expression, descriptor.name)
            result.update(_names_in_value(value, local))
    return result


def _names_in_value(value: object, local: frozenset[str]) -> set[str]:
    if isinstance(value, ast.Expr):
        return expression_names(value, local)
    if isinstance(value, tuple):
        result: set[str] = set()
        for item in value:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], ast.Expr):
                result.update(expression_names(item[1], local))
            else:
                result.update(_names_in_value(item, local))
        return result
    return set()


def check_definition_order(program: ast.Program) -> None:
    """Erzwingt: Ziel zuerst, danach ausschließlich bereits benötigte Namen."""
    if not program.bindings or not program.bindings[0].is_goal:
        binding = program.bindings[0] if program.bindings else None
        error = LangDefinitionOrderError("Das goal muss die erste Definition sein")
        if binding is not None and binding.span is not None:
            error.attach(binding.span.start, binding.span.end)
        raise error

    required = expression_names(program.bindings[0].value)
    for binding in program.bindings[1:]:
        if binding.name not in required:
            error = LangDefinitionOrderError(
                f"{binding.name!r} wird definiert, bevor der Name benötigt wird"
            )
            if binding.span is not None:
                error.attach(binding.span.start, binding.span.end)
            raise error
        required.discard(binding.name)
        required.update(expression_names(binding.value))


def dependency_graph(program: ast.Program) -> dict[str, set[str]]:
    names = {binding.name for binding in program.bindings}
    return {
        binding.name: expression_names(binding.value) & names
        for binding in program.bindings
    }


def check_dependency_cycles(program: ast.Program) -> dict[str, set[str]]:
    graph = dependency_graph(program)
    visiting: list[str] = []
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            start = visiting.index(name)
            cycle = visiting[start:] + [name]
            error = LangDependencyError(
                "Unzulässiger Abhängigkeitszyklus: " + " -> ".join(cycle)
            )
            binding = next(item for item in program.bindings if item.name == name)
            if binding.span is not None:
                error.attach(binding.span.start, binding.span.end)
            raise error
        if name in visited:
            return
        visiting.append(name)
        for dependency in graph[name]:
            visit(dependency)
        visiting.pop()
        visited.add(name)

    for name in graph:
        visit(name)
    return graph
