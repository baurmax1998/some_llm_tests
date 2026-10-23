"""Abstrakter Syntaxbaum der Zielsprache."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceSpan:
    start: int
    end: int


@dataclass(frozen=True)
class TypeRef:
    name: str
    argument: TypeRef | None = None


@dataclass(frozen=True)
class Expr:
    span: SourceSpan | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class Literal(Expr): value: Any


@dataclass(frozen=True)
class Name(Expr):
    name: str
    origin: tuple[str, ...] = ()


@dataclass(frozen=True)
class ListExpr(Expr): items: tuple[Expr, ...]


@dataclass(frozen=True)
class RecordExpr(Expr): fields: tuple[tuple[str, Expr], ...]


@dataclass(frozen=True)
class Annotation(Expr):
    value: Expr
    type_ref: TypeRef


@dataclass(frozen=True)
class Unary(Expr):
    operator: str
    operand: Expr


@dataclass(frozen=True)
class Binary(Expr):
    operator: str
    left: Expr
    right: Expr


@dataclass(frozen=True)
class IfExpr(Expr):
    condition: Expr
    then_branch: Expr
    else_branch: Expr


@dataclass(frozen=True)
class Lambda(Expr):
    parameters: tuple[str, ...]
    body: Expr


@dataclass(frozen=True)
class Call(Expr):
    function: Expr
    arguments: tuple[Expr, ...]


@dataclass(frozen=True)
class Field(Expr):
    value: Expr
    name: str


@dataclass(frozen=True)
class MethodCall(Expr):
    value: Expr
    name: str
    arguments: tuple[Expr, ...]


@dataclass(frozen=True)
class Binding:
    name: str
    value: Expr
    annotation: TypeRef | None = None
    is_goal: bool = False
    span: SourceSpan | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class Program:
    bindings: tuple[Binding, ...]

    @property
    def goal(self) -> Binding:
        goals = [binding for binding in self.bindings if binding.is_goal]
        if len(goals) != 1:
            raise ValueError("Ein Programm benötigt genau ein goal")
        return goals[0]
