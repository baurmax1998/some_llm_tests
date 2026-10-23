"""Interne Typrepräsentation und Unifikation."""
from __future__ import annotations
from dataclasses import dataclass, field
from .errors import LangTypeError


class Type: pass


@dataclass(frozen=True)
class Primitive(Type):
    name: str


BOOL = Primitive("Bool")
INT = Primitive("Int")
STRING = Primitive("String")


@dataclass
class TypeVariable(Type):
    identifier: int
    instance: Type | None = None


@dataclass
class ListType(Type):
    element: Type


@dataclass
class RecordType(Type):
    name: str | None
    fields: dict[str, Type] = field(default_factory=dict)


@dataclass
class FunctionType(Type):
    parameters: list[Type]
    result: Type


def prune(type_: Type) -> Type:
    if isinstance(type_, TypeVariable) and type_.instance is not None:
        type_.instance = prune(type_.instance)
        return type_.instance
    return type_


def occurs(variable: TypeVariable, type_: Type) -> bool:
    type_ = prune(type_)
    if type_ is variable:
        return True
    if isinstance(type_, ListType):
        return occurs(variable, type_.element)
    if isinstance(type_, FunctionType):
        return any(occurs(variable, item) for item in [*type_.parameters, type_.result])
    if isinstance(type_, RecordType):
        return any(occurs(variable, item) for item in type_.fields.values())
    return False


def unify(left: Type, right: Type, context: str = "") -> Type:
    left, right = prune(left), prune(right)
    if left is right:
        return left
    if isinstance(left, TypeVariable):
        if occurs(left, right):
            raise LangTypeError(f"Rekursiver Typ{_at(context)}")
        left.instance = right
        return right
    if isinstance(right, TypeVariable):
        return unify(right, left, context)
    if isinstance(left, Primitive) and isinstance(right, Primitive) and left == right:
        return left
    if isinstance(left, ListType) and isinstance(right, ListType):
        unify(left.element, right.element, context)
        return left
    if isinstance(left, FunctionType) and isinstance(right, FunctionType):
        if len(left.parameters) != len(right.parameters):
            raise LangTypeError(f"Unterschiedliche Parameteranzahl{_at(context)}")
        for a, b in zip(left.parameters, right.parameters):
            unify(a, b, context)
        unify(left.result, right.result, context)
        return left
    if isinstance(left, RecordType) and isinstance(right, RecordType):
        # Benannte Records sammeln Anforderungen; anonyme Literale dürfen mehr Felder haben.
        if left.name and right.name and left.name != right.name:
            raise LangTypeError(f"Record-Typen {left.name} und {right.name} sind verschieden{_at(context)}")
        shared = set(left.fields) & set(right.fields)
        for name in shared:
            unify(left.fields[name], right.fields[name], f"Feld {name}")
        if left.name:
            missing = set(left.fields) - set(right.fields) if right.name is None else set()
            if missing:
                raise LangTypeError(f"Record fehlt Feld {sorted(missing)[0]!r}{_at(context)}")
            for name, value in right.fields.items():
                left.fields.setdefault(name, value)
            return left
        if right.name:
            return unify(right, left, context)
        if set(left.fields) != set(right.fields):
            raise LangTypeError(f"Unterschiedliche Record-Felder{_at(context)}")
        return left
    raise LangTypeError(f"Typkonflikt: {format_type(left)} und {format_type(right)}{_at(context)}")


def format_type(type_: Type) -> str:
    type_ = prune(type_)
    if isinstance(type_, Primitive): return type_.name
    if isinstance(type_, TypeVariable): return f"?{type_.identifier}"
    if isinstance(type_, ListType): return f"List<{format_type(type_.element)}>"
    if isinstance(type_, RecordType):
        if type_.name: return type_.name
        fields = ", ".join(f"{k}: {format_type(v)}" for k, v in type_.fields.items())
        return "{" + fields + "}"
    if isinstance(type_, FunctionType):
        return "(" + ", ".join(map(format_type, type_.parameters)) + ") => " + format_type(type_.result)
    return repr(type_)


def _at(context: str) -> str:
    return f" bei {context}" if context else ""
