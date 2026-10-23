"""Typprüfung mit Vorwärtsreferenzen und rückwärts propagierten Erwartungen."""
from __future__ import annotations
from dataclasses import dataclass
from . import ast
from .dependencies import check_definition_order, check_dependency_cycles
from .errors import LangNameError, LangTypeError
from .registry import ModuleRegistry
from .types import (
    BOOL, INT, STRING, FunctionType, ListType, Primitive, RecordType, Type,
    TypeVariable, format_type, prune, unify,
)


@dataclass(frozen=True)
class CheckedProgram:
    program: ast.Program
    bindings: dict[str, Type]
    records: dict[str, RecordType]

    def type_of(self, name: str) -> str:
        return format_type(self.bindings[name])


class TypeChecker:
    def __init__(self, registry: ModuleRegistry | None = None):
        self.registry = registry or ModuleRegistry()
        self.counter = 0
        self.records: dict[str, RecordType] = {}
        self.bindings: dict[str, Type] = {}

    def fresh(self) -> TypeVariable:
        result = TypeVariable(self.counter)
        self.counter += 1
        return result

    def from_ref(self, reference: ast.TypeRef) -> Type:
        if reference.name == "Bool": return BOOL
        if reference.name == "Int": return INT
        if reference.name == "String": return STRING
        if reference.name == "List":
            assert reference.argument is not None
            return ListType(self.from_ref(reference.argument))
        if reference.argument is not None:
            raise LangTypeError(f"Unbekannter parametrisierter Typ {reference.name}")
        return self.records.setdefault(reference.name, RecordType(reference.name))

    def check(self, program: ast.Program) -> CheckedProgram:
        check_definition_order(program)
        check_dependency_cycles(program)
        # Deklarationsannotationen werden absichtlich noch nicht angewandt:
        # Frühere Verwendungen dürfen den Typ zuerst festlegen. So kann eine
        # spätere, redundante Annotation zuverlässig erkannt werden.
        for binding in program.bindings:
            self.bindings[binding.name] = self.fresh()
        for binding in program.bindings:
            try:
                expected = self.bindings[binding.name]
                if binding.annotation is not None:
                    declared = self.from_ref(binding.annotation)
                    existing = prune(expected)
                    if (
                        self.is_fully_resolved(existing)
                        and format_type(existing) == format_type(declared)
                    ):
                        raise LangTypeError(
                            f"Redundante Typannotation bei {binding.name!r}; "
                            f"der Typ {format_type(existing)} wurde bereits festgelegt"
                        )
                    unify(expected, declared, binding.name)
                inferred = self.infer(binding.value, self.bindings, expected)
                unify(expected, inferred, binding.name)
            except (LangNameError, LangTypeError) as exc:
                span = binding.value.span
                if span is not None:
                    exc.attach(span.start, span.end)
                raise
        for name, type_ in self.bindings.items():
            self.require_resolved(type_, name, set())
        return CheckedProgram(program, dict(self.bindings), dict(self.records))

    def is_fully_resolved(self, type_: Type, seen: set[int] | None = None) -> bool:
        seen = set() if seen is None else seen
        type_ = prune(type_)
        if id(type_) in seen:
            return True
        seen.add(id(type_))
        if isinstance(type_, TypeVariable):
            return False
        if isinstance(type_, ListType):
            return self.is_fully_resolved(type_.element, seen)
        if isinstance(type_, FunctionType):
            return all(self.is_fully_resolved(item, seen) for item in [*type_.parameters, type_.result])
        if isinstance(type_, RecordType):
            return all(self.is_fully_resolved(item, seen) for item in type_.fields.values())
        return True

    def require_resolved(self, type_: Type, context: str, seen: set[int]) -> None:
        type_ = prune(type_)
        if id(type_) in seen:
            return
        seen.add(id(type_))
        if isinstance(type_, TypeVariable):
            raise LangTypeError(f"Typ von {context!r} ist nicht eindeutig; Annotation erforderlich")
        if isinstance(type_, ListType):
            self.require_resolved(type_.element, context, seen)
        elif isinstance(type_, FunctionType):
            for item in [*type_.parameters, type_.result]:
                self.require_resolved(item, context, seen)
        elif isinstance(type_, RecordType):
            for field_name, field_type in type_.fields.items():
                self.require_resolved(field_type, f"{context}.{field_name}", seen)

    def infer(self, expression: ast.Expr, scope: dict[str, Type], expected: Type | None = None) -> Type:
        result: Type
        if isinstance(expression, ast.Literal):
            result = BOOL if isinstance(expression.value, bool) else INT if isinstance(expression.value, int) else STRING
        elif isinstance(expression, ast.Name):
            if expression.origin:
                result = self.registry.resolve(expression.name, expression.origin).type
            else:
                if expression.name not in scope:
                    raise LangNameError(f"Unbekannter Name {expression.name!r}")
                result = scope[expression.name]
        elif isinstance(expression, ast.Annotation):
            result = self.from_ref(expression.type_ref)
            self.infer(expression.value, scope, result)
        elif isinstance(expression, ast.ListExpr):
            element = prune(expected).element if isinstance(prune(expected), ListType) else self.fresh()
            for item in expression.items:
                self.infer(item, scope, element)
            result = ListType(element)
        elif isinstance(expression, ast.RecordExpr):
            expected_record = prune(expected) if expected is not None else None
            fields = {
                name: self.infer(
                    value,
                    scope,
                    expected_record.fields.get(name)
                    if isinstance(expected_record, RecordType) else None,
                )
                for name, value in expression.fields
            }
            result = RecordType(None, fields, closed=True)
        elif isinstance(expression, ast.Unary):
            wanted = BOOL if expression.operator == "not" else INT
            self.infer(expression.operand, scope, wanted)
            result = wanted
        elif isinstance(expression, ast.Binary):
            result = self.infer_binary(expression, scope, expected)
        elif isinstance(expression, ast.IfExpr):
            self.infer(expression.condition, scope, BOOL)
            branch = expected or self.fresh()
            self.infer(expression.then_branch, scope, branch)
            self.infer(expression.else_branch, scope, branch)
            result = branch
        elif isinstance(expression, ast.Field):
            owner = prune(self.infer(expression.value, scope))
            if isinstance(owner, TypeVariable):
                record = RecordType(None, {expression.name: self.fresh()})
                unify(owner, record)
                owner = record
            if not isinstance(owner, RecordType):
                raise LangTypeError(f"Feldzugriff .{expression.name} auf {format_type(owner)}")
            if owner.closed and expression.name not in owner.fields:
                raise LangTypeError(f"Record besitzt kein Feld {expression.name!r}")
            result = owner.fields.setdefault(expression.name, self.fresh())
        elif isinstance(expression, ast.Lambda):
            expected_function = prune(expected) if expected is not None else None
            if isinstance(expected_function, FunctionType):
                if len(expected_function.parameters) != len(expression.parameters):
                    raise LangTypeError("Falsche Parameteranzahl im Lambda")
                parameters = expected_function.parameters
                body_type = expected_function.result
            else:
                parameters = [self.fresh() for _ in expression.parameters]
                body_type = self.fresh()
            local = dict(scope)
            local.update(zip(expression.parameters, parameters))
            self.infer(expression.body, local, body_type)
            result = FunctionType(parameters, body_type)
        elif isinstance(expression, ast.Call):
            arguments = [self.infer(argument, scope) for argument in expression.arguments]
            result = expected or self.fresh()
            self.infer(expression.function, scope, FunctionType(arguments, result))
        elif isinstance(expression, ast.MethodCall):
            result = self.infer_method(expression, scope, expected)
        else:  # pragma: no cover
            raise AssertionError(type(expression))
        if expected is not None:
            unify(expected, result)
            return expected
        return result

    def infer_binary(self, expression: ast.Binary, scope: dict[str, Type], expected: Type | None) -> Type:
        operator = expression.operator
        if operator in {"and", "or"}:
            self.infer(expression.left, scope, BOOL); self.infer(expression.right, scope, BOOL)
            return BOOL
        if operator in {"<", "<=", ">", ">="}:
            self.infer(expression.left, scope, INT); self.infer(expression.right, scope, INT)
            return BOOL
        if operator in {"==", "!="}:
            operand = self.infer(expression.left, scope)
            self.infer(expression.right, scope, operand)
            return BOOL
        if operator in {"-", "*"}:
            self.infer(expression.left, scope, INT); self.infer(expression.right, scope, INT)
            return INT
        if operator == "+":
            wanted = prune(expected) if expected is not None else None
            if wanted not in {INT, STRING}:
                left = prune(self.infer(expression.left, scope))
                if left not in {INT, STRING}:
                    raise LangTypeError("'+' ist ohne Int- oder String-Kontext mehrdeutig")
                wanted = left
            self.infer(expression.left, scope, wanted)
            self.infer(expression.right, scope, wanted)
            return wanted
        raise LangTypeError(f"Unbekannter Operator {operator}")

    def infer_method(self, expression: ast.MethodCall, scope: dict[str, Type], expected: Type | None) -> Type:
        collection = prune(self.infer(expression.value, scope))
        if isinstance(collection, TypeVariable):
            collection.instance = ListType(self.fresh())
            collection = collection.instance
        if not isinstance(collection, ListType):
            raise LangTypeError(f".{expression.name} benötigt eine Liste")
        element = collection.element
        if expression.name == "filter" and len(expression.arguments) == 1:
            self.infer(expression.arguments[0], scope, FunctionType([element], BOOL))
            return collection
        if expression.name == "map" and len(expression.arguments) == 1:
            mapped = prune(expected).element if isinstance(prune(expected), ListType) else self.fresh()
            self.infer(expression.arguments[0], scope, FunctionType([element], mapped))
            return ListType(mapped)
        if expression.name == "sum" and not expression.arguments:
            unify(element, INT, "sum")
            return INT
        if expression.name == "fold" and len(expression.arguments) == 2:
            accumulator = self.infer(expression.arguments[0], scope, expected)
            self.infer(expression.arguments[1], scope, FunctionType([accumulator, element], accumulator))
            return accumulator
        raise LangTypeError(f"Unbekannte oder falsch verwendete Listenoperation .{expression.name}")


def check(program: ast.Program, registry: ModuleRegistry | None = None) -> CheckedProgram:
    return TypeChecker(registry).check(program)
