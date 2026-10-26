"""Programmprüfung, Berechnungen und Fehlerausgänge."""
from __future__ import annotations
from dataclasses import fields, is_dataclass
from .. import ast
from ..calculation import CalculationSpec, ErrorSpec
from ..dependencies import check_definition_order, check_dependency_cycles
from ..errors import LangNameError, LangTypeError
from ..registry import ExternalFunction, ModuleRegistry
from .models import CaptureInfo, CheckedProgram
from ..types import (
    BOOL, FLOAT, INT, NULL, STRING, FunctionType, ListType, MapType,
    NullableType, Primitive, RecordType, Type, TypeVariable, format_type, prune,
    unify,
)


class TypeCheckerBase:
    def __init__(self, registry: ModuleRegistry | None = None):
        self.registry = registry or ModuleRegistry()
        self.counter = 0
        self.records: dict[str, RecordType] = {}
        self.bindings: dict[str, Type] = {}
        self.inputs: dict[str, Type] = {}
        self.direct_errors: dict[str, tuple[ErrorSpec, ...]] = {}
        self.error_payloads: dict[str, Type | None] = {}
        self.call_specs: dict[int, CalculationSpec] = {}
        self.external_calls: dict[int, ExternalFunction] = {}
        self.captures: dict[int, CaptureInfo] = {}
        self.capture_types: dict[int, dict[str, Type]] = {}
        self.program: ast.Program | None = None

    def fresh(self) -> TypeVariable:
        result = TypeVariable(self.counter)
        self.counter += 1
        return result

    def from_ref(self, reference: ast.TypeRef) -> Type:
        if reference.origin:
            result = self.registry.resolve_type(reference.name, reference.origin)
        elif reference.name == "Bool": result: Type = BOOL
        elif reference.name == "Int": result = INT
        elif reference.name == "Float": result = FLOAT
        elif reference.name == "String": result = STRING
        elif reference.name == "List":
            result = ListType(self.from_ref(reference.arguments[0]))
        elif reference.name == "Map":
            result = MapType(
                self.from_ref(reference.arguments[0]),
                self.from_ref(reference.arguments[1]),
            )
        elif reference.arguments:
            raise LangTypeError(f"Unbekannter parametrisierter Typ {reference.name}")
        else:
            result = self.records.setdefault(reference.name, RecordType(reference.name))
        return NullableType(result) if reference.nullable else result

    def check(self, program: ast.Program) -> CheckedProgram:
        check_definition_order(program)
        check_dependency_cycles(program)
        self.program = program
        # Deklarationsannotationen werden absichtlich noch nicht angewandt:
        # Frühere Verwendungen dürfen den Typ zuerst festlegen. So kann eine
        # spätere, redundante Annotation zuverlässig erkannt werden.
        for binding in program.bindings:
            self.bindings[binding.name] = self.fresh()
        self.prepare_errors(program)
        # Mehrfachausgänge müssen vor dem Goal bekannt sein, damit dortige
        # Präsenzprüfungen (`fehler != null`) korrekt verengt werden können.
        for binding in program.bindings:
            if isinstance(binding.value, ast.OutputAccess):
                unify(self.bindings[binding.name], self.infer_capture(binding.value, self.bindings))
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
        for name, type_ in {**self.bindings, **self.inputs}.items():
            self.require_resolved(type_, name, set())
        return CheckedProgram(
            program, dict(self.bindings), dict(self.records), dict(self.inputs),
            dict(self.direct_errors), dict(self.call_specs), dict(self.external_calls),
            dict(self.captures),
        )

    def prepare_errors(self, program: ast.Program) -> None:
        """Reserviert Nutzwerttypen und ordnet Fehler ihren Definitionen zu."""
        def collect(value: object) -> list[ErrorSpec]:
            found: list[ErrorSpec] = []
            if isinstance(value, ast.ErrorExpr):
                payload = self.error_payloads.get(value.name)
                if value.payload is not None:
                    if value.name in self.error_payloads and payload is None:
                        raise LangTypeError(f"Fehler {value.name!r} wird mit und ohne Nutzwert verwendet")
                    payload = payload or self.fresh()
                elif value.name in self.error_payloads and payload is not None:
                    raise LangTypeError(f"Fehler {value.name!r} wird mit und ohne Nutzwert verwendet")
                self.error_payloads[value.name] = payload
                found.append(ErrorSpec(value.name, ("__local__", value.name), payload))
                if value.payload is not None:
                    found.extend(collect(value.payload))
                return found
            if isinstance(value, ast.Call) and isinstance(value.function, ast.Name) and value.function.origin:
                if self.registry.has_calculation(value.function.name, value.function.origin):
                    spec = self.registry.resolve_calculation(value.function.name, value.function.origin)
                    overrides = {item.name for item in value.arguments if isinstance(item, ast.NamedArgument)}
                    found.extend(spec.possible_errors(overrides))
            if isinstance(value, ast.NamedArgument):
                return collect(value.value)
            if isinstance(value, ast.Expr) and is_dataclass(value):
                for descriptor in fields(value):
                    if descriptor.name != "span":
                        found.extend(collect(getattr(value, descriptor.name)))
            elif isinstance(value, tuple):
                for item in value:
                    found.extend(collect(item))
            return found

        for binding in program.bindings:
            unique = {error.identity: error for error in collect(binding.value)}
            self.direct_errors[binding.name] = tuple(unique.values())

    def local_spec(self, symbol: str) -> CalculationSpec:
        assert self.program is not None
        from ..dependencies import expression_names
        dependencies = {
            binding.name: expression_names(binding.value)
            for binding in self.program.bindings
        }
        # Noch nicht besuchte freie Namen erhalten beim Prüfen des Arguments
        # ihren Typ und werden in dasselbe Interface aufgenommen.
        return CalculationSpec(
            symbol, (), self.bindings, self.inputs, dependencies,
            self.direct_errors, None,
        )

    def calculation_spec(self, call: ast.Call) -> CalculationSpec | None:
        if not isinstance(call.function, ast.Name):
            return None
        name = call.function
        if name.origin:
            if self.registry.has_calculation(name.name, name.origin):
                return self.registry.resolve_calculation(name.name, name.origin)
            return None
        if name.name in self.bindings and any(isinstance(arg, ast.NamedArgument) for arg in call.arguments):
            return self.local_spec(name.name)
        return None

    def infer_calculation_call(
        self, call: ast.Call, scope: dict[str, Type], expected: Type | None = None
    ) -> tuple[Type, CalculationSpec]:
        spec = self.calculation_spec(call)
        if spec is None:
            raise LangTypeError("Benannte Argumente benötigen eine aufrufbare Berechnung")
        if any(not isinstance(argument, ast.NamedArgument) for argument in call.arguments):
            raise LangTypeError("Berechnungsaufrufe dürfen benannte und positionale Argumente nicht mischen")
        arguments = list(call.arguments)
        names = [argument.name for argument in arguments]
        if len(names) != len(set(names)):
            raise LangTypeError("Aufruf enthält ein Argument mehrfach")
        allowed = spec.allowed_arguments()
        unknown = set(names) - allowed
        if unknown:
            raise LangNameError(f"Unbekanntes Argument {sorted(unknown)[0]!r} für {spec.symbol!r}")
        missing = spec.required_inputs(set(names))
        if missing:
            raise LangNameError(
                f"Aufruf von {spec.symbol!r} benötigt: {', '.join(sorted(missing))}"
            )
        for argument in arguments:
            argument_type = spec.output_types.get(argument.name)
            if argument_type is None:
                argument_type = spec.input_types.get(argument.name)
            if argument_type is None:
                argument_type = self.inputs.setdefault(argument.name, self.fresh())
                spec.input_types[argument.name] = argument_type
            self.infer(argument.value, scope, argument_type)
        result = spec.result_type
        if expected is not None:
            self.assign(expected, result)
            result = expected
        self.call_specs[id(call)] = spec
        return result, spec

    def infer_capture(self, access: ast.OutputAccess, scope: dict[str, Type]) -> Type:
        key = id(access.captured)
        if key not in self.capture_types:
            _, spec = self.infer_calculation_call(access.captured.call, scope)
            errors = spec.possible_errors({
                item.name for item in access.captured.call.arguments
                if isinstance(item, ast.NamedArgument)
            })
            by_name: dict[str, list[ErrorSpec]] = {}
            for error in errors:
                by_name.setdefault(error.name, []).append(error)
            caught: dict[str, ErrorSpec] = {}
            types: dict[str, Type] = {}
            reachable_outputs = {spec.symbol} | (spec.allowed_arguments() & set(spec.output_types))
            for name in access.captured.outputs:
                matches = by_name.get(name, [])
                if len(matches) > 1:
                    raise LangTypeError(f"Fehlerausgang {name!r} ist mehrdeutig")
                if matches:
                    caught[name] = matches[0]
                    payload = matches[0].payload or BOOL
                    types[name] = NullableType(payload)
                elif name in reachable_outputs:
                    types[name] = spec.output_types[name]
                else:
                    raise LangNameError(f"Berechnung {spec.symbol!r} besitzt keinen Ausgang {name!r}")
            if caught:
                for name in access.captured.outputs:
                    if name not in caught:
                        types[name] = self.as_nullable(types[name])
            self.capture_types[key] = types
            self.captures[key] = CaptureInfo(spec, caught)
        return self.capture_types[key][access.name]

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
        if isinstance(type_, MapType):
            return self.is_fully_resolved(type_.key, seen) and self.is_fully_resolved(type_.value, seen)
        if isinstance(type_, NullableType):
            return self.is_fully_resolved(type_.inner, seen)
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
        elif isinstance(type_, MapType):
            self.require_resolved(type_.key, context, seen)
            self.require_resolved(type_.value, context, seen)
        elif isinstance(type_, NullableType):
            self.require_resolved(type_.inner, context, seen)
        elif isinstance(type_, FunctionType):
            for item in [*type_.parameters, type_.result]:
                self.require_resolved(item, context, seen)
        elif isinstance(type_, RecordType):
            for field_name, field_type in type_.fields.items():
                self.require_resolved(field_type, f"{context}.{field_name}", seen)
