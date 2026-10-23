"""Typprüfung mit Vorwärtsreferenzen und rückwärts propagierten Erwartungen."""
from __future__ import annotations
from dataclasses import dataclass, fields, is_dataclass
from . import ast
from .calculation import CalculationSpec, ErrorSpec
from .dependencies import check_definition_order, check_dependency_cycles
from .errors import LangNameError, LangTypeError
from .registry import ModuleRegistry
from .types import (
    BOOL, FLOAT, INT, NULL, STRING, FunctionType, ListType, MapType,
    NullableType, Primitive, RecordType, Type, TypeVariable, format_type, prune,
    unify,
)


@dataclass(frozen=True)
class CaptureInfo:
    spec: CalculationSpec
    caught: dict[str, ErrorSpec]


@dataclass(frozen=True)
class CheckedProgram:
    program: ast.Program
    bindings: dict[str, Type]
    records: dict[str, RecordType]
    inputs: dict[str, Type]
    direct_errors: dict[str, tuple[ErrorSpec, ...]]
    call_specs: dict[int, CalculationSpec]
    captures: dict[int, CaptureInfo]

    def type_of(self, name: str) -> str:
        return format_type(self.bindings[name])


class TypeChecker:
    def __init__(self, registry: ModuleRegistry | None = None):
        self.registry = registry or ModuleRegistry()
        self.counter = 0
        self.records: dict[str, RecordType] = {}
        self.bindings: dict[str, Type] = {}
        self.inputs: dict[str, Type] = {}
        self.direct_errors: dict[str, tuple[ErrorSpec, ...]] = {}
        self.error_payloads: dict[str, Type | None] = {}
        self.call_specs: dict[int, CalculationSpec] = {}
        self.captures: dict[int, CaptureInfo] = {}
        self.capture_types: dict[int, dict[str, Type]] = {}
        self.program: ast.Program | None = None

    def fresh(self) -> TypeVariable:
        result = TypeVariable(self.counter)
        self.counter += 1
        return result

    def from_ref(self, reference: ast.TypeRef) -> Type:
        if reference.name == "Bool": result: Type = BOOL
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
            dict(self.direct_errors), dict(self.call_specs), dict(self.captures),
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
        from .dependencies import expression_names
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

    def infer(self, expression: ast.Expr, scope: dict[str, Type], expected: Type | None = None) -> Type:
        result: Type
        if isinstance(expression, ast.Literal):
            if expression.value is None:
                result = NULL
            elif isinstance(expression.value, bool):
                result = BOOL
            elif isinstance(expression.value, int):
                result = INT
            elif isinstance(expression.value, float):
                result = FLOAT
            else:
                result = STRING
        elif isinstance(expression, ast.Name):
            if expression.origin:
                if self.registry.has_calculation(expression.name, expression.origin):
                    raise LangTypeError("Eine Berechnung muss mit benannten Argumenten aufgerufen werden")
                result = self.registry.resolve(expression.name, expression.origin).type
            else:
                if expression.name in scope:
                    result = scope[expression.name]
                else:
                    # Freie Namen sind Eingaben der umgebenden Berechnung.
                    result = self.inputs.setdefault(expression.name, self.fresh())
        elif isinstance(expression, ast.Annotation):
            result = self.from_ref(expression.type_ref)
            self.infer(expression.value, scope, result)
        elif isinstance(expression, ast.ErrorExpr):
            payload = self.error_payloads[expression.name]
            if expression.payload is not None and payload is not None:
                self.infer(expression.payload, scope, payload)
            # `err` ist ein Bottom-Ausdruck: der erwartete Erfolgswert wird
            # nicht durch den Fehler eingeschränkt.
            result = expected or self.fresh()
        elif isinstance(expression, ast.OutputAccess):
            result = self.infer_capture(expression, scope)
        elif isinstance(expression, ast.ListExpr):
            element = prune(expected).element if isinstance(prune(expected), ListType) else self.fresh()
            for item in expression.items:
                self.infer(item, scope, element)
            result = ListType(element)
        elif isinstance(expression, ast.RecordExpr):
            expected_record = prune(expected) if expected is not None else None
            if not expression.fields and isinstance(expected_record, MapType):
                return expected_record
            names = [name for name, _ in expression.fields]
            if len(names) != len(set(names)):
                raise LangTypeError("Record enthält doppelte Felder")
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
        elif isinstance(expression, ast.MapExpr):
            expected_map = prune(expected) if expected is not None else None
            key_type = expected_map.key if isinstance(expected_map, MapType) else self.fresh()
            value_type = expected_map.value if isinstance(expected_map, MapType) else self.fresh()
            for key, value in expression.entries:
                self.infer(key, scope, key_type)
                self.infer(value, scope, value_type)
            result = MapType(key_type, value_type)
        elif isinstance(expression, ast.LocalLet):
            local_type = self.from_ref(expression.annotation) if expression.annotation else self.fresh()
            self.infer(expression.value, scope, local_type)
            local = dict(scope)
            local[expression.name] = local_type
            result = self.infer(expression.body, local, expected)
        elif isinstance(expression, ast.Unary):
            wanted = BOOL if expression.operator == "not" else INT
            self.infer(expression.operand, scope, wanted)
            result = wanted
        elif isinstance(expression, ast.Binary):
            result = self.infer_binary(expression, scope, expected)
        elif isinstance(expression, ast.IfExpr):
            self.infer(expression.condition, scope, BOOL)
            branch = expected or self.fresh()
            then_scope, else_scope = self.narrow_scopes(expression.condition, scope)
            self.infer(expression.then_branch, then_scope, branch)
            self.infer(expression.else_branch, else_scope, branch)
            result = branch
        elif isinstance(expression, ast.MatchExpr):
            subject = prune(self.infer(expression.value, scope))
            branch = expected or self.fresh()
            null_covered = False
            catch_all = False
            for case_index, case in enumerate(expression.cases):
                local = dict(scope)
                if isinstance(case.pattern, ast.NullPattern):
                    if not isinstance(subject, NullableType):
                        raise LangTypeError("null-Muster benötigt einen nullable Typ")
                    null_covered = True
                elif isinstance(case.pattern, ast.LiteralPattern):
                    self.infer(ast.Literal(case.pattern.value), scope, subject)
                elif isinstance(case.pattern, ast.BindingPattern):
                    local[case.pattern.name] = (
                        subject.inner if isinstance(subject, NullableType) and null_covered else subject
                    )
                    catch_all = True
                elif isinstance(case.pattern, ast.WildcardPattern):
                    catch_all = True
                if catch_all and case_index != len(expression.cases) - 1:
                    raise LangTypeError("Binding- oder _-Muster muss der letzte match-Fall sein")
                self.infer(case.body, local, branch)
            if not catch_all:
                raise LangTypeError("match benötigt ein abschließendes Binding- oder _-Muster")
            if isinstance(subject, NullableType) and not null_covered:
                raise LangTypeError("match über nullable Typ benötigt einen null-Fall")
            result = branch
        elif isinstance(expression, ast.SafeField):
            owner = prune(self.infer(expression.value, scope))
            if not isinstance(owner, NullableType):
                raise LangTypeError("'?.' benötigt einen nullable Wert")
            record = prune(owner.inner)
            if isinstance(record, TypeVariable):
                inferred_record = RecordType(None, {expression.name: self.fresh()})
                unify(record, inferred_record)
                record = inferred_record
            if not isinstance(record, RecordType):
                raise LangTypeError(f"Sicherer Feldzugriff auf {format_type(record)}")
            if record.closed and expression.name not in record.fields:
                raise LangTypeError(f"Record besitzt kein Feld {expression.name!r}")
            result = self.as_nullable(record.fields.setdefault(expression.name, self.fresh()))
        elif isinstance(expression, ast.SafeMethodCall):
            owner = prune(self.infer(expression.value, scope))
            if not isinstance(owner, NullableType):
                raise LangTypeError("'?.' benötigt einen nullable Wert")
            synthetic = "__safe_receiver__"
            local = dict(scope)
            local[synthetic] = owner.inner
            method = ast.MethodCall(ast.Name(synthetic), expression.name, expression.arguments)
            result = self.as_nullable(self.infer_method(method, local, None))
        elif isinstance(expression, ast.Index):
            owner = prune(self.infer(expression.value, scope))
            if isinstance(owner, ListType):
                self.infer(expression.index, scope, INT)
                result = owner.element
            elif isinstance(owner, MapType):
                self.infer(expression.index, scope, owner.key)
                result = owner.value
            elif owner == STRING:
                self.infer(expression.index, scope, INT)
                result = STRING
            else:
                raise LangTypeError(f"Indexzugriff auf {format_type(owner)}")
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
            calculation = self.calculation_spec(expression)
            if calculation is not None or any(isinstance(arg, ast.NamedArgument) for arg in expression.arguments):
                result, _ = self.infer_calculation_call(expression, scope, expected)
            else:
                arguments = [self.infer(argument, scope) for argument in expression.arguments]
                result = expected or self.fresh()
                self.infer(expression.function, scope, FunctionType(arguments, result))
        elif isinstance(expression, ast.MethodCall):
            result = self.infer_method(expression, scope, expected)
        else:  # pragma: no cover
            raise AssertionError(type(expression))
        if expected is not None:
            self.assign(expected, result)
            return expected
        return result

    def assign(self, expected: Type, actual: Type) -> None:
        expected, actual = prune(expected), prune(actual)
        if isinstance(expected, NullableType):
            if actual == NULL:
                return
            if isinstance(actual, NullableType):
                unify(expected.inner, actual.inner)
            else:
                unify(expected.inner, actual)
            return
        unify(expected, actual)

    def as_nullable(self, type_: Type) -> NullableType:
        type_ = prune(type_)
        return type_ if isinstance(type_, NullableType) else NullableType(type_)

    def narrow_scopes(
        self, condition: ast.Expr, scope: dict[str, Type]
    ) -> tuple[dict[str, Type], dict[str, Type]]:
        then_scope, else_scope = dict(scope), dict(scope)
        if isinstance(condition, ast.Binary) and condition.operator in {"==", "!="}:
            name = None
            if isinstance(condition.left, ast.Name) and isinstance(condition.right, ast.Literal) and condition.right.value is None:
                name = condition.left.name
            elif isinstance(condition.right, ast.Name) and isinstance(condition.left, ast.Literal) and condition.left.value is None:
                name = condition.right.name
            if name in scope:
                current = prune(scope[name])
                if isinstance(current, NullableType):
                    non_null_scope = then_scope if condition.operator == "!=" else else_scope
                    non_null_scope[name] = current.inner
        return then_scope, else_scope

    def infer_binary(self, expression: ast.Binary, scope: dict[str, Type], expected: Type | None) -> Type:
        operator = expression.operator
        if operator == "??":
            result = expected or self.fresh()
            self.infer(expression.left, scope, NullableType(result))
            self.infer(expression.right, scope, result)
            return result
        if operator in {"and", "or"}:
            self.infer(expression.left, scope, BOOL); self.infer(expression.right, scope, BOOL)
            return BOOL
        if operator in {"<", "<=", ">", ">="}:
            operand = prune(self.infer(expression.left, scope))
            if operand not in (INT, FLOAT, STRING):
                raise LangTypeError(f"{operator!r} unterstützt nicht {format_type(operand)}")
            self.infer(expression.right, scope, operand)
            return BOOL
        if operator in {"==", "!="}:
            operand = self.infer(expression.left, scope)
            self.infer(expression.right, scope, operand)
            return BOOL
        if operator in {"-", "*", "%", "/"}:
            wanted = prune(expected) if expected is not None else None
            if wanted not in (INT, FLOAT):
                wanted = prune(self.infer(expression.left, scope))
            if wanted not in (INT, FLOAT):
                raise LangTypeError(f"{operator!r} benötigt Int oder Float")
            if operator == "/" and wanted != FLOAT:
                raise LangTypeError("'/' benötigt Float-Operanden")
            self.infer(expression.left, scope, wanted)
            self.infer(expression.right, scope, wanted)
            return wanted
        if operator == "+":
            wanted = prune(expected) if expected is not None else None
            if isinstance(wanted, ListType):
                self.infer(expression.left, scope, wanted)
                self.infer(expression.right, scope, wanted)
                return wanted
            if wanted not in (INT, FLOAT, STRING):
                left = prune(self.infer(expression.left, scope))
                if left not in (INT, FLOAT, STRING) and not isinstance(left, ListType):
                    raise LangTypeError("'+' benötigt Zahlen, Strings oder Listen")
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
        if isinstance(collection, MapType):
            if expression.name == "keys" and not expression.arguments:
                return ListType(collection.key)
            if expression.name == "values" and not expression.arguments:
                return ListType(collection.value)
            if expression.name == "get" and len(expression.arguments) == 1:
                self.infer(expression.arguments[0], scope, collection.key)
                return NullableType(collection.value)
            if expression.name == "contains_key" and len(expression.arguments) == 1:
                self.infer(expression.arguments[0], scope, collection.key)
                return BOOL
            raise LangTypeError(f"Unbekannte Map-Operation .{expression.name}")
        if not isinstance(collection, ListType):
            raise LangTypeError(f".{expression.name} benötigt eine Liste oder Map")
        element = collection.element
        if expression.name == "filter" and len(expression.arguments) == 1:
            self.infer(expression.arguments[0], scope, FunctionType([element], BOOL))
            return collection
        if expression.name == "map" and len(expression.arguments) == 1:
            mapped = prune(expected).element if isinstance(prune(expected), ListType) else self.fresh()
            self.infer(expression.arguments[0], scope, FunctionType([element], mapped))
            return ListType(mapped)
        if expression.name == "flat_map" and len(expression.arguments) == 1:
            mapped = prune(expected).element if isinstance(prune(expected), ListType) else self.fresh()
            self.infer(expression.arguments[0], scope, FunctionType([element], ListType(mapped)))
            return ListType(mapped)
        if expression.name == "sum" and not expression.arguments:
            numeric = prune(element)
            if numeric not in (INT, FLOAT):
                raise LangTypeError("sum benötigt List<Int> oder List<Float>")
            return numeric
        if expression.name == "fold" and len(expression.arguments) == 2:
            accumulator = self.infer(expression.arguments[0], scope, expected)
            self.infer(expression.arguments[1], scope, FunctionType([accumulator, element], accumulator))
            return accumulator
        if expression.name in {"any", "all"} and len(expression.arguments) == 1:
            self.infer(expression.arguments[0], scope, FunctionType([element], BOOL))
            return BOOL
        if expression.name == "count" and not expression.arguments:
            return INT
        if expression.name in {"take", "drop"} and len(expression.arguments) == 1:
            self.infer(expression.arguments[0], scope, INT)
            return collection
        if expression.name in {"first", "last"} and not expression.arguments:
            return NullableType(element)
        if expression.name == "find" and len(expression.arguments) == 1:
            self.infer(expression.arguments[0], scope, FunctionType([element], BOOL))
            return NullableType(element)
        if expression.name == "sort" and not expression.arguments:
            if prune(element) not in (INT, FLOAT, STRING):
                raise LangTypeError("sort benötigt vergleichbare Elemente")
            return collection
        if expression.name == "group_by" and len(expression.arguments) == 1:
            key = self.fresh()
            self.infer(expression.arguments[0], scope, FunctionType([element], key))
            return MapType(key, ListType(element))
        raise LangTypeError(f"Unbekannte oder falsch verwendete Listenoperation .{expression.name}")


def check(program: ast.Program, registry: ModuleRegistry | None = None) -> CheckedProgram:
    return TypeChecker(registry).check(program)
