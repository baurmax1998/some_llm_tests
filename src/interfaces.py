"""Statische, maschinenlesbare Schnittstellen für SIC-Quelldateien."""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from itertools import combinations
import json
from pathlib import Path
import re
from typing import Iterable

from .lang import CheckedProgram, compile_program, standard_registry
from .lang.calculation import CalculationSpec, ErrorSpec
from .lang.registry import ModuleRegistry
from .lang.types import (
    FunctionType, ListType, MapType, NullableType, RecordType, Type, TypeVariable,
    format_type, prune,
)


@dataclass(frozen=True)
class InterfaceSource:
    path: Path
    checked: CheckedProgram


def _calculation(checked: CheckedProgram, symbol: str) -> CalculationSpec:
    from .lang.dependencies import expression_names

    return CalculationSpec(
        symbol=symbol,
        origin=(),
        output_types=dict(checked.bindings),
        input_types=dict(checked.inputs),
        dependencies={
            binding.name: expression_names(binding.value)
            for binding in checked.program.bindings
        },
        direct_errors=dict(checked.direct_errors),
        checked=checked,
    )


def _reachable_outputs(spec: CalculationSpec) -> set[str]:
    found = {spec.symbol}

    def visit(name: str) -> None:
        for dependency in spec.dependencies.get(name, set()):
            if dependency in spec.output_types and dependency not in found:
                found.add(dependency)
                visit(dependency)

    visit(spec.symbol)
    return found


def _default_order(spec: CalculationSpec, checked: CheckedProgram) -> list[str]:
    """Ordnet Defaults nach ihren Voraussetzungen und danach wie im Quelltext."""
    source_index = {
        binding.name: index for index, binding in enumerate(checked.program.bindings)
    }
    ordered: list[str] = []
    visited: set[str] = set()

    def visit(name: str) -> None:
        for dependency in sorted(
            spec.dependencies.get(name, set()),
            key=lambda item: source_index.get(item, len(source_index)),
        ):
            if dependency not in spec.output_types or dependency in visited:
                continue
            visited.add(dependency)
            visit(dependency)
            ordered.append(dependency)

    visit(spec.symbol)
    return ordered


def _active_overrides(spec: CalculationSpec, selected: set[str]) -> set[str]:
    """Entfernt Overrides, deren Vorfahr bereits den Teilgraphen abschneidet."""
    active: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        visited.add(name)
        for dependency in spec.dependencies.get(name, set()):
            if dependency in selected:
                active.add(dependency)
            elif dependency in spec.output_types:
                visit(dependency)

    visit(spec.symbol)
    return active


def _input_variants(spec: CalculationSpec) -> list[tuple[set[str], set[str]]]:
    """Liefert jeden wirksamen Schnitt durch den Abhängigkeitsgraphen.

    Der erste Wert ist die Menge überschriebener Zwischenwerte, der zweite die
    tatsächlich zu übergebende Argumentmenge einschließlich freier Eingaben.
    """
    internal = sorted(_reachable_outputs(spec) - {spec.symbol})
    variants: list[tuple[set[str], set[str]]] = []
    for size in range(len(internal) + 1):
        for items in combinations(internal, size):
            selected = set(items)
            if _active_overrides(spec, selected) != selected:
                continue
            arguments = selected | spec.required_inputs(selected)
            variants.append((selected, arguments))
    variants.sort(key=lambda item: (len(item[1]), sorted(item[1]), sorted(item[0])))
    return variants


def _interface_type(type_: Type) -> str:
    type_ = prune(type_)
    if isinstance(type_, ListType):
        return f"List<{_interface_type(type_.element)}>"
    if isinstance(type_, MapType):
        return f"Map<{_interface_type(type_.key)}, {_interface_type(type_.value)}>"
    if isinstance(type_, NullableType):
        return f"{_interface_type(type_.inner)}?"
    if isinstance(type_, RecordType):
        if type_.name:
            return type_.name
        fields = " ".join(
            f"{_interface_type(field_type)} {name};"
            for name, field_type in type_.fields.items()
        )
        return f"record {{ {fields} }}"
    if isinstance(type_, FunctionType):
        parameters = ", ".join(_interface_type(item) for item in type_.parameters)
        return f"({parameters}) -> {_interface_type(type_.result)}"
    return format_type(type_)


def _error_value(error: ErrorSpec) -> dict[str, object]:
    return {
        "name": error.name,
        "type": _interface_type(error.payload) if error.payload is not None else "Bool",
        "payload": error.payload is not None,
        "identity": "::".join(error.identity),
    }


def _collect_records(types: Iterable[Type]) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    visiting: set[int] = set()

    def visit(type_: Type) -> None:
        type_ = prune(type_)
        if id(type_) in visiting:
            return
        visiting.add(id(type_))
        if isinstance(type_, RecordType):
            for field_type in type_.fields.values():
                visit(field_type)
            if type_.name:
                records[type_.name] = {
                    name: _interface_type(field_type)
                    for name, field_type in sorted(type_.fields.items())
                }
        elif isinstance(type_, ListType):
            visit(type_.element)
        elif isinstance(type_, MapType):
            visit(type_.key)
            visit(type_.value)
        elif isinstance(type_, NullableType):
            visit(type_.inner)
        elif isinstance(type_, FunctionType):
            for item in (*type_.parameters, type_.result):
                visit(item)

    for item in types:
        visit(item)
    return dict(sorted(records.items()))


def _program_imports(program: object) -> list[dict[str, object]]:
    """Sammelt qualifizierte Typen und Symbole in Quelltextreihenfolge."""
    from .lang import ast

    imports: list[dict[str, object]] = []
    seen: set[tuple[str, ...]] = set()

    def add(name: str, origin: tuple[str, ...]) -> None:
        # SIC adressiert vom Symbol zur Herkunft (`write_json::files::std`).
        # Imports verwenden die aus C#/Java/TypeScript vertraute Richtung.
        path = (name, *origin)
        if path not in seen:
            seen.add(path)
            imports.append({
                "name": ".".join(reversed(path)),
                "path": path,
            })

    def visit(value: object) -> None:
        # Bei Top-Level-Bindings steht die Annotation syntaktisch vor dem Wert,
        # obwohl das AST-Feld ``value`` aus Laufzeitgründen zuerst definiert ist.
        if isinstance(value, ast.Binding):
            visit(value.annotation)
            visit(value.value)
            return
        if isinstance(value, ast.LocalLet):
            visit(value.annotation)
            visit(value.value)
            visit(value.body)
            return
        if isinstance(value, ast.Name) and value.origin:
            add(value.name, value.origin)
        elif isinstance(value, ast.TypeRef) and value.origin:
            add(value.name, value.origin)
        if is_dataclass(value):
            for descriptor in fields(value):
                if descriptor.name != "span":
                    visit(getattr(value, descriptor.name))
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)

    visit(program)
    return imports


def _default_text(expression: object, dependencies: set[str]) -> str:
    """Zeigt Konstanten direkt und sonst die abgeleiteten Abhängigkeiten an."""
    from .lang import ast

    if isinstance(expression, ast.Literal):
        if expression.value is None:
            return "null"
        if isinstance(expression.value, bool):
            return str(expression.value).lower()
        if isinstance(expression.value, str):
            return json.dumps(expression.value, ensure_ascii=False)
        return str(expression.value)
    if (
        isinstance(expression, ast.Unary)
        and expression.operator == "-"
        and isinstance(expression.operand, ast.Literal)
    ):
        return f"-{expression.operand.value}"
    arguments = ", ".join(sorted(dependencies))
    return f"derive({arguments})"


def describe_program(checked: CheckedProgram, source: str) -> dict[str, object]:
    """Erzeugt die vollständige Schnittstellenbeschreibung eines Programms."""
    calculations: list[dict[str, object]] = []
    binding_by_name = {binding.name: binding for binding in checked.program.bindings}
    all_types: list[Type] = [*checked.bindings.values(), *checked.inputs.values()]
    for errors in checked.direct_errors.values():
        all_types.extend(error.payload for error in errors if error.payload is not None)

    for binding in checked.program.bindings:
        spec = _calculation(checked, binding.name)
        reachable = _reachable_outputs(spec)
        # Die Reihenfolge folgt den Deklarationen im Quelltext statt der
        # Iteration über Sets. Das Goal steht dadurch zuerst, danach die Lets.
        output_names = [
            item.name for item in checked.program.bindings
            if item.name in reachable
        ]
        signatures: list[dict[str, object]] = []
        for overrides, arguments in _input_variants(spec):
            errors = sorted(
                spec.possible_errors(overrides),
                key=lambda error: (error.name, error.identity),
            )
            signatures.append({
                "inputs": {
                    name: _interface_type(
                        spec.output_types[name]
                        if name in spec.output_types else spec.input_types[name]
                    )
                    for name in sorted(arguments)
                },
                "overrides": sorted(overrides),
                "outputs": {
                    name: _interface_type(spec.output_types[name])
                    for name in output_names
                },
                "errors": [_error_value(error) for error in errors],
            })
        calculations.append({
            "name": binding.name,
            "type": _interface_type(spec.result_type),
            "goal": binding.is_goal,
            "defaults": {
                name: _default_text(
                    binding_by_name[name].value,
                    spec.dependencies.get(name, set()),
                )
                for name in _default_order(spec, checked)
            },
            "signatures": signatures,
        })

    return {
        "format": "sic-interface-v1",
        "source": source,
        "goal": checked.program.goal.name,
        "package": _package_name(source),
        "imports": _program_imports(checked.program),
        "types": _collect_records(all_types),
        "calculations": calculations,
    }


def _identifier(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return f"_{name}" if name[:1].isdigit() else name


def _package_name(source: str) -> str:
    path = Path(source)
    directories = () if path.parent.parts == (".",) else path.parent.parts
    return ".".join((*(_identifier(part) for part in directories), _identifier(path.stem)))


def _throws(errors: list[dict[str, object]]) -> str:
    rendered: list[str] = []
    for error in errors:
        value = str(error["name"])
        if error["payload"]:
            value += f"({error['type']})"
        identity = str(error["identity"])
        if not identity.startswith("__local__::"):
            value += f" from {identity}"
        rendered.append(value)
    return ", ".join(rendered)


def render_interface(description: dict[str, object]) -> str:
    """Rendert eine kompakte, C-ähnliche und weiterhin deterministische Ansicht."""
    source = str(description["source"])
    lines = [
        "// Generated by `sic describe`; do not edit.",
        f"// Source: {source}",
        "",
    ]
    package = description.get("package")
    if package:
        lines.extend([f"package {package};", ""])
    imports = description.get("imports", [])
    assert isinstance(imports, list)
    if imports:
        for imported in imports:
            if isinstance(imported, dict):
                lines.append(f"import {imported['name']};")
            else:  # Kompatibel zu bereits erzeugten Beschreibungsobjekten.
                lines.append(f"import {imported};")
        lines.append("")
    types = description["types"]
    assert isinstance(types, dict)
    for name, fields in types.items():
        lines.append(f"record {name} {{")
        assert isinstance(fields, dict)
        for field_name, field_type in fields.items():
            lines.append(f"    {field_type} {field_name};")
        lines.extend(["}", ""])

    calculations = description["calculations"]
    assert isinstance(calculations, list)
    for calculation_index, calculation in enumerate(calculations):
        assert isinstance(calculation, dict)
        signatures = calculation["signatures"]
        defaults = calculation["defaults"]
        assert isinstance(signatures, list)
        assert isinstance(defaults, dict)
        if calculation_index:
            lines.append("")
        if calculation["goal"]:
            lines.append("// goal")

        # Die kombinatorischen Signaturen bleiben im Analysemodell erhalten.
        # In der Textansicht fasst ein Default je überschreibbarem Zwischenwert
        # dieselben Varianten wesentlich kompakter zusammen.
        inputs: dict[str, str] = {}
        overrides: set[str] = set()
        outputs: dict[str, str] = {}
        errors_by_identity: dict[str, dict[str, object]] = {}
        for signature in signatures:
            assert isinstance(signature, dict)
            inputs.update(signature["inputs"])
            overrides.update(signature["overrides"])
            outputs.update(signature["outputs"])
            for error in signature["errors"]:
                errors_by_identity[str(error["identity"])] = error

        required = sorted(set(inputs) - overrides)
        optional = [name for name in defaults if name in overrides]
        optional.extend(sorted(overrides - set(optional)))
        parameters = [f"{inputs[name]} {name}" for name in required]
        parameters.extend(
            f"override {inputs[name]} {name} = {defaults.get(name, 'derived')}"
            for name in optional
        )
        if parameters:
            lines.append(f"{calculation['type']} {calculation['name']}(")
            for index, parameter in enumerate(parameters):
                comma = "," if index != len(parameters) - 1 else ""
                lines.append(f"    {parameter}{comma}")
            lines.append(")")
        else:
            lines.append(f"{calculation['type']} {calculation['name']}()")
        additional = {
            name: type_ for name, type_ in outputs.items()
            if name != calculation["name"]
        }
        if additional:
            lines.append("    outputs {")
            for name, type_ in additional.items():
                lines.append(f"        {type_} {name};")
            lines.append("    }")
        if errors_by_identity:
            errors = [errors_by_identity[key] for key in sorted(errors_by_identity)]
            lines.append(f"    throws {_throws(errors)}")
        lines[-1] += ";"

    lines.append("")
    return "\n".join(lines)


def _import_type(type_: Type, variables: dict[int, str]) -> str:
    type_ = prune(type_)
    if isinstance(type_, TypeVariable):
        if type_.identifier not in variables:
            index = len(variables)
            variables[type_.identifier] = chr(ord("T") + index) if index < 26 else f"T{index}"
        return variables[type_.identifier]
    if isinstance(type_, ListType):
        return f"List<{_import_type(type_.element, variables)}>"
    if isinstance(type_, MapType):
        return f"Map<{_import_type(type_.key, variables)}, {_import_type(type_.value, variables)}>"
    if isinstance(type_, NullableType):
        return f"{_import_type(type_.inner, variables)}?"
    if isinstance(type_, FunctionType):
        parameters = ", ".join(_import_type(item, variables) for item in type_.parameters)
        return f"({parameters}) -> {_import_type(type_.result, variables)}"
    if isinstance(type_, RecordType):
        if type_.name:
            return ".".join(reversed(type_.name.split("::")))
        fields = " ".join(
            f"{_import_type(value, variables)} {name};"
            for name, value in type_.fields.items()
        )
        return f"record {{ {fields} }}"
    rendered = format_type(type_)
    return ".".join(reversed(rendered.split("::"))) if "::" in rendered else rendered


def _type_import_paths(*types: Type) -> list[tuple[str, ...]]:
    found: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()

    def visit(type_: Type) -> None:
        type_ = prune(type_)
        name: str | None = None
        if isinstance(type_, RecordType):
            name = type_.name
            for field_type in type_.fields.values():
                visit(field_type)
        elif hasattr(type_, "name"):
            candidate = getattr(type_, "name")
            name = candidate if isinstance(candidate, str) else None
        if name and "::" in name:
            path = tuple(name.split("::"))
            if path not in seen:
                seen.add(path)
                found.append(path)
        if isinstance(type_, ListType):
            visit(type_.element)
        elif isinstance(type_, MapType):
            visit(type_.key)
            visit(type_.value)
        elif isinstance(type_, NullableType):
            visit(type_.inner)
        elif isinstance(type_, FunctionType):
            for item in (*type_.parameters, type_.result):
                visit(item)

    for item in types:
        visit(item)
    return found


def _import_header(package: str, dependencies: list[tuple[str, ...]]) -> list[str]:
    lines = [
        "// Generated import interface; do not edit.",
        "",
        f"package {package};",
        "",
    ]
    if dependencies:
        lines.extend(f"import {'.'.join(reversed(path))};" for path in dependencies)
        lines.append("")
    return lines


def _render_external_import(
    external: object, package: str, dependencies: list[tuple[str, ...]],
) -> str:
    variables: dict[int, str] = {}
    function = external.type
    parameter_types = [_import_type(item, variables) for item in function.parameters]
    result = _import_type(function.result, variables)
    generic = f"<{', '.join(variables.values())}>" if variables else ""
    names = external.parameter_names or tuple(
        f"arg{index}" for index in range(len(parameter_types))
    )
    lines = _import_header(package, dependencies)
    if parameter_types:
        lines.append(f"{result} {external.symbol}{generic}(")
        for index, (name, type_) in enumerate(zip(names, parameter_types)):
            comma = "," if index != len(parameter_types) - 1 else ""
            lines.append(f"    {type_} {name}{comma}")
        lines.append(");")
    else:
        lines.append(f"{result} {external.symbol}{generic}();")
    lines.append("")
    return "\n".join(lines)


def _render_type_import(
    symbol: str, type_: Type, package: str, dependencies: list[tuple[str, ...]],
) -> str:
    variables: dict[int, str] = {}
    type_ = prune(type_)
    lines = _import_header(package, dependencies)
    if isinstance(type_, RecordType):
        lines.append(f"record {symbol} {{")
        for name, field_type in type_.fields.items():
            lines.append(f"    {_import_type(field_type, variables)} {name};")
        lines.append("}")
    else:
        rendered = _import_type(type_, variables)
        lines.append(f"type {symbol} = {rendered};" if rendered != symbol else f"type {symbol};")
    lines.append("")
    return "\n".join(lines)


def _render_calculation_import(spec: CalculationSpec, path: tuple[str, ...]) -> str:
    source = "/".join(reversed(path)) + ".sic"
    description = describe_program(spec.checked, source)
    description["calculations"] = [
        calculation for calculation in description["calculations"]
        if calculation["name"] == spec.symbol
    ]
    return render_interface(description)


def _is_web_project(path: Path) -> Path | None:
    candidate = path if path.is_file() else path / "app.sic"
    return candidate if candidate.is_file() and candidate.name == "app.sic" else None


def _project_sources(config_path: Path) -> tuple[Path, list[Path], ModuleRegistry, Path]:
    # Der Import bleibt lokal, damit die Sprachwerkzeuge nicht von FastAPI abhängen.
    from .web import _project_registry, load_web_config

    config = load_web_config(config_path)
    registry = _project_registry(config.root, config.modules, config.endpoints)
    declared = [
        config_path.resolve(),
        *(module.source for module in config.modules),
        *config.pages,
        *(endpoint.source for endpoint in config.endpoints),
    ]
    discovered = sorted(
        item.resolve() for item in config.root.rglob("*.sic")
        if ".out" not in item.relative_to(config.root).parts
    )
    # Deklarierte Dateien kommen zuerst und erhalten den vollständigen
    # Projektkontext; zusätzliche SIC-Dateien der Codebasis werden ebenfalls
    # beschrieben, auch wenn sie noch nicht in app.sic eingebunden sind.
    unique = list(dict.fromkeys([*(path.resolve() for path in declared), *discovered]))
    return config.root, unique, registry, config.output / "i"


def _plain_sources(path: Path) -> tuple[Path, list[Path], ModuleRegistry, Path]:
    if path.is_file():
        if path.suffix != ".sic":
            raise ValueError("SIC-Quelldateien benötigen die Endung .sic")
        root, paths = path.parent, [path]
    elif path.is_dir():
        root = path
        paths = sorted(
            item for item in path.rglob("*.sic")
            if ".out" not in item.relative_to(path).parts
        )
    else:
        raise ValueError(f"Codebasis existiert nicht: {path}")
    if not paths:
        raise ValueError(f"Keine SIC-Dateien unter {path}")
    return root.resolve(), [item.resolve() for item in paths], standard_registry(root), root / ".out" / "i"


def describe_codebase(
    path: str | Path = ".", output: str | Path | None = None,
) -> list[Path]:
    """Beschreibt eine Datei oder Codebasis und schreibt gespiegelte ``.i``-Dateien.

    Bei einem Webprojekt werden die in ``app.sic`` deklarierten Module, Seiten
    und Endpunkte mit demselben Register wie beim Build analysiert.
    """
    requested = Path(path).resolve()
    config_path = _is_web_project(requested)
    if config_path is not None:
        root, paths, registry, default_output = _project_sources(config_path)
    else:
        root, paths, registry, default_output = _plain_sources(requested)
    output_root = Path(output).resolve() if output is not None else default_output

    generated: list[Path] = []
    destinations: set[Path] = set()
    imported_paths: dict[tuple[str, ...], str] = {}
    for source_path in paths:
        relative = source_path.relative_to(root)
        checked = compile_program(source_path.read_text(encoding="utf-8"), registry)
        description = describe_program(checked, relative.as_posix())
        for imported in description["imports"]:
            path_tuple = tuple(imported["path"])
            imported_paths[path_tuple] = str(imported["name"])
        destination = output_root / relative.with_suffix(".i")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(render_interface(description), encoding="utf-8")
        generated.append(destination)
        destinations.add(destination.resolve())

    # Jeder Import erhält eine gleichnamige Schnittstellendatei. Abhängige
    # Typen solcher Schnittstellen werden rekursiv ergänzt; `web.App` zieht
    # beispielsweise auch `web.Endpoint` nach.
    pending = list(imported_paths)
    processed: set[tuple[str, ...]] = set()
    while pending:
        path_tuple = pending.pop(0)
        if path_tuple in processed:
            continue
        processed.add(path_tuple)
        package = ".".join(reversed(path_tuple))
        symbol, origin = path_tuple[0], path_tuple[1:]
        destination = output_root.joinpath(*reversed(path_tuple)).with_suffix(".i")
        if destination.resolve() in destinations:
            continue

        dependencies: list[tuple[str, ...]] = []
        if registry.has_calculation(symbol, origin):
            content = _render_calculation_import(
                registry.resolve_calculation(symbol, origin), path_tuple
            )
        elif registry.has_function(symbol, origin):
            external = registry.resolve(symbol, origin)
            dependencies = [
                path for path in _type_import_paths(external.type)
                if path != path_tuple
            ]
            content = _render_external_import(external, package, dependencies)
        elif registry.has_type(symbol, origin):
            imported_type = registry.resolve_type(symbol, origin)
            dependencies = [
                path for path in _type_import_paths(imported_type)
                if path != path_tuple
            ]
            content = _render_type_import(
                symbol, imported_type, package, dependencies
            )
        else:  # Der Checker hätte unbekannte qualifizierte Namen bereits abgelehnt.
            continue

        for dependency in dependencies:
            if dependency not in processed:
                pending.append(dependency)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        generated.append(destination)
        destinations.add(destination.resolve())
    return generated
