"""Ergebnisse und Metadaten der statischen Typprüfung."""
from __future__ import annotations

from dataclasses import dataclass

from .. import ast
from ..calculation import CalculationSpec, ErrorSpec
from ..registry import ExternalFunction
from ..types import RecordType, Type, format_type


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
    external_calls: dict[int, ExternalFunction]
    captures: dict[int, CaptureInfo]

    def type_of(self, name: str) -> str:
        return format_type(self.bindings[name])
