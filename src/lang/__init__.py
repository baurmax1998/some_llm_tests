"""Parser, Typprüfung und Laufzeit der zielorientierten Sprache."""
from .ast import Program
from .checker import CheckedProgram, TypeChecker, check
from .constraint import LangConstraint
from .errors import (
    LangError, LangIncompleteError, LangNameError, LangRuntimeError,
    LangSyntaxError, LangTypeError,
)
from .incremental import LangPrefixState, PrefixCheck, PrefixStatus, check_prefix
from .parser import Parser, parse
from .registry import ExternalFunction, ModuleRegistry
from .runtime import ExecutionResult, Evaluator, execute
from .types import BOOL, INT, STRING, FunctionType, ListType, RecordType, format_type


def compile_program(source: str, registry: ModuleRegistry | None = None) -> CheckedProgram:
    """Quelltext parsen und vollständig typprüfen."""
    return check(parse(source), registry)


def run(source: str, registry: ModuleRegistry | None = None) -> ExecutionResult:
    """Quelltext prüfen und anschließend sein Ziel auswerten."""
    checked = compile_program(source, registry)
    return execute(checked, registry)


__all__ = [
    "BOOL", "INT", "STRING", "CheckedProgram", "Evaluator", "ExecutionResult",
    "ExternalFunction", "FunctionType", "LangError", "LangIncompleteError",
    "LangConstraint", "LangNameError", "LangPrefixState", "LangRuntimeError", "LangSyntaxError",
    "LangTypeError", "ListType", "ModuleRegistry", "Parser", "PrefixCheck",
    "PrefixStatus", "Program", "RecordType", "TypeChecker", "check",
    "check_prefix", "compile_program", "execute", "format_type", "parse", "run",
]
