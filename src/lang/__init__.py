"""Parser, Typprüfung und Laufzeit der zielorientierten Sprache."""
from .ast import Program
from .checker import CheckedProgram, TypeChecker, check
from .errors import LangError, LangNameError, LangRuntimeError, LangSyntaxError, LangTypeError
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
    "ExternalFunction", "FunctionType", "LangError", "LangNameError",
    "LangRuntimeError", "LangSyntaxError", "LangTypeError", "ListType",
    "ModuleRegistry", "Parser", "Program", "RecordType", "TypeChecker", "check",
    "compile_program", "execute", "format_type", "parse", "run",
]
