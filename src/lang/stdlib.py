"""Kleine Standardbibliothek für Datei-, JSON- und Konsolenoperationen."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from .registry import ModuleRegistry
from .types import BOOL, STRING, FunctionType, TypeVariable


def standard_registry(root: str | Path | None = None) -> ModuleRegistry:
    """Erzeugt ein Register; relative Dateipfade bleiben optional unter ``root``."""
    base = Path(root).resolve() if root is not None else None
    registry = ModuleRegistry()

    def resolve_path(raw: str) -> Path:
        path = Path(raw)
        if base is None:
            return path
        resolved = (base / path).resolve() if not path.is_absolute() else path.resolve()
        try:
            resolved.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"Dateipfad verlässt das erlaubte Wurzelverzeichnis: {raw}") from exc
        return resolved

    def read_json(path: str) -> Any:
        with resolve_path(path).open(encoding="utf-8") as handle:
            return json.load(handle)

    def write_json(value: Any, path: str) -> bool:
        with resolve_path(path).open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return True

    def read_text(path: str) -> str:
        return resolve_path(path).read_text(encoding="utf-8")

    def write_text(value: str, path: str) -> bool:
        resolve_path(path).write_text(value, encoding="utf-8")
        return True

    def print_value(value: Any) -> Any:
        print(value)
        return value

    json_value = TypeVariable(-1)
    printed_value = TypeVariable(-2)
    registry.register("read_json", ("files", "std"), FunctionType([STRING], json_value), read_json)
    registry.register("write_json", ("files", "std"), FunctionType([json_value, STRING], BOOL), write_json)
    registry.register("read_text", ("files", "std"), FunctionType([STRING], STRING), read_text)
    registry.register("write_text", ("files", "std"), FunctionType([STRING, STRING], BOOL), write_text)
    registry.register(
        "print", ("console", "std"),
        FunctionType([printed_value], printed_value), print_value,
    )
    return registry
