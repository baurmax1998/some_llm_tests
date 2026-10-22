"""Konvertierung von Pydantic-Modellen in unterstützte JSON-Schemas."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Type

from ..mask.schema import UnsupportedSchemaError, validate_supported_schema


def schema_from_pydantic(model_type: Type[Any]) -> dict[str, Any]:
    """Create the supported schema subset from a Pydantic v2 model.

    Pydantic emits nested models through local ``$defs``/``$ref`` references.
    The decoding engine intentionally does not implement references, therefore
    this small boundary adapter inlines only Pydantic's local references. It
    rejects recursive models instead of weakening their schema.
    """
    try:
        from pydantic import BaseModel
    except ImportError as exc:  # pragma: no cover - pinned runtime dependency
        raise RuntimeError("Pydantic ist nicht installiert") from exc
    if not isinstance(model_type, type) or not issubclass(model_type, BaseModel):
        raise TypeError("model_type muss eine Pydantic-BaseModel-Klasse sein")

    original = model_type.model_json_schema(mode="validation")
    definitions = original.get("$defs", {})

    def inline(node: Any, resolving: tuple[str, ...] = ()) -> Any:
        if isinstance(node, list):
            return [inline(value, resolving) for value in node]
        if not isinstance(node, dict):
            return deepcopy(node)
        if "$ref" in node:
            reference = node["$ref"]
            prefix = "#/$defs/"
            if not isinstance(reference, str) or not reference.startswith(prefix):
                raise UnsupportedSchemaError(
                    f"Nur lokale Pydantic-$refs unter {prefix} werden unterstützt"
                )
            name = reference[len(prefix):]
            if name not in definitions:
                raise UnsupportedSchemaError(f"Unbekannte Pydantic-Referenz: {reference}")
            if name in resolving:
                raise UnsupportedSchemaError(
                    f"Rekursive Pydantic-Modelle werden nicht unterstützt: {name}"
                )
            resolved = inline(definitions[name], resolving + (name,))
            siblings = {key: value for key, value in node.items() if key != "$ref"}
            if siblings:
                resolved.update(inline(siblings, resolving))
            return resolved
        return {
            key: inline(value, resolving)
            for key, value in node.items()
            if key != "$defs"
        }

    schema = inline(original)
    validate_supported_schema(schema)
    return schema
