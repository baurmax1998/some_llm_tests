"""Replaceable JSON-Schema constraint adapter backed by Outlines Core."""
from __future__ import annotations
from abc import ABC, abstractmethod
import json
from typing import Any
import numpy as np
from jsonschema import Draft202012Validator, SchemaError


class UnsupportedSchemaError(ValueError):
    pass


class SchemaConstraint(ABC):
    @abstractmethod
    def allowed_token_mask(self) -> np.ndarray: ...
    @abstractmethod
    def accept_token(self, token_id: int) -> None: ...
    @abstractmethod
    def is_complete(self) -> bool: ...


_ANNOTATIONS = {"title", "description", "$comment", "$schema"}


def validate_supported_schema(schema: Any, path: str = "$") -> None:
    """Reject every constraint which this prototype did not promise to honor."""
    if not isinstance(schema, dict):
        raise UnsupportedSchemaError(f"{path}: Schema muss ein Objekt sein")
    try:
        if path == "$":
            Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise UnsupportedSchemaError(f"Ungültiges JSON Schema: {exc.message}") from exc

    schema_type = schema.get("type")
    if not isinstance(schema_type, str) or schema_type not in {"object", "string", "boolean", "integer", "array"}:
        raise UnsupportedSchemaError(
            f"{path}: nur object, string, boolean, integer und array werden unterstützt"
        )
    allowed_keys = {"type"} | _ANNOTATIONS
    if schema_type == "object":
        allowed_keys |= {"properties", "required", "additionalProperties"}
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            raise UnsupportedSchemaError(f"{path}: object benötigt 'properties'")
        if schema.get("additionalProperties") is not False:
            raise UnsupportedSchemaError(
                f"{path}: additionalProperties muss explizit false sein"
            )
        required = schema.get("required", [])
        if not isinstance(required, list) or any(x not in properties for x in required):
            raise UnsupportedSchemaError(f"{path}: 'required' verweist auf unbekannte Felder")
        for name, child in properties.items():
            validate_supported_schema(child, f"{path}.properties[{name!r}]")
    elif schema_type == "array":
        allowed_keys.add("items")
        if "items" not in schema or not isinstance(schema["items"], dict):
            raise UnsupportedSchemaError(f"{path}: array benötigt ein einzelnes 'items'-Schema")
        validate_supported_schema(schema["items"], f"{path}.items")
    elif schema_type == "string" and "enum" in schema:
        allowed_keys.add("enum")
        enum = schema["enum"]
        if not isinstance(enum, list) or not enum or not all(isinstance(v, str) for v in enum):
            raise UnsupportedSchemaError(f"{path}: String-enum muss eine nichtleere String-Liste sein")

    unknown = set(schema) - allowed_keys
    if unknown:
        raise UnsupportedSchemaError(
            f"{path}: nicht unterstützte Schlüssel: {', '.join(sorted(unknown))}"
        )


class OutlinesSchemaConstraint(SchemaConstraint):
    """Outlines Core Guide; tokenizer token strings are compiled as a vocabulary."""
    def __init__(self, schema: dict[str, Any], tokenizer: Any, vocab_size: int | None = None):
        validate_supported_schema(schema)
        from outlines_core import Guide, Index, Vocabulary
        from outlines_core.json_schema import build_regex_from_schema

        vocab = tokenizer.get_vocab()
        eos_id = tokenizer.eos_token_id
        eos_token = tokenizer.eos_token
        if eos_id is None or eos_token is None:
            raise ValueError("Tokenizer benötigt eos_token und eos_token_id")
        formatted: dict[str, list[int]] = {}
        for token, token_id in vocab.items():
            text = tokenizer.convert_tokens_to_string([token])
            formatted.setdefault(text, []).append(int(token_id))
        formatted.pop(tokenizer.convert_tokens_to_string([eos_token]), None)
        vocabulary = Vocabulary(int(eos_id), formatted)
        regex = build_regex_from_schema(json.dumps(schema, ensure_ascii=False))
        self._guide = Guide(Index(regex, vocabulary))
        tokenizer_size = max(int(i) for i in vocab.values()) + 1
        self._vocab_size = vocab_size or tokenizer_size
        if self._vocab_size < tokenizer_size:
            raise ValueError("Modellvokabular ist kleiner als das Tokenizervokabular")

    def allowed_token_mask(self) -> np.ndarray:
        mask = np.zeros(self._vocab_size, dtype=bool)
        ids = self._guide.get_tokens()
        mask[np.asarray(ids, dtype=np.int64)] = True
        return mask

    def accept_token(self, token_id: int) -> None:
        if not self._guide.accepts_tokens([int(token_id)]):
            raise ValueError(f"Token {token_id} ist im aktuellen Grammatikzustand unzulässig")
        self._guide.advance(int(token_id), return_tokens=False)

    def is_complete(self) -> bool:
        return bool(self._guide.is_finished())
