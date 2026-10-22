import json
from typing import Any
import jsonschema


def validate_output(text: str, schema: dict[str, Any]) -> Any:
    """Both required independent final checks; never repairs output."""
    value = json.loads(text)
    jsonschema.validate(instance=value, schema=schema)
    return value
