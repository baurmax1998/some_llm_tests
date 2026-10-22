from __future__ import annotations
import argparse, json
from pathlib import Path
from .model_backend import DEFAULT_MODEL, MLXModelBackend
from .schema_engine import OutlinesSchemaConstraint, validate_supported_schema
from .generation import generate


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lokales JSON-Schema constrained decoding mit MLX")
    p.add_argument("--prompt", required=True)
    p.add_argument("--schema", required=True, type=Path, help="JSON-Schema-Datei")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--trace", type=Path, help="vollständigen Trace als JSON schreiben")
    return p


def main() -> int:
    args = parser().parse_args()
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    validate_supported_schema(schema)
    backend = MLXModelBackend(args.model)
    constraint = OutlinesSchemaConstraint(schema, backend.tokenizer, backend.vocab_size)
    result = generate(backend, constraint, schema, args.prompt, args.temperature,
                      args.top_p, args.seed, args.max_new_tokens)
    print(result.text)
    print(f"\nStatus: {result.status}")
    if result.error: print(result.error)
    if args.trace: args.trace.write_text(result.to_json(), encoding="utf-8")
    return 0 if result.status == "abgeschlossen" else 1


if __name__ == "__main__":
    raise SystemExit(main())
