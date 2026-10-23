"""Kommandozeile für Prüfen, Ausführen und lokale Codegenerierung."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .lang import LangConstraint, LangError, compile_program, run as run_program
from .lang.stdlib import standard_registry


LANG_SYSTEM_PROMPT = """Erzeuge ausschließlich Quelltext der zielorientierten Sprache.
Ein Programm enthält genau ein `goal name: Typ = Ausdruck` und optional `let`-Definitionen.
Typen sind Bool, Int, String und List<T>. Antworte ohne Markdown oder Erklärung.
"""


def _print_value(value: Any) -> None:
    if isinstance(value, (dict, list)):
        print(json.dumps(value, ensure_ascii=False, indent=2))
    elif isinstance(value, bool):
        print(str(value).lower())
    else:
        print(value)


def _source(path: str) -> tuple[Path, str]:
    file = Path(path)
    return file, file.read_text(encoding="utf-8")


def command_check(args: argparse.Namespace) -> int:
    file, source = _source(args.file)
    try:
        checked = compile_program(source, standard_registry(file.parent))
    except LangError as exc:
        print(exc.diagnostic(source, str(file)), file=sys.stderr)
        return 1
    print(f"OK: goal {checked.program.goal.name}: {checked.type_of(checked.program.goal.name)}")
    return 0


def command_run(args: argparse.Namespace) -> int:
    file, source = _source(args.file)
    try:
        result = run_program(source, standard_registry(file.parent))
    except LangError as exc:
        print(exc.diagnostic(source, str(file)), file=sys.stderr)
        return 1
    _print_value(result.value)
    return 0


def command_generate(args: argparse.Namespace) -> int:
    from .llm import DEFAULT_MODEL, MLXModelBackend, generate

    backend = MLXModelBackend(args.model or DEFAULT_MODEL)
    registry = standard_registry(Path.cwd())
    constraint = LangConstraint(backend.tokenizer, backend.vocab_size, registry=registry)
    trace = generate(
        backend=backend,
        constraint=constraint,
        schema=None,
        prompt=args.prompt,
        system_prompt=LANG_SYSTEM_PROMPT,
        output_validator=lambda source: bool(compile_program(source, registry)),
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        max_new_tokens=args.max_tokens,
    )
    if trace.status != "abgeschlossen":
        print(trace.error or trace.text, file=sys.stderr)
        return 1
    if args.output:
        Path(args.output).write_text(trace.text + "\n", encoding="utf-8")
    else:
        print(trace.text)
    return 0


def command_repl(args: argparse.Namespace) -> int:
    registry = standard_registry(Path.cwd())
    print("Lang REPL – Programm eingeben, mit leerer Zeile ausführen, :quit beendet.")
    lines: list[str] = []
    while True:
        try:
            line = input("... " if lines else ">>> ")
        except EOFError:
            print()
            return 0
        if not lines and line.strip() in {":q", ":quit"}:
            return 0
        if line.strip():
            lines.append(line)
            continue
        if not lines:
            continue
        source = "\n".join(lines)
        lines.clear()
        try:
            _print_value(run_program(source, registry).value)
        except LangError as exc:
            print(f"Fehler: {exc}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lang", description="Zielorientierte Programmiersprache")
    commands = parser.add_subparsers(dest="command", required=True)

    check_parser = commands.add_parser("check", help="Programm statisch prüfen")
    check_parser.add_argument("file")
    check_parser.set_defaults(handler=command_check)

    run_parser = commands.add_parser("run", help="Programm ausführen")
    run_parser.add_argument("file")
    run_parser.set_defaults(handler=command_run)

    generate_parser = commands.add_parser("generate", help="Programm mit lokalem LLM erzeugen")
    generate_parser.add_argument("prompt")
    generate_parser.add_argument("-o", "--output")
    generate_parser.add_argument("--model")
    generate_parser.add_argument("--temperature", type=float, default=0.0)
    generate_parser.add_argument("--top-p", type=float, default=1.0)
    generate_parser.add_argument("--seed", type=int, default=0)
    generate_parser.add_argument("--max-tokens", type=int, default=256)
    generate_parser.set_defaults(handler=command_generate)

    repl_parser = commands.add_parser("repl", help="Interaktive Sitzung starten")
    repl_parser.set_defaults(handler=command_repl)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (LangError, OSError, ValueError) as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
