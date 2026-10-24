"""Lokaler 5-Anfragen-Vergleich für KV-Cache und Ausgabeformat."""

import time

import pytest

from src.games.gamer import GamerCommandDecoder


SYSTEM_PROMPT = """Du steuerst Snake. Gib genau eine sichere Aktion aus.
Die erlaubten Aktionen und das exakte Ausgabeformat stehen in den nachfolgenden Anweisungen.
"""


def _request(index: int) -> str:
    return f"""Anfrage {index}. Die Schlange bewegt sich nach rechts. Gib genau eine Aktion aus.
ASCII:
+--------+
|        |
| oo@  * |
|        |
+--------+
"""


def _measure(backend, *, use_cache: bool, output_format: str) -> list[dict]:
    decoder = GamerCommandDecoder(
        backend, use_cache=use_cache, output_format=output_format
    )
    measurements = []
    for index in range(1, 6):
        started = time.perf_counter()
        _actions, text, metrics = decoder.generate_actions(
            _request(index),
            SYSTEM_PROMPT,
            temperature=0.0,
            top_p=1.0,
            seed=index,
            max_tokens=64,
        )
        measurements.append({
            "request": index,
            "format": output_format,
            "cache": use_cache,
            "output": text,
            "wall_seconds": time.perf_counter() - started,
            **metrics,
        })
    return measurements


def _print_table(groups: list[list[dict]]) -> None:
    print("\n| Format | KV-Cache | Anfrage | Cache-Treffer | Ausgabe | Prompt-Token | Output-Token | Modell | Gesamt |")
    print("|---|---:|---:|---:|---|---:|---:|---:|---:|")
    for values in groups:
        for value in values:
            print(
                f"| {value['format']} | {'ja' if value['cache'] else 'nein'} "
                f"| {value['request']} | {'ja' if value['cache_reused'] else 'nein'} "
                f"| `{value['output']}` | {value['prompt_tokens_processed']} | {value['generated_tokens']} "
                f"| {value['model_seconds']:.3f}s | {value['wall_seconds']:.3f}s |"
            )

    print("\n| Format | KV-Cache | 5 Anfragen gesamt | Output-Token gesamt |")
    print("|---|---:|---:|---:|")
    for values in groups:
        print(
            f"| {values[0]['format']} | {'ja' if values[0]['cache'] else 'nein'} "
            f"| {sum(value['wall_seconds'] for value in values):.3f}s "
            f"| {sum(value['generated_tokens'] for value in values)} |"
        )


@pytest.mark.integration
def test_five_requests_compare_cache_and_output_formats(backend):
    """Gibt eine Tabelle aus; Zeitwerte haben wegen Hardware keine harte Grenze."""
    groups = [
        _measure(backend, use_cache=False, output_format="json"),
        _measure(backend, use_cache=True, output_format="json"),
        _measure(backend, use_cache=False, output_format="compact"),
        _measure(backend, use_cache=True, output_format="compact"),
    ]
    _print_table(groups)

    for values in groups:
        if values[0]["cache"]:
            assert values[0]["cache_reused"] is False
            assert all(value["cache_reused"] is True for value in values[1:])
        else:
            assert all(value["cache_reused"] is False for value in values)

    compact_cached = groups[3]
    assert all(value["output"].endswith("!") for value in compact_cached)
