"""Accuracy- und Laufzeitvergleich der Decoder und Modelloptimierungen."""
from __future__ import annotations

import csv
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import statistics
import time
from typing import Callable, Iterator

import pytest

from src.llm import MLXModelBackend
from tests.test_categorization import (
    ClassifierAnswer,
    run_digit_decoder,
    run_json_schema,
    run_reduced_lm_head,
)
from tests.test_typed import Category


NORMAL_SKIP_LAYERS = {7, 17, 34}
EXTREME_SKIP_LAYERS = {2, 7, 17, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35}
PRUNED_MODEL_PATH = Path("models/qwen3-categorization-pruned")


@dataclass(frozen=True)
class ComparisonCase:
    name: str
    context: str
    expected: Category


# Bewusst neue Texte, die nicht für Auswahl oder Pruning verwendet wurden.
COMPARISON_CASES = (
    ComparisonCase(
        "handball",
        "Die Handballmannschaft entscheidet das Endspiel mit einem Treffer in der letzten Sekunde.",
        Category.SPORT,
    ),
    ComparisonCase(
        "skispringen",
        "Beim Skispringen erreicht der Athlet nach einem weiten Sprung die höchste Punktzahl.",
        Category.SPORT,
    ),
    ComparisonCase(
        "basketball",
        "Die Basketballerin verwandelt kurz vor der Schlusssirene den entscheidenden Dreipunktewurf.",
        Category.SPORT,
    ),
    ComparisonCase(
        "boxtraining",
        "Der Boxer trainiert Beinarbeit und Ausdauer für seinen nächsten Wettkampf im Ring.",
        Category.SPORT,
    ),
    ComparisonCase(
        "klettern",
        "Die Kletterin bewältigt eine besonders schwierige Route in der nationalen Meisterschaft.",
        Category.SPORT,
    ),
    ComparisonCase(
        "oper",
        "Das Opernhaus bringt Mozarts Zauberflöte mit einem neuen Bühnenbild zur Aufführung.",
        Category.KULTUR,
    ),
    ComparisonCase(
        "skulptur",
        "Die Galerie zeigt monumentale Skulpturen einer zeitgenössischen Bildhauerin.",
        Category.KULTUR,
    ),
    ComparisonCase(
        "tanzfestival",
        "Auf dem Tanzfestival präsentieren internationale Ensembles moderne Choreografien.",
        Category.KULTUR,
    ),
    ComparisonCase(
        "filmretrospektive",
        "Das Kino widmet dem Regisseur eine Retrospektive mit seinen wichtigsten Filmen.",
        Category.KULTUR,
    ),
    ComparisonCase(
        "denkmalpflege",
        "Restauratoren legen die ursprünglichen Fresken in einer historischen Kapelle frei.",
        Category.KULTUR,
    ),
    ComparisonCase(
        "datenbank",
        "Die neue Datenbank verteilt Abfragen automatisch auf mehrere Server.",
        Category.TECHNIK,
    ),
    ComparisonCase(
        "feststoffakku",
        "Ein Feststoffakku soll Elektroautos eine größere Reichweite und kürzere Ladezeiten ermöglichen.",
        Category.TECHNIK,
    ),
    ComparisonCase(
        "dreidruck",
        "Der 3D-Drucker fertigt aus einer digitalen Vorlage ein passgenaues Ersatzteil.",
        Category.TECHNIK,
    ),
    ComparisonCase(
        "cloudplattform",
        "Die Cloudplattform skaliert zusätzliche Recheninstanzen abhängig von der aktuellen Auslastung.",
        Category.TECHNIK,
    ),
    ComparisonCase(
        "mikrochip",
        "Der Mikrochip integriert mehr Transistoren und beschleunigt die Verarbeitung von Sensordaten.",
        Category.TECHNIK,
    ),
    ComparisonCase(
        "smoke-test",
        "Nach jedem Build prüft ein Smoke-Test, ob die wichtigsten Programmfunktionen noch starten.",
        Category.TEST,
    ),
    ComparisonCase(
        "fahrpruefung",
        "Während der praktischen Fahrprüfung bewertet die Prüferin das sichere Verhalten im Straßenverkehr.",
        Category.TEST,
    ),
    ComparisonCase(
        "abnahmetest",
        "Vor der Übergabe kontrolliert der Abnahmetest alle vereinbarten Anforderungen des Systems.",
        Category.TEST,
    ),
    ComparisonCase(
        "materialpruefung",
        "Eine Materialprüfung misst, bei welcher Belastung die Metallprobe bricht.",
        Category.TEST,
    ),
    ComparisonCase(
        "quiz",
        "Im Wissenstest müssen die Teilnehmer fünfzig Fragen aus verschiedenen Fachgebieten beantworten.",
        Category.TEST,
    ),
)


@dataclass(frozen=True)
class BenchmarkResult:
    model_name: str
    method_name: str
    correct: int
    durations: tuple[float, ...]

    @property
    def accuracy(self) -> float:
        return self.correct / len(COMPARISON_CASES)

    @property
    def total_seconds(self) -> float:
        return sum(self.durations)

    @property
    def average_seconds(self) -> float:
        return statistics.mean(self.durations)

    @property
    def median_seconds(self) -> float:
        return statistics.median(self.durations)


@contextmanager
def skipped_transformer_layers(backend, skip_layers: set[int]) -> Iterator[None]:
    """Ersetzt Layer temporär durch Identitäten, sodass alle Decoder sie skippen."""
    if not skip_layers:
        yield
        return

    import mlx.nn as nn

    class IdentityTransformerLayer(nn.Module):
        def __call__(self, hidden, mask=None, cache=None):
            return hidden

    layers = backend.model.model.layers
    invalid = sorted(index for index in skip_layers if not 0 <= index < len(layers))
    if invalid:
        raise ValueError(f"Ungültige Skip-Layer: {invalid}")

    originals = {index: layers[index] for index in skip_layers}
    try:
        for index in skip_layers:
            layers[index] = IdentityTransformerLayer()
        yield
    finally:
        for index, layer in originals.items():
            layers[index] = layer


def benchmark_combination(
    backend,
    model_name: str,
    method_name: str,
    classifier: Callable[[object, str], ClassifierAnswer],
    skip_layers: set[int],
) -> BenchmarkResult:
    """Wärmt eine Kombination auf und misst danach alle neuen Fälle einzeln."""
    durations = []
    correct = 0
    with skipped_transformer_layers(backend, skip_layers):
        # Kompilierung und einmalige Initialisierung nicht der Inferenz zurechnen.
        classifier(backend, COMPARISON_CASES[0].context)

        for number, case in enumerate(COMPARISON_CASES, start=1):
            started = time.perf_counter()
            answer = classifier(backend, case.context)
            duration = time.perf_counter() - started
            durations.append(duration)
            actual = answer.categorization.category
            is_correct = actual is case.expected
            correct += is_correct
            print(
                f"{model_name:<20} | {method_name:<14} | "
                f"{number:>2}/{len(COMPARISON_CASES)} | "
                f"{'✓' if is_correct else '✗'} {case.name:<20} | "
                f"{duration:>7.3f}s | {actual.value}",
                flush=True,
            )

    return BenchmarkResult(
        model_name=model_name,
        method_name=method_name,
        correct=correct,
        durations=tuple(durations),
    )


@pytest.mark.integration
def test_categorization_accuracy_and_performance_comparison(backend):
    """Vergleicht 4 Modellvarianten × 3 Decoder auf 20 neuen Testfällen."""
    if not (PRUNED_MODEL_PATH / "config.json").exists():
        pytest.skip(
            "Gepruntes Modell fehlt; zuerst structural_mlp_channel_analysis ausführen"
        )

    print("\nLade das strukturell geprunte Vergleichsmodell ...", flush=True)
    load_started = time.perf_counter()
    pruned_backend = MLXModelBackend(
        str(PRUNED_MODEL_PATH), trust_remote_code=True
    )
    pruned_load_seconds = time.perf_counter() - load_started
    print(f"Gepruntes Modell geladen: {pruned_load_seconds:.3f}s", flush=True)

    methods = (
        ("JSON-Schema", run_json_schema),
        ("Ziffer-Decoder", run_digit_decoder),
        ("Reduzierter Head", run_reduced_lm_head),
    )
    variants = (
        ("Original", backend, set()),
        ("Skip normal", backend, NORMAL_SKIP_LAYERS),
        ("Skip extrem", backend, EXTREME_SKIP_LAYERS),
        ("Strukturell geprunt", pruned_backend, set()),
    )

    results = []
    for model_name, current_backend, skip_layers in variants:
        for method_name, classifier in methods:
            print(
                f"\n{'=' * 118}\n{model_name} × {method_name}\n{'=' * 118}",
                flush=True,
            )
            results.append(
                benchmark_combination(
                    current_backend,
                    model_name,
                    method_name,
                    classifier,
                    skip_layers,
                )
            )

    original_times = {
        result.method_name: result.average_seconds
        for result in results
        if result.model_name == "Original"
    }

    print("\n" + "=" * 118)
    print("GESAMTÜBERSICHT AUF 20 NEUEN TESTFÄLLEN")
    print("=" * 118)
    print(
        f"{'Modell':<20} | {'Methode':<14} | {'Korrekt':>7} | "
        f"{'Accuracy':>9} | {'Gesamt':>9} | {'Ø/Fall':>9} | "
        f"{'Median':>9} | {'Fälle/s':>8} | {'Speedup':>8}"
    )
    print("-" * 118)
    for result in results:
        speedup = original_times[result.method_name] / result.average_seconds
        print(
            f"{result.model_name:<20} | {result.method_name:<14} | "
            f"{result.correct:>2}/{len(COMPARISON_CASES):<4} | "
            f"{result.accuracy:>8.1%} | {result.total_seconds:>8.3f}s | "
            f"{result.average_seconds:>8.3f}s | {result.median_seconds:>8.3f}s | "
            f"{1 / result.average_seconds:>8.2f} | {speedup:>7.2f}x"
        )
    print("-" * 118)
    print("Speedup = Laufzeit des Originals mit demselben Decoder / aktuelle Laufzeit.")
    print(f"Ladezeit des geprunten Modells (nicht enthalten): {pruned_load_seconds:.3f}s")
    print("=" * 118, flush=True)

    output_path = Path("categorization_eval_logs/performance_comparison.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        fieldnames = (
            "model",
            "method",
            "correct",
            "total_cases",
            "accuracy",
            "total_seconds",
            "average_seconds",
            "median_seconds",
            "cases_per_second",
            "speedup_vs_original",
        )
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "model": result.model_name,
                    "method": result.method_name,
                    "correct": result.correct,
                    "total_cases": len(COMPARISON_CASES),
                    "accuracy": result.accuracy,
                    "total_seconds": result.total_seconds,
                    "average_seconds": result.average_seconds,
                    "median_seconds": result.median_seconds,
                    "cases_per_second": 1 / result.average_seconds,
                    "speedup_vs_original": (
                        original_times[result.method_name] / result.average_seconds
                    ),
                }
            )
    print(f"CSV: {output_path.resolve()}", flush=True)

    assert len(COMPARISON_CASES) == 20
    assert len(results) == len(variants) * len(methods) == 12
