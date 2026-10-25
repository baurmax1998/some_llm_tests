"""Misst die Neuronenaktivierungen der Kategorisierung am reduzierten LM-Head.

Prompt, Kategorien und Testfälle werden absichtlich aus ``test_categorization``
übernommen, damit diese Messung direkt mit den dortigen Evals vergleichbar ist.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from tests.test_categorization import (
    CASES,
    CATEGORY_BY_CODE,
    ClassifierAnswer,
    classify_hidden_with_reduced_lm_head,
    prepare_digit_prompt,
)
from tests.test_typed import Category


TOP_NEURONS = 25


@dataclass(frozen=True)
class NeuronMeasurement:
    """Klassifikation und finaler Hidden State eines Testfalls."""

    answer: ClassifierAnswer
    activations: np.ndarray


@dataclass(frozen=True)
class NeuronScore:
    """Kategorie-Selektivität eines einzelnen Neurons."""

    neuron: int
    score: float
    mean_absolute_activation: float
    category_means: dict[Category, float]


def run_reduced_lm_head_and_measure_neurons(
    backend, context: str
) -> NeuronMeasurement:
    """Entspricht ``run_reduced_lm_head`` und gibt zusätzlich Aktivierungen zurück.

    Gemessen wird der normalisierte Hidden State des letzten Prompt-Tokens direkt
    vor dem LM-Head. Eine Kopie als NumPy-float32 macht die Werte unabhängig vom
    jeweils nächsten Modellaufruf und leicht weiterverarbeitbar.
    """
    inputs = prepare_digit_prompt(backend, context)
    hidden = backend.model.model(inputs, cache=backend.cache)[:, -1, :]
    answer = classify_hidden_with_reduced_lm_head(backend, hidden)

    hidden_float32 = hidden.astype(backend.mx.float32)
    backend.mx.eval(hidden_float32)
    activations = np.array(hidden_float32[0], dtype=np.float32, copy=True)
    return NeuronMeasurement(answer=answer, activations=activations)


def rank_category_selective_neurons(
    activations: np.ndarray,
    expected_categories: list[Category],
) -> list[NeuronScore]:
    """Ordnet Neuronen nach Varianz zwischen relativ zu Varianz in Kategorien.

    Ein hoher Score bedeutet, dass sich die mittlere Aktivierung zwischen den
    Kategorien stark unterscheidet und innerhalb einer Kategorie stabil ist.
    Der Score zeigt Korrelation, nicht zwingend einen kausalen Einfluss.
    """
    if activations.ndim != 2:
        raise ValueError("activations muss die Form (Testfälle, Neuronen) haben")
    if activations.shape[0] != len(expected_categories):
        raise ValueError("Für jeden Testfall wird genau eine Kategorie benötigt")

    categories = tuple(CATEGORY_BY_CODE.values())
    category_masks = [
        np.array([expected is category for expected in expected_categories])
        for category in categories
    ]
    if any(not mask.any() for mask in category_masks):
        raise ValueError("Jede Kategorie muss mindestens einen Testfall enthalten")

    category_means = np.stack(
        [activations[mask].mean(axis=0) for mask in category_masks]
    )
    overall_mean = activations.mean(axis=0)
    between_variance = ((category_means - overall_mean) ** 2).mean(axis=0)

    within_variance = np.stack(
        [
            ((activations[mask] - category_means[index]) ** 2).mean(axis=0)
            for index, mask in enumerate(category_masks)
        ]
    ).mean(axis=0)
    selectivity = between_variance / (within_variance + 1e-8)
    mean_absolute = np.abs(activations).mean(axis=0)

    scores = [
        NeuronScore(
            neuron=index,
            score=float(selectivity[index]),
            mean_absolute_activation=float(mean_absolute[index]),
            category_means={
                category: float(category_means[category_index, index])
                for category_index, category in enumerate(categories)
            },
        )
        for index in range(activations.shape[1])
    ]
    return sorted(scores, key=lambda result: result.score, reverse=True)


@pytest.mark.integration
def test_reduced_lm_head_neuron_measurements(backend):
    """Misst dieselben 20 Fälle und zeigt die selektivsten Neuronen an."""
    measurements: list[NeuronMeasurement] = []

    print(f"\n{'=' * 112}")
    print("NEURONENMESSUNG AM REDUZIERTEN LM-HEAD")
    print(f"{'=' * 112}", flush=True)

    for number, case in enumerate(CASES, start=1):
        measurement = run_reduced_lm_head_and_measure_neurons(
            backend, case.context
        )
        measurements.append(measurement)
        actual = measurement.answer.categorization.category
        marker = "✓" if actual is case.expected else "✗"
        print(
            f"[{number:>2}/{len(CASES)}] {marker} {case.name:<22} "
            f"erwartet={case.expected.value:<8} erhalten={actual.value:<8} "
            f"konfidenz={measurement.answer.confidence:.1%}",
            flush=True,
        )

    activation_matrix = np.stack(
        [measurement.activations for measurement in measurements]
    )
    ranking = rank_category_selective_neurons(
        activation_matrix,
        [case.expected for case in CASES],
    )

    correct = sum(
        measurement.answer.categorization.category is case.expected
        for case, measurement in zip(CASES, measurements)
    )
    print("\n" + "-" * 112)
    print(
        f"Korrekt: {correct}/{len(CASES)} | "
        f"Hidden-Größe: {activation_matrix.shape[1]} | "
        f"Top-{min(TOP_NEURONS, len(ranking))} nach Kategorie-Selektivität"
    )
    print("-" * 112)
    print(
        f"{'Neuron':>7} | {'Score':>12} | {'Ø |Aktivierung|':>16} | "
        + " | ".join(f"{category.value:>10}" for category in CATEGORY_BY_CODE.values())
    )
    print("-" * 112)
    for result in ranking[:TOP_NEURONS]:
        means = " | ".join(
            f"{result.category_means[category]:>10.4f}"
            for category in CATEGORY_BY_CODE.values()
        )
        print(
            f"{result.neuron:>7} | {result.score:>12.4f} | "
            f"{result.mean_absolute_activation:>16.4f} | {means}"
        )
    print("=" * 112, flush=True)

    # Modellqualität und Aktivierungswerte sind Messergebnisse, keine Testfehler.
    assert activation_matrix.shape == (len(CASES), activation_matrix.shape[1])
    assert activation_matrix.shape[1] > 0
    assert np.isfinite(activation_matrix).all()
    assert len(ranking) == activation_matrix.shape[1]
