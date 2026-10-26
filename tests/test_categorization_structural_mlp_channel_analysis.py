"""Analyse echter Qwen3-MLP-Kanäle als Vorbereitung für strukturelles Pruning.

Prompt und Testfälle kommen aus ``test_categorization``. Gemessen wird der
SwiGLU-Ausgang zwischen ``gate_proj``/``up_proj`` und ``down_proj``. Ein Kanal
an dieser Stelle lässt sich strukturell entfernen, indem dieselbe Zeile aus
``gate_proj`` und ``up_proj`` sowie dieselbe Spalte aus ``down_proj`` entfällt.
"""
import csv
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import pytest

from src.llm.structural_pruning import (
    align_channels_for_quantized_export,
    export_structurally_pruned_qwen3,
)
from tests.test_categorization import (
    CASES,
    CATEGORY_BY_CODE,
    ClassifierAnswer,
    decode_category,
    prepare_digit_prompt,
)
from tests.test_typed import Category


PRUNING_RATIOS = (
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.10,
    0.11,
    0.12,
    0.13,
    0.14,
    0.15,
    0.175,
    0.20,
)
ACTIVATION_THRESHOLD = float(os.getenv("NEURON_ACTIVATION_THRESHOLD", "0.01"))


@dataclass(frozen=True)
class MLPChannelStatistics:
    """Vektorisierte Statistiken aller strukturellen MLP-Kanäle."""

    selectivity_scores: np.ndarray
    mean_absolute_activations: np.ndarray
    activation_rates: np.ndarray


@dataclass(frozen=True)
class ForwardResult:
    answer: ClassifierAnswer
    category_logits: np.ndarray
    # Pro Layer der SwiGLU-Ausgang des letzten Prompt-Tokens.
    mlp_activations: tuple[np.ndarray, ...]


def _category_token_ids(backend) -> tuple[int, ...]:
    token_ids = []
    for code in CATEGORY_BY_CODE:
        encoded = backend.tokenizer.encode(code, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"Kategorie-Code {code!r} ist kein einzelnes Token")
        token_ids.append(encoded[0])
    return tuple(token_ids)


def _category_logits(backend, hidden):
    """Projiziert den finalen Hidden State nur auf die vier Kategorie-Tokens."""
    mx = backend.mx
    head = backend.model.lm_head
    selected_ids = mx.array(_category_token_ids(backend))
    if hasattr(head, "scales"):
        logits = mx.quantized_matmul(
            hidden,
            head.weight[selected_ids],
            scales=head.scales[selected_ids],
            biases=head.biases[selected_ids] if head.biases is not None else None,
            transpose=True,
            group_size=head.group_size,
            bits=head.bits,
            mode=head.mode,
        )
    else:
        logits = hidden @ head.weight[selected_ids].T
    if hasattr(head, "bias"):
        logits = logits + head.bias[selected_ids]
    return logits


def run_with_mlp_channel_ablation(
    backend,
    context: str,
    ablated_channels: set[tuple[int, int]] | None = None,
    *,
    collect_activations: bool = False,
) -> ForwardResult:
    """Führt Qwen3 aus und deaktiviert ausgewählte MLP-Kanäle im ganzen Prompt.

    ``(layer, channel)`` bezeichnet einen SwiGLU-Kanal. Er wird für sämtliche
    Tokenpositionen auf null gesetzt. Das simuliert funktional das gemeinsame
    Entfernen seiner gate-/up-Zeilen und seiner down-Spalte, ohne Gewichte zu
    verändern. Ohne Ablation entspricht der Klassifikationspfad dem reduzierten
    LM-Head aus ``test_categorization``.
    """
    from mlx_lm.models.activations import swiglu
    from mlx_lm.models.base import create_attention_mask

    ablated_channels = ablated_channels or set()
    inputs = prepare_digit_prompt(backend, context)
    transformer = backend.model.model
    layers = transformer.layers
    layer_count = len(layers)
    intermediate_size = int(transformer.args.intermediate_size)

    invalid = sorted(
        (layer, channel)
        for layer, channel in ablated_channels
        if not 0 <= layer < layer_count or not 0 <= channel < intermediate_size
    )
    if invalid:
        raise ValueError(f"Ungültige MLP-Kanäle: {invalid}")

    channels_by_layer: dict[int, list[int]] = {}
    for layer, channel in ablated_channels:
        channels_by_layer.setdefault(layer, []).append(channel)

    hidden = transformer.embed_tokens(inputs)
    attention_mask = create_attention_mask(hidden, backend.cache[0])
    measured: list[np.ndarray] = []

    for layer_index, (layer, cache) in enumerate(zip(layers, backend.cache)):
        attention_output = layer.self_attn(
            layer.input_layernorm(hidden), attention_mask, cache
        )
        residual = hidden + attention_output
        mlp_input = layer.post_attention_layernorm(residual)
        mlp_hidden = swiglu(
            layer.mlp.gate_proj(mlp_input),
            layer.mlp.up_proj(mlp_input),
        )

        if collect_activations:
            last_token = mlp_hidden[0, -1].astype(backend.mx.float32)
            backend.mx.eval(last_token)
            measured.append(np.array(last_token, dtype=np.float32, copy=True))

        layer_channels = channels_by_layer.get(layer_index)
        if layer_channels:
            # Auf alle Prompt-Tokens anwenden: Das ist eine Kanal-Ablation und
            # nicht bloß eine Manipulation der finalen Repräsentation.
            channel_mask = np.ones(intermediate_size, dtype=np.float32)
            channel_mask[layer_channels] = 0.0
            mlp_hidden = mlp_hidden * backend.mx.array(channel_mask).astype(
                mlp_hidden.dtype
            )

        hidden = residual + layer.mlp.down_proj(mlp_hidden)

    hidden = transformer.norm(hidden)[:, -1, :]
    logits = _category_logits(backend, hidden)
    probabilities = backend.mx.softmax(logits, axis=-1)
    backend.mx.eval(logits, probabilities)
    selected_index = int(backend.mx.argmax(logits, axis=-1).item())
    code = tuple(CATEGORY_BY_CODE)[selected_index]
    answer = ClassifierAnswer(
        categorization=decode_category(code),
        raw_answer=code,
        confidence=float(probabilities[0, selected_index].item()),
    )
    logits_float32 = logits.astype(backend.mx.float32)
    backend.mx.eval(logits_float32)
    return ForwardResult(
        answer=answer,
        category_logits=np.array(logits_float32[0], dtype=np.float32, copy=True),
        mlp_activations=tuple(measured),
    )


def calculate_mlp_channel_statistics(
    activations: np.ndarray,
    expected_categories: list[Category],
    *,
    activation_threshold: float = ACTIVATION_THRESHOLD,
) -> MLPChannelStatistics:
    """Berechnet Kategorie-Selektivität und Aktivität aller MLP-Kanäle.

    ``activations`` hat die Form ``(Fälle, Layer, MLP-Kanäle)``. Der
    Selektivitäts-Score ist die Varianz der Klassenmittelwerte geteilt durch die
    mittlere Varianz innerhalb der Klassen. Die Pruning-Kurve entfernt zuerst
    die Kanäle mit den niedrigsten Scores und prüft deren kausalen Einfluss.
    """
    if activations.ndim != 3:
        raise ValueError("activations muss (Testfälle, Layer, Kanäle) enthalten")
    if activations.shape[0] != len(expected_categories):
        raise ValueError("Für jeden Testfall wird genau eine Kategorie benötigt")

    categories = tuple(CATEGORY_BY_CODE.values())
    masks = [
        np.array([expected is category for expected in expected_categories])
        for category in categories
    ]
    if any(not mask.any() for mask in masks):
        raise ValueError("Jede Kategorie muss mindestens einen Testfall enthalten")

    category_means = np.stack([activations[mask].mean(axis=0) for mask in masks])
    overall_mean = activations.mean(axis=0)
    between_variance = ((category_means - overall_mean) ** 2).mean(axis=0)
    within_variance = np.stack(
        [
            ((activations[mask] - category_means[index]) ** 2).mean(axis=0)
            for index, mask in enumerate(masks)
        ]
    ).mean(axis=0)

    return MLPChannelStatistics(
        selectivity_scores=between_variance / (within_variance + 1e-8),
        mean_absolute_activations=np.abs(activations).mean(axis=0),
        activation_rates=(np.abs(activations) > activation_threshold).mean(axis=0),
    )


def _evaluate_ablation(
    backend,
    channels: set[tuple[int, int]],
) -> tuple[list[ForwardResult], int]:
    results = [
        run_with_mlp_channel_ablation(backend, case.context, channels)
        for case in CASES
    ]
    correct = sum(
        result.answer.categorization.category is case.expected
        for case, result in zip(CASES, results)
    )
    return results, correct


@pytest.mark.integration
def test_structural_mlp_channel_analysis(backend):
    """Erstellt eine kumulative Pruning-Kurve der niedrigsten MLP-Scores."""
    print(f"\n{'=' * 120}")
    print("STRUKTURELLE QWEN3-MLP-KANALANALYSE")
    print(f"Aktivierungsschwellwert: |x| > {ACTIVATION_THRESHOLD:g}")
    print(f"{'=' * 120}", flush=True)

    baseline: list[ForwardResult] = []
    for number, case in enumerate(CASES, start=1):
        result = run_with_mlp_channel_ablation(
            backend, case.context, collect_activations=True
        )
        baseline.append(result)
        actual = result.answer.categorization.category
        marker = "✓" if actual is case.expected else "✗"
        print(
            f"[{number:>2}/{len(CASES)}] {marker} {case.name:<22} "
            f"erwartet={case.expected.value:<8} erhalten={actual.value:<8}",
            flush=True,
        )

    activations = np.stack(
        [np.stack(result.mlp_activations) for result in baseline]
    )
    statistics = calculate_mlp_channel_statistics(
        activations, [case.expected for case in CASES]
    )
    baseline_correct = sum(
        result.answer.categorization.category is case.expected
        for case, result in zip(CASES, baseline)
    )
    baseline_classes = len(
        {result.answer.categorization.category for result in baseline}
    )
    total_channels = activations.shape[1] * activations.shape[2]
    overall_activation_rate = float(statistics.activation_rates.mean())
    # Stable Sort sorgt dafür, dass Gleichstände reproduzierbar nach dem flachen
    # (Layer-, Kanal-)Index aufgelöst werden.
    lowest_first = np.argsort(
        statistics.selectivity_scores.reshape(-1), kind="stable"
    )

    print("\n" + "-" * 120)
    print(
        f"Baseline: {baseline_correct}/{len(CASES)} korrekt | "
        f"ausgegebene Klassen: {baseline_classes} | Layer: {activations.shape[1]} | "
        f"Kanäle/Layer: {activations.shape[2]} | Kanäle gesamt: {total_channels:,}"
    )
    print(
        f"Aktivierungsquote über Fälle, Layer und Kanäle: "
        f"{overall_activation_rate:.2%}"
    )

    print("\n" + "=" * 120)
    print("PRUNING-KURVE: KUMULATIV DIE NIEDRIGSTEN SELEKTIVITÄTS-SCORES")
    print("=" * 120)
    print(
        f"{'Anteil':>8} | {'Kanäle':>10} | {'Accuracy':>10} | "
        f"{'Klassen':>7} | {'Änderungen':>11} | {'Ø |Δ Logit|':>13} | "
        f"{'max. entfernter Score':>21}"
    )
    print("-" * 120)

    baseline_logits = np.stack([result.category_logits for result in baseline])
    curve_rows = [
        {
            "pruning_ratio": 0.0,
            "removed_channels": 0,
            "correct": baseline_correct,
            "accuracy": baseline_correct / len(CASES),
            "distinct_classes": baseline_classes,
            "changed_answers": 0,
            "mean_absolute_logit_delta": 0.0,
            "maximum_removed_score": "",
        }
    ]
    print(
        f"{0:>7.1%} | {0:>10,} | {baseline_correct / len(CASES):>9.1%} | "
        f"{baseline_classes:>7} | {0:>11} | {0.0:>13.6f} | {'-':>21}"
    )

    for ratio in PRUNING_RATIOS:
        remove_count = min(total_channels, max(1, round(total_channels * ratio)))
        selected_flat_indices = lowest_first[:remove_count]
        layer_indices, channel_indices = np.unravel_index(
            selected_flat_indices, statistics.selectivity_scores.shape
        )
        channels = set(
            zip(layer_indices.astype(int).tolist(), channel_indices.astype(int).tolist())
        )

        ablated, correct = _evaluate_ablation(backend, channels)
        ablated_logits = np.stack([result.category_logits for result in ablated])
        mean_logit_delta = float(np.abs(ablated_logits - baseline_logits).mean())
        changed_answers = sum(
            current.answer.categorization.category
            is not original.answer.categorization.category
            for current, original in zip(ablated, baseline)
        )
        distinct_classes = len(
            {result.answer.categorization.category for result in ablated}
        )
        maximum_removed_score = float(
            statistics.selectivity_scores.reshape(-1)[selected_flat_indices[-1]]
        )
        curve_rows.append(
            {
                "pruning_ratio": ratio,
                "removed_channels": remove_count,
                "correct": correct,
                "accuracy": correct / len(CASES),
                "distinct_classes": distinct_classes,
                "changed_answers": changed_answers,
                "mean_absolute_logit_delta": mean_logit_delta,
                "maximum_removed_score": maximum_removed_score,
            }
        )
        print(
            f"{ratio:>7.1%} | {remove_count:>10,} | "
            f"{correct / len(CASES):>9.1%} | {distinct_classes:>7} | "
            f"{changed_answers:>11} | {mean_logit_delta:>13.6f} | "
            f"{maximum_removed_score:>21.6g}",
            flush=True,
        )

    log_directory = Path("categorization_eval_logs")
    log_directory.mkdir(parents=True, exist_ok=True)
    csv_path = log_directory / "mlp_pruning_curve.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=tuple(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)

    # "Bestes Modell" = größter getesteter Anteil mit mindestens der
    # Baseline-Accuracy und weiterhin allen Baseline-Klassen. Für den physischen
    # 4-Bit-Export wird die Kanalzahl jedes Layers auf Quantisierungsgruppen
    # ausgerichtet und dieser konkrete Plan vor dem Speichern erneut evaluiert.
    eligible_rows = sorted(
        (
            row
            for row in curve_rows
            if row["pruning_ratio"] > 0
            and row["correct"] >= baseline_correct
            and row["distinct_classes"] == baseline_classes
        ),
        key=lambda row: row["pruning_ratio"],
        reverse=True,
    )
    if not eligible_rows:
        raise RuntimeError(
            "Keine Pruning-Stufe hält Baseline-Accuracy und Klassenabdeckung"
        )

    down_proj = backend.model.model.layers[0].mlp.down_proj
    quantization_group_size = int(getattr(down_proj, "group_size", 1))
    export_selection = None
    for candidate in eligible_rows:
        aligned_channels = align_channels_for_quantized_export(
            statistics.selectivity_scores,
            float(candidate["pruning_ratio"]),
            group_size=quantization_group_size,
        )
        aligned_results, aligned_correct = _evaluate_ablation(
            backend, aligned_channels
        )
        aligned_classes = len(
            {result.answer.categorization.category for result in aligned_results}
        )
        actual_ratio = len(aligned_channels) / total_channels
        print(
            f"Exportprüfung {float(candidate['pruning_ratio']):.1%}: "
            f"gruppenausgerichtet={actual_ratio:.3%}, "
            f"Accuracy={aligned_correct}/{len(CASES)}, "
            f"Klassen={aligned_classes}",
            flush=True,
        )
        if aligned_correct >= baseline_correct and aligned_classes == baseline_classes:
            export_selection = (
                candidate,
                aligned_channels,
                aligned_correct,
                aligned_classes,
                actual_ratio,
            )
            break

    if export_selection is None:
        raise RuntimeError(
            "Keine gruppenausgerichtete Exportvariante hält die Baseline-Qualität"
        )

    candidate, channels, export_correct, export_classes, actual_ratio = export_selection
    export_path = export_structurally_pruned_qwen3(
        backend,
        channels,
        Path("models/qwen3-categorization-pruned"),
        pruning_ratio=float(candidate["pruning_ratio"]),
        accuracy=export_correct / len(CASES),
        distinct_classes=export_classes,
    )

    print("-" * 120)
    print(f"Kurvendaten: {csv_path.resolve()}")
    print(
        f"Exportiertes Modell: {export_path.resolve()} | "
        f"entfernte Kanäle={len(channels):,}/{total_channels:,} "
        f"({actual_ratio:.3%})"
    )
    print(
        "Ausführen: .venv/bin/python "
        "models/qwen3-categorization-pruned/run_pruned.py \"TEXT\""
    )
    print("=" * 120, flush=True)

    # Accuracy-Verluste sind Messergebnisse und keine Testfehler.
    assert activations.shape[0] == len(CASES)
    assert np.isfinite(activations).all()
    assert statistics.selectivity_scores.shape == activations.shape[1:]
    assert len(curve_rows) == len(PRUNING_RATIOS) + 1
