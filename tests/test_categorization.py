"""Vergleich von drei Decodern für die thematische LLM-Kategorisierung."""
import heapq
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pytest

from src.llm import generate
from src.mask import OutlinesRegexConstraint, OutlinesSchemaConstraint
from src.typed import schema_from_pydantic
from tests.test_typed import Category, Categorization


@dataclass(frozen=True)
class CategorizationCase:
    name: str
    context: str
    expected: Category


CASES = (
    # Sport
    CategorizationCase(
        "fussball",
        "Der FC gewinnt das Fußballspiel durch ein Tor in der Nachspielzeit.",
        Category.SPORT,
    ),
    CategorizationCase(
        "marathon",
        "Die Läuferin bereitet sich mit langen Trainingseinheiten auf den Marathon vor.",
        Category.SPORT,
    ),
    CategorizationCase(
        "tennis",
        "Im Tennisfinale entscheidet der Tie-Break des dritten Satzes über den Sieg.",
        Category.SPORT,
    ),
    CategorizationCase(
        "schwimmen",
        "Der Schwimmer verbessert über 100 Meter Freistil seinen persönlichen Rekord.",
        Category.SPORT,
    ),
    CategorizationCase(
        "radsport-mit-sensor",
        "Das Radsportteam wertet die Sensordaten aus, um die Leistung beim Rennen zu steigern.",
        Category.SPORT,
    ),
    # Kultur
    CategorizationCase(
        "theater",
        "Das Stadttheater zeigt eine neue Inszenierung von Goethes Faust.",
        Category.KULTUR,
    ),
    CategorizationCase(
        "museum",
        "Das Museum eröffnet eine Ausstellung über Malerei der Renaissance.",
        Category.KULTUR,
    ),
    CategorizationCase(
        "literatur",
        "Die Autorin liest auf dem Literaturfestival aus ihrem neuen Roman.",
        Category.KULTUR,
    ),
    CategorizationCase(
        "konzert",
        "Das Sinfonieorchester spielt bei seinem Konzert Werke von Beethoven.",
        Category.KULTUR,
    ),
    CategorizationCase(
        "digitales-archiv",
        "Ein digitales Archiv macht historische Fotografien und Handschriften öffentlich zugänglich.",
        Category.KULTUR,
    ),
    # Technik
    CategorizationCase(
        "prozessor",
        "Ein neuer Prozessor macht Notebooks schneller und verbraucht dabei weniger Strom.",
        Category.TECHNIK,
    ),
    CategorizationCase(
        "robotik",
        "Ein Roboter übernimmt in der Fabrik das präzise Verschweißen von Bauteilen.",
        Category.TECHNIK,
    ),
    CategorizationCase(
        "cybersicherheit",
        "Ein Sicherheitsupdate schließt eine kritische Lücke im Betriebssystem.",
        Category.TECHNIK,
    ),
    CategorizationCase(
        "ki",
        "Das neue KI-Modell übersetzt Texte automatisch in mehrere Sprachen.",
        Category.TECHNIK,
    ),
    CategorizationCase(
        "sport-app",
        "Die Entwickler veröffentlichen eine Smartphone-App, die Trainingsdaten synchronisiert.",
        Category.TECHNIK,
    ),
    # Test und Qualitätssicherung
    CategorizationCase(
        "unit-test",
        "Ein Unit-Test prüft, ob die Funktion für ungültige Eingaben den erwarteten Fehler auslöst.",
        Category.TEST,
    ),
    CategorizationCase(
        "integrationstest",
        "Der Integrationstest kontrolliert das Zusammenspiel von Datenbank und Webservice.",
        Category.TEST,
    ),
    CategorizationCase(
        "regressionstest",
        "Vor der Veröffentlichung führt das QA-Team alle automatisierten Regressionstests aus.",
        Category.TEST,
    ),
    CategorizationCase(
        "pruefung",
        "In der Abschlussprüfung beantworten die Schüler zwanzig Fragen unter Zeitdruck.",
        Category.TEST,
    ),
    CategorizationCase(
        "geraetetest",
        "Im Belastungstest wird geprüft, ob das Gerät auch bei extremer Hitze zuverlässig arbeitet.",
        Category.TEST,
    ),
)


CATEGORY_BY_CODE = {
    "0": Category.SPORT,
    "1": Category.KULTUR,
    "2": Category.TECHNIK,
    "3": Category.TEST,
}


CATEGORY_DESCRIPTIONS = """- sport: Wettkampf, Training und sportliche Leistungen
- kultur: Kunst, Literatur, Musik, Theater und kulturelles Erbe
- technik: Computer, Software, Geräte, Maschinen und technische Entwicklungen
- test: Prüfungen, Tests und Qualitätssicherung"""

JSON_PROMPT_TEMPLATE = """Ordne den folgenden Kontext genau einer Kategorie zu.

Bedeutung der Kategorien:
{descriptions}

Entscheide nach dem Hauptthema. Gib ausschließlich ein kurzes JSON-Dokument mit
dem Feld \"category\" zurück.

Kontext: {context}
"""

DIGIT_PROMPT_TEMPLATE = """Ordne den folgenden Kontext genau einer Kategorie zu.

Antworte ausschließlich mit der zugehörigen einzelnen Ziffer:
- 0 = sport: Wettkampf, Training und sportliche Leistungen
- 1 = kultur: Kunst, Literatur, Musik, Theater und kulturelles Erbe
- 2 = technik: Computer, Software, Geräte, Maschinen und technische Entwicklungen
- 3 = test: Prüfungen, Tests und Qualitätssicherung

Entscheide nach dem Hauptthema des Kontexts, auch wenn einzelne Wörter zu einer
anderen Kategorie passen. Keine Erklärung, kein JSON und kein zusätzlicher Text.

Kontext: {context}
Antwort:
"""

DIGIT_SYSTEM_PROMPT = (
    "Du bist ein Klassifikator. Befolge exakt das verlangte Ausgabeformat und "
    "antworte ohne Erklärung."
)


@dataclass(frozen=True)
class ClassifierAnswer:
    categorization: Categorization
    raw_answer: str
    confidence: float | None = None


def decode_category(code: str) -> Categorization:
    """Dekodiert die Modellziffer direkt in das typisierte Pydantic-Objekt."""
    normalized_code = code.strip()
    if normalized_code not in CATEGORY_BY_CODE:
        raise ValueError(f"Ungültiger Kategorie-Code: {code!r}")
    return Categorization(category=CATEGORY_BY_CODE[normalized_code])


def run_json_schema(backend, context: str) -> ClassifierAnswer:
    """Variante 1: vollständiges JSON mit Pydantic-/JSON-Schema-Constraint."""
    schema = schema_from_pydantic(Categorization)
    constraint = OutlinesSchemaConstraint(schema, backend.tokenizer, backend.vocab_size)
    result = generate(
        backend=backend,
        constraint=constraint,
        schema=schema,
        prompt=JSON_PROMPT_TEMPLATE.format(
            descriptions=CATEGORY_DESCRIPTIONS, context=context
        ),
        temperature=0,
        top_p=1,
        seed=0,
        max_new_tokens=32,
    )
    assert result.status == "abgeschlossen", result.error or result.text
    return ClassifierAnswer(
        categorization=Categorization.model_validate(json.loads(result.text)),
        raw_answer=result.text,
    )


def run_digit_decoder(backend, context: str) -> ClassifierAnswer:
    """Variante 2: eine Ziffer generieren und mit dem Custom Decoder umwandeln."""
    constraint = OutlinesRegexConstraint(
        r"[0-3]", backend.tokenizer, backend.vocab_size
    )
    result = generate(
        backend=backend,
        constraint=constraint,
        schema=None,
        prompt=DIGIT_PROMPT_TEMPLATE.format(context=context),
        temperature=0,
        top_p=1,
        seed=0,
        max_new_tokens=4,
        output_validator=decode_category,
        system_prompt=DIGIT_SYSTEM_PROMPT,
    )
    assert result.status == "abgeschlossen", result.error or result.text
    categorization = result.validation.get("value")
    assert isinstance(categorization, Categorization)
    return ClassifierAnswer(categorization, result.text)


def classify_hidden_with_reduced_lm_head(backend, hidden) -> ClassifierAnswer:
    """Projiziert einen Hidden State nur auf die vier Kategorie-Token."""
    mx = backend.mx
    model = backend.model
    if getattr(model.args, "tie_word_embeddings", True):
        raise RuntimeError("Der reduzierte LM-Head benötigt einen separaten lm_head")

    codes = tuple(CATEGORY_BY_CODE)
    token_ids = []
    for code in codes:
        encoded = backend.tokenizer.encode(code, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"Kategorie-Code {code!r} ist kein einzelnes Token")
        token_ids.append(encoded[0])

    head = model.lm_head
    selected_ids = mx.array(token_ids)
    if hasattr(head, "scales"):
        selected_logits = mx.quantized_matmul(
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
        selected_logits = hidden @ head.weight[selected_ids].T
    if hasattr(head, "bias"):
        selected_logits = selected_logits + head.bias[selected_ids]

    probabilities = mx.softmax(selected_logits, axis=-1)
    mx.eval(selected_logits, probabilities)
    selected_index = int(mx.argmax(selected_logits, axis=-1).item())
    confidence = float(probabilities[0, selected_index].item())
    code = codes[selected_index]
    return ClassifierAnswer(decode_category(code), code, confidence)


def prepare_digit_prompt(backend, context: str):
    """Bereitet den gemeinsamen Prompt vor und gibt seine Token als MLX-Array zurück."""
    prompt = DIGIT_PROMPT_TEMPLATE.format(context=context)
    backend.prepare_prompt(prompt, system_prompt=DIGIT_SYSTEM_PROMPT)
    input_ids = backend._prompt_ids
    if not input_ids:
        raise RuntimeError("Der vorbereitete Prompt enthält keine Tokens")
    return backend.mx.array(input_ids)[None]


def run_reduced_lm_head(backend, context: str) -> ClassifierAnswer:
    """Variante 3: berechnet am LM-Head ausschließlich die vier Ziffern-Logits."""
    inputs = prepare_digit_prompt(backend, context)
    hidden = backend.model.model(inputs, cache=backend.cache)[:, -1, :]
    return classify_hidden_with_reduced_lm_head(backend, hidden)


def run_reduced_lm_head_with_skips(
    backend, context: str, skip_layers: set[int]
) -> ClassifierAnswer:
    """Variante 4: überspringt Transformer-Layer vor dem reduzierten LM-Head."""
    from mlx_lm.models.base import create_attention_mask

    inputs = prepare_digit_prompt(backend, context)
    transformer = backend.model.model
    layer_count = len(transformer.layers)
    invalid_layers = sorted(index for index in skip_layers if not 0 <= index < layer_count)
    if invalid_layers:
        raise ValueError(
            f"Ungültige Layer {invalid_layers}; das Modell hat {layer_count} Layer"
        )

    hidden = transformer.embed_tokens(inputs)
    mask = create_attention_mask(hidden, backend.cache[0])
    for index, (layer, cache) in enumerate(zip(transformer.layers, backend.cache)):
        if index not in skip_layers:
            hidden = layer(hidden, mask, cache)
    hidden = transformer.norm(hidden)[:, -1, :]

    return classify_hidden_with_reduced_lm_head(backend, hidden)


def run_eval(
    backend,
    name: str,
    classifier: Callable[[object, str], ClassifierAnswer],
) -> None:
    """Führt dieselben 20 Fälle aus und zeigt Antwort und Laufzeit sofort an."""
    results: list[tuple[CategorizationCase, ClassifierAnswer, float]] = []

    print(f"\n{'=' * 88}\n{name}\n{'=' * 88}", flush=True)
    total_started = time.perf_counter()
    for number, case in enumerate(CASES, start=1):
        started = time.perf_counter()
        answer = classifier(backend, case.context)
        duration = time.perf_counter() - started
        results.append((case, answer, duration))

        actual = answer.categorization.category
        marker = "✓" if actual is case.expected else "✗"
        confidence = (
            f", konfidenz={answer.confidence:.1%}"
            if answer.confidence is not None
            else ""
        )
        print(
            f"[{number:>2}/{len(CASES)}] {marker} {case.name:<22} "
            f"zeit={duration:7.3f}s, antwort={answer.raw_answer!r}, "
            f"kategorie={actual.value}{confidence}",
            flush=True,
        )

    total_duration = time.perf_counter() - total_started
    correct = sum(
        answer.categorization.category is case.expected
        for case, answer, _ in results
    )
    failed = len(results) - correct

    print("-" * 88)
    print(f"Korrekt:              {correct:>2}/{len(results)}")
    print(f"Fehlklassifikationen: {failed:>2}/{len(results)}")
    print(f"Genauigkeit:          {correct / len(results):>6.1%}")
    print(f"Gesamtzeit:           {total_duration:>7.3f}s")
    print(f"Durchschnitt:         {total_duration / len(results):>7.3f}s pro Antwort")
    print("=" * 88, flush=True)

    # Falsche Kategorien sind ein Eval-Ergebnis und kein Testfehler.
    assert len(results) == len(CASES) == 20


@pytest.mark.integration
def test_1_json_schema_eval(backend):
    """Eval einzeln: normales strukturiertes JSON."""
    run_eval(backend, "VARIANTE 1: JSON-SCHEMA", run_json_schema)


@pytest.mark.integration
def test_2_digit_decoder_eval(backend):
    """Eval einzeln: eingeschränkte Zifferngenerierung mit Custom Decoder."""
    run_eval(backend, "VARIANTE 2: ZIFFER + CUSTOM DECODER", run_digit_decoder)


@pytest.mark.integration
def test_3_reduced_lm_head_eval(backend):
    """Eval einzeln: direkte Klassifikation über vier LM-Head-Zeilen."""
    run_eval(backend, "VARIANTE 3: REDUZIERTER LM-HEAD", run_reduced_lm_head)


@pytest.mark.integration
def test_4_reduced_lm_head_with_skipped_layers_eval(backend):
    """Eval einzeln: reduzierter LM-Head mit konfigurierbaren Layer-Skips."""
    skip_layers = {7,17,34,  26, 27, 28, 29} # 15, 16, 4, 9, 20, drei hintereinander von den besten einzelnen, landen auf random
    ordered_layers = ", ".join(str(index) for index in sorted(skip_layers))
    run_eval(
        backend,
        f"VARIANTE 4: REDUZIERTER LM-HEAD, SKIP LAYERS {{{ordered_layers}}}",
        lambda current_backend, context: run_reduced_lm_head_with_skips(
            current_backend, context, skip_layers
        ),
    )

@pytest.mark.integration
def test_4_reduced_lm_head_with_skipped_layers_14_eval(backend):
    """Eval einzeln: reduzierter LM-Head mit konfigurierbaren Layer-Skips."""
    skip_layers = {2,7,17,25,26,27,28,29,30,31,32,33,34,35}
    ordered_layers = ", ".join(str(index) for index in sorted(skip_layers))
    run_eval(
        backend,
        f"VARIANTE 4: REDUZIERTER LM-HEAD, SKIP LAYERS {{{ordered_layers}}}",
        lambda current_backend, context: run_reduced_lm_head_with_skips(
            current_backend, context, skip_layers
        ),
    )

@pytest.mark.integration
def test_4_2_reduced_lm_head_with_skipped_layers_eval(backend):
    """Eval einzeln: reduzierter LM-Head mit konfigurierbaren Layer-Skips."""
    skip_layers = {7, 17, 34} # 7, 17, 26 | 7, 17, 34 sind die wo mit 3 Layern noch 100% haben
    ordered_layers = ", ".join(str(index) for index in sorted(skip_layers))
    run_eval(
        backend,
        f"VARIANTE 4: REDUZIERTER LM-HEAD, SKIP LAYERS {{{ordered_layers}}}",
        lambda current_backend, context: run_reduced_lm_head_with_skips(
            current_backend, context, skip_layers
        ),
    )

@pytest.mark.integration
def test_4_1_reduced_lm_head_with_skipped_layers_7_17_eval(backend):
    """Eval einzeln: reduzierter LM-Head mit konfigurierbaren Layer-Skips."""
    skip_layers = {7, 17} # Die wo mit 2 100% haben
    ordered_layers = ", ".join(str(index) for index in sorted(skip_layers))
    run_eval(
        backend,
        f"VARIANTE 4: REDUZIERTER LM-HEAD, SKIP LAYERS {{{ordered_layers}}}",
        lambda current_backend, context: run_reduced_lm_head_with_skips(
            current_backend, context, skip_layers
        ),
    )

@pytest.mark.integration
def test_4__reduced_lm_head_with_skipped_layers_eval(backend):
        """Eval einzeln: reduzierter LM-Head mit konfigurierbaren Layer-Skips."""
        skip_layers = {7, 17, 34, 26, 27, 28, 29}  # random hint auf 95%
        ordered_layers = ", ".join(str(index) for index in sorted(skip_layers))
        run_eval(
            backend,
            f"VARIANTE 4: REDUZIERTER LM-HEAD, SKIP LAYERS {{{ordered_layers}}}",
            lambda current_backend, context: run_reduced_lm_head_with_skips(
                current_backend, context, skip_layers
            ),
        )

@pytest.mark.integration
def test_5_single_layer_ablation_eval(backend):
    """Entfernt jeden Layer einzeln und misst den Genauigkeitsverlust."""
    layer_count = len(backend.model.model.layers)

    print(f"\n{'=' * 88}")
    print(f"SINGLE-LAYER-ABLATION ÜBER {layer_count} LAYER UND {len(CASES)} FÄLLE")
    print("Jeder Layer wird separat entfernt; die Entfernung ist nicht kumulativ.")
    print(f"{'=' * 88}", flush=True)

    baseline_started = time.perf_counter()
    baseline_answers = [
        run_reduced_lm_head(backend, case.context).categorization.category
        for case in CASES
    ]
    baseline_duration = time.perf_counter() - baseline_started
    baseline_correct = sum(
        actual is case.expected for case, actual in zip(CASES, baseline_answers)
    )
    baseline_accuracy = baseline_correct / len(CASES)
    print(
        f"Baseline ohne Skip: {baseline_correct}/{len(CASES)} korrekt "
        f"({baseline_accuracy:.1%}), Zeit={baseline_duration:.3f}s\n",
        flush=True,
    )

    layer_results: list[tuple[int, int, float, float, int]] = []
    for layer_index in range(layer_count):
        started = time.perf_counter()
        answers = []
        for case_number, case in enumerate(CASES, start=1):
            answer = run_reduced_lm_head_with_skips(
                backend, case.context, {layer_index}
            )
            answers.append(answer.categorization.category)
            print(
                f"  Layer {layer_index:>2} | Fall {case_number:>2}/{len(CASES)} "
                f"| Antwort={answer.categorization.category.value}",
                flush=True,
            )

        duration = time.perf_counter() - started
        correct = sum(
            actual is case.expected for case, actual in zip(CASES, answers)
        )
        accuracy = correct / len(CASES)
        accuracy_cost = baseline_accuracy - accuracy
        changed_answers = sum(
            actual is not baseline
            for actual, baseline in zip(answers, baseline_answers)
        )
        layer_results.append(
            (layer_index, correct, accuracy_cost, duration, changed_answers)
        )
        print(
            f"=> Layer {layer_index:>2} entfernt: {correct}/{len(CASES)} korrekt, "
            f"Genauigkeit={accuracy:.1%}, Kosten={accuracy_cost:+.1%}, "
            f"geänderte Antworten={changed_answers}, Zeit={duration:.3f}s\n",
            flush=True,
        )

    print("\n" + "=" * 88)
    print("ERGEBNIS DER SINGLE-LAYER-ABLATION")
    print("=" * 88)
    print(
        f"{'Layer':>5} | {'Korrekt':>7} | {'Genauigkeit':>12} | "
        f"{'Kosten':>9} | {'Geändert':>8} | {'Zeit':>9}"
    )
    print("-" * 88)
    for layer_index, correct, accuracy_cost, duration, changed_answers in layer_results:
        accuracy = correct / len(CASES)
        print(
            f"{layer_index:>5} | {correct:>2}/{len(CASES):<4} | "
            f"{accuracy:>11.1%} | {accuracy_cost:>+8.1%} | "
            f"{changed_answers:>8} | {duration:>8.3f}s"
        )
    print("-" * 88)
    print(
        f"Baseline: {baseline_correct}/{len(CASES)} korrekt "
        f"({baseline_accuracy:.1%})"
    )
    print(
        "Kosten = Baseline-Genauigkeit minus Genauigkeit ohne den jeweiligen Layer."
    )
    print("=" * 88, flush=True)

    # Schlechtere Ergebnisse sind Teil der Eval und sollen den Test nicht abbrechen.
    assert len(layer_results) == layer_count


@pytest.mark.integration
def test_6_layer_combination_eval(backend):
    """Vergleicht ausgewählte Dreierkombinationen mit Baseline und {7, 17}."""
    layer_combinations = (
        {7, 17, 4},   # Layer 4 hatte einzeln 95 % Genauigkeit.
        {7, 17, 9},   # Layer 9 hatte einzeln 95 % Genauigkeit.
        {7, 17, 15},  # Layer 15 änderte einzeln nur eine Antwort.
        {7, 17, 16},  # Layer 16 hatte einzeln 95 % Genauigkeit.
        {7, 17, 20},  # Layer 20 änderte einzeln nur eine Antwort.
        {7, 17, 26},  # Layer 26 änderte einzeln keine Antwort.
        {7, 17, 34},  # Layer 34 änderte einzeln keine Antwort.
    )
    pair_layers = {7, 17}

    def evaluate(skip_layers: set[int], label: str) -> tuple[list[Category], float]:
        answers: list[Category] = []
        started = time.perf_counter()
        for case_number, case in enumerate(CASES, start=1):
            if skip_layers:
                answer = run_reduced_lm_head_with_skips(
                    backend, case.context, skip_layers
                )
            else:
                answer = run_reduced_lm_head(backend, case.context)
            answers.append(answer.categorization.category)
            marker = "✓" if answer.categorization.category is case.expected else "✗"
            print(
                f"  {label:<18} | {case_number:>2}/{len(CASES)} | {marker} "
                f"{case.name:<22} | Antwort={answer.categorization.category.value}",
                flush=True,
            )
        return answers, time.perf_counter() - started

    print(f"\n{'=' * 100}")
    print("EVAL DER LAYER-KOMBINATIONEN")
    print(f"{'=' * 100}", flush=True)

    baseline_answers, baseline_duration = evaluate(set(), "Baseline")
    baseline_correct = sum(
        actual is case.expected for case, actual in zip(CASES, baseline_answers)
    )
    baseline_accuracy = baseline_correct / len(CASES)

    pair_answers, pair_duration = evaluate(pair_layers, "Skip {7,17}")
    pair_correct = sum(
        actual is case.expected for case, actual in zip(CASES, pair_answers)
    )
    pair_accuracy = pair_correct / len(CASES)

    combination_results: list[
        tuple[set[int], int, float, float, int, int]
    ] = []
    for skip_layers in layer_combinations:
        ordered = ",".join(str(index) for index in sorted(skip_layers))
        answers, duration = evaluate(skip_layers, f"Skip {{{ordered}}}")
        correct = sum(
            actual is case.expected for case, actual in zip(CASES, answers)
        )
        accuracy = correct / len(CASES)
        changed_from_baseline = sum(
            actual is not baseline
            for actual, baseline in zip(answers, baseline_answers)
        )
        changed_from_pair = sum(
            actual is not pair
            for actual, pair in zip(answers, pair_answers)
        )
        combination_results.append(
            (
                skip_layers,
                correct,
                duration,
                baseline_accuracy - accuracy,
                changed_from_baseline,
                changed_from_pair,
            )
        )
        print(
            f"=> Skip {{{ordered}}}: {correct}/{len(CASES)} korrekt "
            f"({accuracy:.1%}), Kosten zur Baseline="
            f"{baseline_accuracy - accuracy:+.1%}, Änderung zu {{7,17}}="
            f"{changed_from_pair}\n",
            flush=True,
        )

    print("\n" + "=" * 100)
    print("ERGEBNIS DER LAYER-KOMBINATIONEN")
    print("=" * 100)
    print(
        f"Baseline:    {baseline_correct}/{len(CASES)} korrekt "
        f"({baseline_accuracy:.1%}), {baseline_duration:.3f}s"
    )
    print(
        f"Skip {{7,17}}: {pair_correct}/{len(CASES)} korrekt "
        f"({pair_accuracy:.1%}), {pair_duration:.3f}s"
    )
    print("-" * 100)
    print(
        f"{'Skip-Layer':<15} | {'Korrekt':>7} | {'Genauigkeit':>12} | "
        f"{'Kosten':>9} | {'Δ Basis':>7} | {'Δ Paar':>6} | {'Zeit':>9}"
    )
    print("-" * 100)
    for (
        skip_layers,
        correct,
        duration,
        accuracy_cost,
        changed_from_baseline,
        changed_from_pair,
    ) in combination_results:
        ordered = ",".join(str(index) for index in sorted(skip_layers))
        label = f"{{{ordered}}}"
        print(
            f"{label:<15} | {correct:>2}/{len(CASES):<4} | "
            f"{correct / len(CASES):>11.1%} | {accuracy_cost:>+8.1%} | "
            f"{changed_from_baseline:>7} | {changed_from_pair:>6} | "
            f"{duration:>8.3f}s"
        )
    print("=" * 100, flush=True)

    # Unterschiede zwischen Kombinationen sind Messergebnisse, keine Testfehler.
    assert len(combination_results) == len(layer_combinations) == 7


@pytest.mark.integration
def test_7_add_one_layer_to_7_17_34_ablation_eval(backend):
    """Ergänzt {7, 17, 34} jeweils um jeden anderen Layer und misst die Folgen."""
    base_skip_layers = {7, 17, 34}
    layer_count = len(backend.model.model.layers)
    candidate_layers = [
        index for index in range(layer_count) if index not in base_skip_layers
    ]

    def evaluate(
        skip_layers: set[int], label: str
    ) -> tuple[list[Category], float]:
        answers: list[Category] = []
        started = time.perf_counter()
        for case_number, case in enumerate(CASES, start=1):
            answer = run_reduced_lm_head_with_skips(
                backend, case.context, skip_layers
            )
            actual = answer.categorization.category
            answers.append(actual)
            marker = "✓" if actual is case.expected else "✗"
            print(
                f"  {label:<20} | {case_number:>2}/{len(CASES)} | {marker} "
                f"{case.name:<22} | Antwort={actual.value}",
                flush=True,
            )
        return answers, time.perf_counter() - started

    print(f"\n{'=' * 104}")
    print("ADD-ONE-LAYER-ABLATION AUF BASIS VON SKIP {7,17,34}")
    print(
        "Zu {7,17,34} wird jeweils genau ein weiterer Layer hinzugefügt; "
        "die Kombinationen sind nicht kumulativ."
    )
    print(f"{'=' * 104}", flush=True)

    base_answers, base_duration = evaluate(base_skip_layers, "Skip {7,17,34}")
    base_correct = sum(
        actual is case.expected for case, actual in zip(CASES, base_answers)
    )
    base_accuracy = base_correct / len(CASES)
    print(
        f"Basis {{7,17,34}}: {base_correct}/{len(CASES)} korrekt "
        f"({base_accuracy:.1%}), Zeit={base_duration:.3f}s\n",
        flush=True,
    )

    results: list[tuple[int, int, float, int, float, int]] = []
    for added_layer in candidate_layers:
        skip_layers = {*base_skip_layers, added_layer}
        ordered = f"7,17,34,{added_layer}"
        answers, duration = evaluate(skip_layers, f"Skip {{{ordered}}}")
        correct = sum(
            actual is case.expected for case, actual in zip(CASES, answers)
        )
        accuracy = correct / len(CASES)
        accuracy_cost = base_accuracy - accuracy
        changed_answers = sum(
            actual is not base
            for actual, base in zip(answers, base_answers)
        )
        distinct_categories = len(set(answers))
        results.append(
            (
                added_layer,
                correct,
                accuracy_cost,
                changed_answers,
                duration,
                distinct_categories,
            )
        )
        print(
            f"=> + Layer {added_layer:>2}: {correct}/{len(CASES)} korrekt "
            f"({accuracy:.1%}), Kosten zu {{7,17,34}}={accuracy_cost:+.1%}, "
            f"geändert={changed_answers}, Kategorien={distinct_categories}, "
            f"Zeit={duration:.3f}s\n",
            flush=True,
        )

    print("\n" + "=" * 104)
    print("ERGEBNIS: {7,17,34} PLUS JEWEILS EIN WEITERER LAYER")
    print("=" * 104)
    print(
        f"Basis {{7,17,34}}: {base_correct}/{len(CASES)} korrekt "
        f"({base_accuracy:.1%}), {base_duration:.3f}s"
    )
    print("-" * 104)
    print(
        f"{'Zusatz':>6} | {'Skip-Kombination':<14} | {'Korrekt':>7} | "
        f"{'Genauigkeit':>12} | {'Kosten':>9} | {'Geändert':>8} | "
        f"{'Klassen':>7} | {'Zeit':>9}"
    )
    print("-" * 104)
    for (
        added_layer,
        correct,
        accuracy_cost,
        changed_answers,
        duration,
        distinct_categories,
    ) in results:
        ordered = f"7,17,34,{added_layer}"
        label = f"{{{ordered}}}"
        print(
            f"{added_layer:>6} | {label:<14} | {correct:>2}/{len(CASES):<4} | "
            f"{correct / len(CASES):>11.1%} | {accuracy_cost:>+8.1%} | "
            f"{changed_answers:>8} | {distinct_categories:>7} | "
            f"{duration:>8.3f}s"
        )
    print("-" * 104)
    print(
        "Kosten = Genauigkeit von {7,17,34} minus Genauigkeit mit dem Zusatz-Layer."
    )
    print(
        "Klassen = Anzahl unterschiedlicher ausgegebener Kategorien; ein starker "
        "Abfall kann auf kollabierte Ausgaben hinweisen."
    )
    print("=" * 104, flush=True)

    # Qualitätsverluste sind Messergebnisse und sollen den Test nicht abbrechen.
    assert len(results) == layer_count - len(base_skip_layers)


@pytest.mark.integration
def test_8_reverse_add_one_layer_to_7_17_26_27_28_29_34_eval(backend):
    """Erweitert die Skip-Basis rückwärts um jeweils genau einen weiteren Layer."""
    base_skip_layers = {7, 17, 34, 26, 27, 28, 29}
    base_label = "7,17,34,26,27,28,29"
    layer_count = len(backend.model.model.layers)
    candidate_layers = [
        index
        for index in reversed(range(layer_count))
        if index not in base_skip_layers
    ]

    def evaluate(
        skip_layers: set[int], label: str
    ) -> tuple[list[Category], float]:
        answers: list[Category] = []
        started = time.perf_counter()
        for case_number, case in enumerate(CASES, start=1):
            answer = run_reduced_lm_head_with_skips(
                backend, case.context, skip_layers
            )
            actual = answer.categorization.category
            answers.append(actual)
            marker = "✓" if actual is case.expected else "✗"
            print(
                f"  {label:<32} | {case_number:>2}/{len(CASES)} | {marker} "
                f"{case.name:<22} | Antwort={actual.value}",
                flush=True,
            )
        return answers, time.perf_counter() - started

    print(f"\n{'=' * 112}")
    print(f"RÜCKWÄRTS-ABLATION AUF BASIS VON SKIP {{{base_label}}}")
    print(
        "Es wird in absteigender Reihenfolge jeweils genau ein weiterer Layer "
        "entfernt; die Kombinationen sind nicht kumulativ."
    )
    print(f"Reihenfolge: {candidate_layers}")
    print(f"{'=' * 112}", flush=True)

    base_answers, base_duration = evaluate(
        base_skip_layers, f"Skip {{{base_label}}}"
    )
    base_correct = sum(
        actual is case.expected for case, actual in zip(CASES, base_answers)
    )
    base_accuracy = base_correct / len(CASES)
    print(
        f"Basis {{{base_label}}}: {base_correct}/{len(CASES)} korrekt "
        f"({base_accuracy:.1%}), Zeit={base_duration:.3f}s\n",
        flush=True,
    )

    results: list[tuple[int, int, float, int, int, float]] = []
    for added_layer in candidate_layers:
        skip_layers = {*base_skip_layers, added_layer}
        combination_label = f"{base_label},{added_layer}"
        answers, duration = evaluate(
            skip_layers, f"Skip {{{combination_label}}}"
        )
        correct = sum(
            actual is case.expected for case, actual in zip(CASES, answers)
        )
        accuracy = correct / len(CASES)
        accuracy_cost = base_accuracy - accuracy
        changed_answers = sum(
            actual is not base
            for actual, base in zip(answers, base_answers)
        )
        distinct_categories = len(set(answers))
        results.append(
            (
                added_layer,
                correct,
                accuracy_cost,
                changed_answers,
                distinct_categories,
                duration,
            )
        )
        print(
            f"=> + Layer {added_layer:>2}: {correct}/{len(CASES)} korrekt "
            f"({accuracy:.1%}), Kosten zur Basis={accuracy_cost:+.1%}, "
            f"geändert={changed_answers}, Kategorien={distinct_categories}, "
            f"Zeit={duration:.3f}s\n",
            flush=True,
        )

    print("\n" + "=" * 112)
    print("ERGEBNIS DER RÜCKWÄRTS-ABLATION")
    print("=" * 112)
    print(
        f"Basis {{{base_label}}}: {base_correct}/{len(CASES)} korrekt "
        f"({base_accuracy:.1%}), {base_duration:.3f}s"
    )
    print("-" * 112)
    print(
        f"{'Zusatz':>6} | {'Korrekt':>7} | {'Genauigkeit':>12} | "
        f"{'Kosten':>9} | {'Geändert':>8} | {'Klassen':>7} | {'Zeit':>9}"
    )
    print("-" * 112)
    for (
        added_layer,
        correct,
        accuracy_cost,
        changed_answers,
        distinct_categories,
        duration,
    ) in results:
        print(
            f"{added_layer:>6} | {correct:>2}/{len(CASES):<4} | "
            f"{correct / len(CASES):>11.1%} | {accuracy_cost:>+8.1%} | "
            f"{changed_answers:>8} | {distinct_categories:>7} | "
            f"{duration:>8.3f}s"
        )
    print("-" * 112)
    print(
        "Kosten = Genauigkeit der Skip-Basis minus Genauigkeit mit dem "
        "zusätzlich entfernten Layer."
    )
    print("=" * 112, flush=True)

    # Qualitätsverluste sind Messergebnisse und sollen den Test nicht abbrechen.
    assert len(results) == layer_count - len(base_skip_layers)
    assert [result[0] for result in results] == candidate_layers


@pytest.mark.integration
def test_9_prioritized_recursive_layer_search(backend):
    """Sucht best-first nach möglichst großen Skip-Mengen mit mindestens 80 %."""
    start_layers = frozenset({7, 17, 26, 27, 28, 29, 34})
    minimum_correct = 16  # 16/20 = 80 %
    layer_count = len(backend.model.model.layers)
    all_layers = tuple(range(layer_count - 1, -1, -1))

    log_directory = Path("categorization_eval_logs")
    log_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    log_path = log_directory / f"recursive_layer_search_{timestamp}.log"

    # Alle getesteten Mengen landen hier. Erfolgreiche Mengen werden zusätzlich
    # nach Genauigkeit priorisiert und später in der Zusammenfassung gruppiert.
    evaluated: dict[frozenset[int], tuple[int, float, tuple[Category, ...]]] = {}
    accepted: dict[frozenset[int], tuple[int, float, tuple[Category, ...]]] = {}
    queue: list[tuple[int, int, int, frozenset[int]]] = []
    queue_counter = 0
    interrupted = False
    pending_error: BaseException | None = None

    def layer_label(layers: frozenset[int]) -> str:
        return "{" + ",".join(str(index) for index in sorted(layers)) + "}"

    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        def emit(message: str = "") -> None:
            print(message, flush=True)
            log_file.write(message + "\n")
            log_file.flush()

        def evaluate(layers: frozenset[int]) -> tuple[int, float, tuple[Category, ...]]:
            label = layer_label(layers)
            answers: list[Category] = []
            started = time.perf_counter()
            emit(f"\nTEST {label} ({len(layers)} Skips)")
            for case_number, case in enumerate(CASES, start=1):
                answer = run_reduced_lm_head_with_skips(
                    backend, case.context, set(layers)
                )
                actual = answer.categorization.category
                answers.append(actual)
                marker = "OK" if actual is case.expected else "FALSCH"
                emit(
                    f"  [{case_number:>2}/{len(CASES)}] {marker:<6} "
                    f"{case.name:<22} erwartet={case.expected.value:<8} "
                    f"erhalten={actual.value}"
                )

            duration = time.perf_counter() - started
            correct = sum(
                actual is case.expected for case, actual in zip(CASES, answers)
            )
            result = (correct, duration, tuple(answers))
            evaluated[layers] = result
            accepted_text = "WEITER" if correct >= minimum_correct else "STOP"
            emit(
                f"ERGEBNIS {label}: {correct}/{len(CASES)} "
                f"({correct / len(CASES):.0%}), {duration:.3f}s, {accepted_text}"
            )
            return result

        def enqueue(layers: frozenset[int], correct: int) -> None:
            nonlocal queue_counter
            # Höhere Genauigkeit zuerst; bei Gleichstand tiefere Skip-Mengen zuerst.
            heapq.heappush(
                queue,
                (-correct, -len(layers), queue_counter, layers),
            )
            queue_counter += 1

        def write_summary() -> None:
            emit("\n" + "=" * 110)
            emit("ZUSAMMENFASSUNG DER PRIORISIERTEN LAYER-SUCHE")
            emit("=" * 110)
            emit(f"Logdatei: {log_path.resolve()}")
            emit(f"Getestete Kombinationen: {len(evaluated)}")
            emit(f"Kombinationen ab 80 %:   {len(accepted)}")
            emit(f"Noch in der Warteschlange:{len(queue):>5}")
            emit(f"Manuell abgebrochen:      {'ja' if interrupted else 'nein'}")

            for correct in range(len(CASES), minimum_correct - 1, -1):
                matching = [
                    (layers, result)
                    for layers, result in accepted.items()
                    if result[0] == correct
                ]
                matching.sort(key=lambda item: (-len(item[0]), tuple(sorted(item[0]))))
                emit("\n" + "-" * 110)
                emit(
                    f"{correct / len(CASES):.0%} GENAUIGKEIT "
                    f"({len(matching)} Kombinationen)"
                )
                emit("-" * 110)
                for layers, (_, duration, _) in matching:
                    emit(
                        f"Skips={len(layers):>2} | {layer_label(layers):<82} "
                        f"| {duration:>8.3f}s"
                    )
            emit("=" * 110)

        emit("PRIORISIERTE REKURSIVE LAYER-SUCHE")
        emit(f"Startmenge: {layer_label(start_layers)}")
        emit(f"Layeranzahl: {layer_count}")
        emit(f"Schwellwert: {minimum_correct}/{len(CASES)} (80 %)")
        emit(
            "Priorität: Genauigkeit absteigend, danach Anzahl der Skips absteigend."
        )
        emit("Mit Ctrl-C kann die Suche beendet und die Zusammenfassung geschrieben werden.")
        emit(f"Logdatei: {log_path.resolve()}")

        try:
            start_result = evaluate(start_layers)
            if start_result[0] >= minimum_correct:
                accepted[start_layers] = start_result
                enqueue(start_layers, start_result[0])
            else:
                emit("Die Startmenge liegt unter 80 %; es gibt nichts zu erweitern.")

            while queue:
                neg_correct, neg_depth, _, current_layers = heapq.heappop(queue)
                emit(
                    f"\nEXPANDIERE {layer_label(current_layers)} | "
                    f"Genauigkeit={-neg_correct}/{len(CASES)} | "
                    f"Skips={-neg_depth} | Queue={len(queue)}"
                )

                # Die Layer werden rückwärts geprüft. Neue Treffer landen sofort
                # in der globalen Priority Queue und gelten über alle Tiefen hinweg.
                for layer_index in all_layers:
                    if layer_index in current_layers:
                        continue
                    candidate = current_layers | {layer_index}
                    if candidate in evaluated:
                        continue

                    result = evaluate(candidate)
                    if result[0] >= minimum_correct:
                        accepted[candidate] = result
                        enqueue(candidate, result[0])
                        emit(
                            f"QUEUE + {layer_label(candidate)} | "
                            f"Priorität={result[0]}/{len(CASES)} | "
                            f"Skips={len(candidate)}"
                        )
        except KeyboardInterrupt:
            interrupted = True
            emit("\nABBRUCH empfangen; schreibe den bisherigen Stand.")
        except BaseException as exc:
            pending_error = exc
            emit(f"\nFEHLER: {type(exc).__name__}: {exc}")

        write_summary()

    print(f"\nVollständiges Suchprotokoll: {log_path.resolve()}", flush=True)
    if pending_error is not None:
        raise pending_error

    # Ein manueller Abbruch und schwächere Kombinationen sind gültige Eval-Ergebnisse.
    assert start_layers in evaluated
