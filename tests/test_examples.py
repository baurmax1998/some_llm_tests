"""Echte lokale End-to-End-Läufe mit aus Pydantic abgeleiteten Schemata."""
import json
import math
import os
import re
from enum import Enum

import pytest
from pydantic import BaseModel, ConfigDict

from json_schema_playground.generation import generate
from json_schema_playground.model_backend import DEFAULT_MODEL, MLXModelBackend
from json_schema_playground.schema_engine import (
    OutlinesSchemaConstraint,
    schema_from_pydantic,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Sentiment(str, Enum):
    POSITIV = "positiv"
    NEUTRAL = "neutral"
    NEGATIV = "negativ"


class ReviewDetails(StrictModel):
    score: int


class Review(StrictModel):
    sentiment: Sentiment
    verified: bool
    tags: list[str]
    details: ReviewDetails


REVIEW_PROMPT = """Extrahiere aus dem folgenden Kontext eine strukturierte Bewertung.

Kontext: Eine verifizierte Käuferin schreibt: „Das Produkt gefällt mir sehr und
funktioniert ausgezeichnet.“ Sie vergibt 5 Punkte. Verwende als Tags ausschließlich
„produkt“ und „positiv“.

Gib ein möglichst kurzes JSON-Dokument zurück. sentiment soll „positiv“, verified
soll true und details.score soll 5 sein.
"""


class Category(str, Enum):
    TECHNIK = "technik"
    TEST = "test"
    SPORT = "sport"
    KULTUR = "kultur"


class Categorization(StrictModel):
    category: Category


CATEGORY_PROMPT = """Ordne den folgenden Kontext genau einer erlaubten Kategorie zu.

Kontext: Apple hat einen neuen Prozessor für seine Mac-Computer vorgestellt, mit dem lassen sich sport Werte besser berechnen.
"""


class Approval(StrictModel):
    approved: bool


APPROVAL_PROMPT = """Triff anhand des Kontexts eine Ja/Nein-Entscheidung.

Kontext: Der Antrag enthält alle Pflichtangaben, die Identität wurde geprüft und
sämtliche Freigabebedingungen sind erfüllt.

Setze approved auf true und gib ausschließlich das kurze JSON-Dokument zurück.
"""


@pytest.fixture(scope="module")
def backend():
    """Das große Modell nur einmal laden; der KV-Cache wird pro Lauf erneuert."""
    return MLXModelBackend(os.getenv("TEST_MODEL_ID", DEFAULT_MODEL))


def json_leaf_value_spans(text):
    """Zeichenbereiche skalarer JSON-Werte, ohne Schlüssel und Strukturzeichen."""
    spans = []
    index = 0
    while index < len(text):
        if text[index] == '"':
            end = index + 1
            while end < len(text):
                if text[end] == "\\":
                    end += 2
                    continue
                if text[end] == '"':
                    break
                end += 1
            following = end + 1
            while following < len(text) and text[following].isspace():
                following += 1
            # Ein String vor einem Doppelpunkt ist ein Feldname, sonst ein Wert.
            if following >= len(text) or text[following] != ":":
                if end > index + 1:  # leere Strings besitzen kein Inhaltstoken
                    spans.append((index + 1, end))
            index = end + 1
            continue
        if text.startswith("true", index):
            spans.append((index, index + 4))
            index += 4
            continue
        if text.startswith("false", index):
            spans.append((index, index + 5))
            index += 5
            continue
        if text[index] == "-" or text[index].isdigit():
            end = index + 1
            while end < len(text) and text[end] in "0123456789.eE+-":
                end += 1
            spans.append((index, end))
            index = end
            continue
        index += 1
    return spans


def leaf_value_options(schema, value):
    """Legale endliche Werte in derselben Reihenfolge wie die JSON-Blätter."""
    schema_type = schema["type"]
    if schema_type == "object":
        options = []
        for name, child in schema["properties"].items():
            if name in value:
                options.extend(leaf_value_options(child, value[name]))
        return options
    if schema_type == "array":
        options = []
        for item in value:
            options.extend(leaf_value_options(schema["items"], item))
        return options
    if schema_type == "string" and "enum" in schema:
        # Inhalt zwischen den JSON-Anführungszeichen, inklusive Escapes.
        values = [json.dumps(item, ensure_ascii=False)[1:-1] for item in schema["enum"]]
        return [("string", values)]
    if schema_type == "boolean":
        return [("boolean", ["true", "false"])]
    # Freie Strings und Integer haben keine endliche aufzählbare Wertemenge.
    return [(schema_type, None)]


def candidate_value_fragment(text, string_value):
    if not string_value:
        text = text.lstrip()
        end = 0
        while end < len(text) and text[end].isalpha():
            end += 1
        return text[:end]
    # Innerhalb eines JSON-Strings nur bis zum schließenden, nicht escapten Quote.
    end = 0
    escaped = False
    while end < len(text):
        char = text[end]
        if char == '"' and not escaped:
            break
        escaped = char == "\\" and not escaped
        if char != "\\":
            escaped = False
        end += 1
    return text[:end]


def distinct_string_token_options(candidates, limit=5):
    """Wahrscheinlichste String-Anfänge ohne gegenseitige Präfix-Inklusion."""
    by_fragment = {}
    for candidate in candidates:
        fragment = candidate_value_fragment(candidate.text, string_value=True)
        if fragment:
            by_fragment[fragment] = by_fragment.get(fragment, 0.0) + candidate.probability
    ranked = sorted(by_fragment.items(), key=lambda item: item[1], reverse=True)
    selected = []
    for fragment, probability in ranked:
        normalized = fragment.strip().casefold()
        if not normalized:
            continue
        if any(
            normalized.startswith(existing.strip().casefold())
            or existing.strip().casefold().startswith(normalized)
            for existing, _ in selected
        ):
            continue
        selected.append((fragment, probability))
        if len(selected) == limit:
            break
    return selected


def grouped_integer_probabilities(candidates):
    """Varianten wie ``5`` und `` 5`` zu derselben Ganzzahl gruppieren."""
    grouped = {}
    for candidate in candidates:
        match = re.match(r"-?(?:0|[1-9][0-9]*)", candidate.text.lstrip())
        if match:
            value = int(match.group(0))
            grouped[value] = grouped.get(value, 0.0) + candidate.probability
    total = sum(grouped.values())
    if total > 1.0:
        grouped = {value: probability / total for value, probability in grouped.items()}
    return sorted(grouped.items(), key=lambda item: item[1], reverse=True)


def grouped_value_probabilities(candidates, prefix, legal_values, string_value):
    """Top-10-Tokenmasse auf kompatible Endwerte verteilen.

    Ist ein Präfix mit mehreren Werten kompatibel, wird seine Masse wie gewünscht
    gleichmäßig auf diese Werte aufgeteilt.
    """
    compatible_values = [
        value for value in legal_values
        if value.startswith(prefix) or prefix.startswith(value)
    ]
    grouped = {value: 0.0 for value in compatible_values}
    for candidate in candidates:
        fragment = candidate_value_fragment(candidate.text, string_value)
        combined = prefix + fragment
        matches = [
            value for value in compatible_values
            if value.startswith(combined) or combined.startswith(value)
        ]
        if matches:
            share = candidate.probability / len(matches)
            for value in matches:
                grouped[value] += share
    # Float32-Rundung kann die aufsummierte Masse um wenige ULP über 1 heben.
    total = sum(grouped.values())
    if total > 1.0:
        grouped = {value: probability / total for value, probability in grouped.items()}
    return sorted(grouped.items(), key=lambda item: item[1], reverse=True)


def print_value_token_logprobs(result, schema, show_detail_logprobs: bool = True):
    """Value-Logprobs ausgeben.

    Mit ``show_detail_logprobs=False`` werden die einzelnen Token-Top-10
    ausgeblendet; die nach legalem Endwert gruppierte Ausgabe bleibt sichtbar.
    """
    spans = json_leaf_value_spans(result.text)
    parsed = json.loads(result.text)
    options = leaf_value_options(schema, parsed)
    leaves = list(zip(spans, options))

    if not show_detail_logprobs:
        # Eine kompakte Ausgabe pro vollständigem JSON-Blattwert. Dadurch werden
        # beispielsweise die Tokens ``posit`` + ``iv`` als ``positiv`` gezeigt.
        token_ranges = []
        token_start = 0
        for step in result.steps:
            token_end = len(step.generated_text)
            token_ranges.append((token_start, token_end, step))
            token_start = token_end

        for (start, end), (value_type, legal_values) in leaves:
            value_steps = [
                step for token_start, token_end, step in token_ranges
                if start < token_end and end > token_start
            ]
            if not value_steps:
                continue
            selected_value = result.text[start:end]
            first_step = value_steps[0]
            last_step = value_steps[-1]
            step_label = (
                str(first_step.step)
                if first_step.step == last_step.step
                else f"{first_step.step}–{last_step.step}"
            )
            print(f"\nValue {selected_value!r} (Schritt {step_label})")
            selected_log_probability = sum(
                math.log(step.selected_constrained_probability)
                for step in value_steps
            )
            selected_probability = math.exp(selected_log_probability)
            candidates = first_step.constrained_top20[:10]
            grouped_heading = "Gruppierte legale Werte (Masse der Token-Top-10):"
            if legal_values:
                grouped = grouped_value_probabilities(
                    candidates, "", legal_values,
                    string_value=value_type == "string",
                )
                grouped_limit = 10
            elif value_type == "integer":
                grouped = grouped_integer_probabilities(candidates)
                grouped_limit = 5
            elif value_type == "string":
                grouped = distinct_string_token_options(
                    first_step.constrained_top20, limit=5
                )
                grouped_limit = 5
                grouped_heading = (
                    "Gruppierte legale Werte (erste unterschiedliche String-Tokens):"
                )
            else:
                grouped = None
                grouped_limit = 0

            if grouped is not None:
                print(f"  {grouped_heading}")
                for rank, (value, probability) in enumerate(
                    grouped[:grouped_limit], start=1
                ):
                    log_probability = math.log(probability) if probability else -math.inf
                    print(
                        f"  {rank:2}. Value={value!r:<18} "
                        f"p={probability:.8f} log(p)={log_probability:.6f}"
                    )
            else:
                print(
                    "  Wahrscheinlichkeit des gewählten Value-Tokenpfads: "
                    f"p={selected_probability:.8g} "
                    f"log(p)={selected_log_probability:.6f}"
                )
        return

    previous_end = 0
    for step in result.steps:
        token_end = len(step.generated_text)
        leaf = next(
            ((span, legal) for span, legal in leaves
             if span[0] < token_end and span[1] > previous_end),
            None,
        )
        if leaf is None:
            previous_end = token_end
            continue
        (start, end), (value_type, legal_values) = leaf
        candidates = step.constrained_top20[:10]
        print(
            f"\nValue-Schritt {step.step}: gewählt {step.selected_token} "
            f"(ID={step.selected_token_id})"
        )
        if show_detail_logprobs:
            print("  Alle Top 10 nach Grammatikmaske:")
            for rank, candidate in enumerate(candidates, start=1):
                print(
                    f"  {rank:2}. ID={candidate.token_id:<7} "
                    f"Token={candidate.token:<18} "
                    f"p={candidate.probability:.8f} "
                    f"log(p)={candidate.log_probability:.6f}"
                )
        if legal_values:
            prefix = result.text[start:min(previous_end, end)]
            grouped = grouped_value_probabilities(
                candidates, prefix, legal_values,
                string_value=value_type == "string",
            )
            print("  Gruppierte legale Werte (Masse der Token-Top-10):")
            for rank, (value, probability) in enumerate(grouped[:10], start=1):
                log_probability = math.log(probability) if probability else -math.inf
                print(
                    f"  {rank:2}. Value={value!r:<18} "
                    f"p={probability:.8f} log(p)={log_probability:.6f}"
                )
        previous_end = token_end


def run_local_example(
    backend,
    model_type,
    prompt,
    max_new_tokens=128,
    show_detail_logprobs: bool = False,
):
    schema = schema_from_pydantic(model_type)
    constraint = OutlinesSchemaConstraint(schema, backend.tokenizer, backend.vocab_size)
    result = generate(
        backend=backend,
        constraint=constraint,
        schema=schema,
        prompt=prompt,
        temperature=0,
        top_p=1,
        seed=0,
        max_new_tokens=max_new_tokens,
    )
    assert result.status == "abgeschlossen", result.error or result.text
    value = json.loads(result.text)
    assert result.validation == {
        "json": True,
        "jsonschema": True,
        "value": value,
    }
    assert result.steps
    print("\nGeneriertes JSON:")
    print(json.dumps(value, ensure_ascii=False, indent=2))
    print_value_token_logprobs(
        result, schema, show_detail_logprobs=show_detail_logprobs
    )
    # Nur das typisierte Pydantic-Objekt an den Test zurückgeben.
    return model_type.model_validate(value)


@pytest.mark.integration
def test_enum_category_schema_with_local_model(backend):
    categorization = run_local_example(
        backend, Categorization, CATEGORY_PROMPT, max_new_tokens=32
    )
    assert categorization.category is Category.TECHNIK

@pytest.mark.integration
def test_review_schema_with_local_model(backend):
    review = run_local_example(backend, Review, REVIEW_PROMPT)
    assert review.sentiment is Sentiment.POSITIV
    assert review.verified is True
    assert isinstance(review.tags, list)
    assert review.details.score == 5



@pytest.mark.integration
def test_boolean_approval_schema_with_local_model(backend):
    approval = run_local_example(
        backend, Approval, APPROVAL_PROMPT, max_new_tokens=32
    )
    assert approval.approved is True
