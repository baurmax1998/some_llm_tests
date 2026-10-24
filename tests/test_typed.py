"""Echte lokale End-to-End-Läufe mit aus Pydantic abgeleiteten Schemata."""
import json
import time
from enum import Enum

import pytest
from pydantic import BaseModel, ConfigDict

from src.llm import generate
from src.mask import OutlinesSchemaConstraint
from src.typed import (
    analyze_value_probabilities,
    format_value_probabilities,
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


def run_local_example(
    backend,
    model_type,
    prompt,
    max_new_tokens=128,
    show_detail_logprobs: bool = False,
):
    total_started = time.perf_counter()

    started = time.perf_counter()
    schema = schema_from_pydantic(model_type)
    schema_seconds = time.perf_counter() - started

    started = time.perf_counter()
    constraint = OutlinesSchemaConstraint(schema, backend.tokenizer, backend.vocab_size)
    constraint_seconds = time.perf_counter() - started

    generation_started = time.perf_counter()
    first_token_seconds = None

    def record_first_token(_trace):
        nonlocal first_token_seconds
        if first_token_seconds is None:
            first_token_seconds = time.perf_counter() - generation_started

    result = generate(
        backend=backend,
        constraint=constraint,
        schema=schema,
        prompt=prompt,
        temperature=0,
        top_p=1,
        seed=0,
        max_new_tokens=max_new_tokens,
        on_step=record_first_token,
    )
    generation_seconds = time.perf_counter() - generation_started
    assert result.status == "abgeschlossen", result.error or result.text

    started = time.perf_counter()
    value = json.loads(result.text)
    json_seconds = time.perf_counter() - started
    assert result.validation == {
        "json": True,
        "jsonschema": True,
        "value": value,
    }
    assert result.steps
    print("\nGeneriertes JSON:")
    print(json.dumps(value, ensure_ascii=False, indent=2))

    started = time.perf_counter()
    analyses = analyze_value_probabilities(
        result, schema, include_token_details=show_detail_logprobs
    )
    analysis_seconds = time.perf_counter() - started
    print(format_value_probabilities(analyses))

    started = time.perf_counter()
    typed_value = model_type.model_validate(value)
    pydantic_seconds = time.perf_counter() - started
    total_seconds = time.perf_counter() - total_started

    model_seconds = sum(step.model_seconds for step in result.steps)
    mask_seconds = sum(step.mask_seconds for step in result.steps)
    other_generation_seconds = max(
        0.0, generation_seconds - model_seconds - mask_seconds
    )
    token_count = len(result.steps)
    print(f"\nTiming für {model_type.__name__} ({token_count} Tokens):")
    print(f"  Schema aus Pydantic:       {schema_seconds:8.3f} s")
    print(f"  Constraint initialisieren: {constraint_seconds:8.3f} s")
    print(f"  Zeit bis zum ersten Token: {first_token_seconds or 0.0:8.3f} s")
    print(f"  Modell-Inferenz gesamt:    {model_seconds:8.3f} s")
    print(f"    davon Prompt-Prefill:     {result.steps[0].model_seconds:8.3f} s")
    print(f"    davon weitere Tokens:     {model_seconds - result.steps[0].model_seconds:8.3f} s")
    print(f"  Schema-Masken gesamt:      {mask_seconds:8.3f} s")
    print(f"  Sonstige Generierung:      {other_generation_seconds:8.3f} s")
    print(f"  Generierung gesamt:        {generation_seconds:8.3f} s")
    print(f"  JSON parsen:               {json_seconds:8.3f} s")
    print(f"  Wahrscheinlichkeitsanalyse:{analysis_seconds:8.3f} s")
    print(f"  Pydantic validieren:       {pydantic_seconds:8.3f} s")
    print(f"  Gesamtlauf:                {total_seconds:8.3f} s")
    print(f"  Durchsatz:                 {token_count / generation_seconds:8.2f} Token/s")

    # Nur das typisierte Pydantic-Objekt an den Test zurückgeben.
    return typed_value


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
