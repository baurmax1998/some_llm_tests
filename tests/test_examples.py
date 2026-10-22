"""Echte lokale End-to-End-Läufe mit aus Pydantic abgeleiteten Schemata."""
import json
import os
from enum import Enum

import pytest
from pydantic import BaseModel, ConfigDict

from json_schema_playground.generation import generate
from json_schema_playground.model_backend import DEFAULT_MODEL, MLXModelBackend
from json_schema_playground.value_analysis import (
    analyze_value_probabilities,
    format_value_probabilities,
)
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
    analyses = analyze_value_probabilities(
        result, schema, include_token_details=show_detail_logprobs
    )
    print(format_value_probabilities(analyses))
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
