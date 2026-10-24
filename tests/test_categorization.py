"""End-to-End-Test für die thematische Kategorisierung durch das lokale LLM."""
from dataclasses import dataclass

import pytest

from src.llm import generate
from src.mask import OutlinesRegexConstraint
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


PROMPT_TEMPLATE = """Ordne den folgenden Kontext genau einer Kategorie zu.

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


def decode_category(code: str) -> Categorization:
    """Dekodiert die Modellziffer direkt in das typisierte Pydantic-Objekt."""
    normalized_code = code.strip()
    if normalized_code not in CATEGORY_BY_CODE:
        raise ValueError(f"Ungültiger Kategorie-Code: {code!r}")
    return Categorization(category=CATEGORY_BY_CODE[normalized_code])


def run_categorization(backend, prompt: str) -> tuple[Categorization, str]:
    """Generiert exakt eine Ziffer und dekodiert sie über ``decode_category``."""
    constraint = OutlinesRegexConstraint(
        r"[0-3]", backend.tokenizer, backend.vocab_size
    )
    result = generate(
        backend=backend,
        constraint=constraint,
        schema=None,
        prompt=prompt,
        temperature=0,
        top_p=1,
        seed=0,
        max_new_tokens=4,
        output_validator=decode_category,
    )

    assert result.status == "abgeschlossen", result.error or result.text
    categorization = result.validation.get("value")
    assert isinstance(categorization, Categorization)
    return categorization, result.text


@pytest.mark.integration
def test_llm_categorization_eval(backend):
    """Wertet 20 Beispiele aus, ohne bei Fehlklassifikationen fehlzuschlagen."""
    results: list[tuple[CategorizationCase, Category]] = []

    for number, case in enumerate(CASES, start=1):
        categorization, raw_code = run_categorization(
            backend, PROMPT_TEMPLATE.format(context=case.context)
        )
        actual = categorization.category
        results.append((case, actual))

        marker = "✓" if actual is case.expected else "✗"
        print(
            f"\n[{number:>2}/{len(CASES)}] {marker} {case.name}: "
            f"code={raw_code!r}, erwartet={case.expected.value}, "
            f"erhalten={actual.value}",
            flush=True,
        )

    correct = sum(actual is case.expected for case, actual in results)
    failed = len(results) - correct
    accuracy = correct / len(results)

    print("\n" + "=" * 72)
    print("KATEGORISIERUNGS-EVAL")
    print("=" * 72)
    for case, actual in results:
        marker = "✓" if actual is case.expected else "✗"
        print(
            f"{marker} {case.name:<22} "
            f"erwartet: {case.expected.value:<8} erhalten: {actual.value}"
        )

    print("-" * 72)
    print(f"Korrekt:              {correct:>2}/{len(results)}")
    print(f"Fehlklassifikationen: {failed:>2}/{len(results)}")
    print(f"Genauigkeit:          {accuracy:>6.1%}")

    print("\nErgebnis je Kategorie:")
    for category in Category:
        category_results = [
            actual is case.expected
            for case, actual in results
            if case.expected is category
        ]
        category_correct = sum(category_results)
        print(
            f"  {category.value:<8} "
            f"{category_correct}/{len(category_results)} korrekt "
            f"({category_correct / len(category_results):.0%})"
        )
    print("=" * 72)

    # Fehlklassifikationen sind das Messergebnis der Eval und kein Testfehler.
    # Nur ein unvollständiger Eval-Lauf soll als Fehler gelten.
    assert len(results) == len(CASES) == 20
