# Lokales JSON-Schema-Decoding-Playground

Ein transparenter Prototyp für echtes constrained decoding auf Apple Silicon. Das Modell erzeugt nur Tokenfolgen, die sich noch zu einem Dokument des vorgegebenen JSON Schemas vervollständigen lassen. Es findet **keine nachträgliche JSON-Reparatur** statt.

## Voraussetzungen und Installation

- Apple-Silicon-Mac (MLX läuft nicht auf Intel-Macs/Linux)
- macOS, Python 3.11 oder neuer
- genügend RAM für `Qwen3-8B-4bit` (Download beim ersten Start)

Mit `uv` (empfohlen; kein separates `pip` oder manuelles venv nötig):

```bash
uv sync
```

`uv sync` legt `.venv` an, installiert die Anwendung sowie die `dev`-Gruppe und verwendet den eingecheckten Lockfile. Für eine Installation ohne Testwerkzeuge: `uv sync --no-dev`.

Die Versionen sind nach einem API-Abgleich mit `mlx-lm 0.32.2`, `outlines 1.3.3` und `outlines-core 0.2.14` festgeschrieben. Outlines besitzt in dieser Version einen offiziellen `MLXLM`-Adapter. Dieses Projekt verwendet absichtlich die tiefer liegenden, tokenizerkompatiblen Outlines-Core-Bausteine `Vocabulary`, `Index` und `Guide`: So kann es die erlaubte Menge vor dem Sampling auslesen und den vollständigen Trace erstellen. Die Vokabularerzeugung entspricht dem offiziellen Outlines-MLX-Backend (`convert_tokens_to_string` statt isolierter Token-Decodes). Der MLX-Pfad wurde gegen die aktuelle `load`-/Modell-Cache-API geprüft; ein vollständiger 8B-Modelllauf ist wegen Download und Speicherbedarf kein Bestandteil der normalen Tests.

## CLI

```bash
uv run python -m json_schema_playground.cli \
  --prompt 'Bewerte: Das Produkt gefällt mir sehr.' \
  --schema examples/review.schema.json \
  --temperature 0.7 --top-p 0.9 --seed 42 \
  --max-new-tokens 256 --trace trace.json
```

Alternativ über das definierte Kommando:

```bash
uv run json-schema-playground --prompt '...' --schema examples/review.schema.json
```

Das Standardmodell ist `mlx-community/Qwen3-8B-4bit`. `--model` überschreibt es. Qwens Thinking-Modus wird beim Chat-Template explizit mit `enable_thinking=False` deaktiviert. Bei Tokenizern ohne diese Option wird das normale Chat-Template benutzt.

### Beispiele

Drei JSON-Schema-Beispiele samt zugehörigem Prompt/Kontext liegen unter `examples/`:

- `review.schema.json` + `review.prompt.txt`: strukturierte Bewertung mit Enum, Boolean, Array und verschachteltem Objekt
- `category.schema.json` + `category.prompt.txt`: Auswahl genau eines Wertes aus `technik`, `sport` oder `kultur`
- `approval.schema.json` + `approval.prompt.txt`: reine Ja/Nein-Entscheidung im Feld `approved`

```bash
uv run json-schema-playground \
  --prompt 'Ordne den Text der passenden Kategorie zu: Apple stellt einen neuen Chip vor.' \
  --schema examples/category.schema.json

uv run json-schema-playground \
  --prompt 'Entscheide, ob die Anfrage genehmigt werden soll: Alle Bedingungen sind erfüllt.' \
  --schema examples/approval.schema.json
```

## Streamlit

```bash
uv run streamlit run app.py
```

Die Oberfläche zeigt Live-JSON, Status, einen auswählbaren Schritt, `allowed_mass`, Laufzeiten sowie Top-20 vor der Maske, nach der Maske und nach Temperature/Top-p. Gesperrte Einträge sind gekennzeichnet. Im Greedy-Modus wird die Auswahl ausdrücklich als deterministisch angezeigt.

## JSON Schema aus Pydantic

Mit `schema_from_pydantic()` kann das Schema direkt aus einer Pydantic-v2-Modellklasse abgeleitet werden:

```python
from enum import Enum
from pydantic import BaseModel, ConfigDict
from json_schema_playground.schema_engine import schema_from_pydantic

class Category(str, Enum):
    TECHNIK = "technik"
    SPORT = "sport"
    KULTUR = "kultur"

class Result(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: Category

schema = schema_from_pydantic(Result)
```

`extra="forbid"` ist notwendig, damit Pydantic `additionalProperties: false` erzeugt. Verschachtelte Pydantic-Modelle und String-Enums werden unterstützt. Der Adapter löst ausschließlich lokale, von Pydantic erzeugte `$defs`/`$ref` auf; rekursive Modelle werden verständlich abgelehnt. Nach der Generierung kann das Ergebnis zusätzlich mit `Result.model_validate(...)` typisiert geprüft werden.

## Unterstützter Schemaumfang

- `object` mit `properties`, `required` und **explizit** `additionalProperties: false`
- `string`, auch String-`enum`
- `boolean`
- `integer`
- Arrays mit genau einem einheitlichen `items`-Schema
- beliebig daraus verschachtelte Objekte
- Annotationen `title`, `description`, `$comment` und `$schema`

Andere Typen und Constraints (`number`, `null`, `pattern`, Längen-/Zahlengrenzen, Tupel-Arrays, `$ref`, `anyOf`, `oneOf` usw.) werden vor dem Laden des Modells verständlich abgelehnt. Es gibt keine stillschweigende Abschwächung. Das ursprüngliche Schema wird am Ende unabhängig mit `json.loads` und `jsonschema` validiert.

**Feldreihenfolge:** Der aktuelle Outlines-Core-Regexcompiler gibt eine feste, aus der Reihenfolge von `properties` abgeleitete Objektfeldreihenfolge vor. Das ist eine Einschränkung des Prototyps, obwohl JSON selbst keine Feldreihenfolge fordert. Der Adapter in `schema_engine.py` kapselt diese Implementierung und kann später ersetzt werden.

## Decoding und Trace

Die Komponenten sind getrennt:

- `model_backend.py`: MLX-Modell, Tokenizer, Chat-Template und KV-Cache
- `schema_engine.py`: Teilmengenprüfung, Outlines-Compiler und `SchemaConstraint`
- `sampling.py`: float32-Softmax, Maske, Temperature, Top-p, Auswahl
- `generation.py`: Schleife und Abbruchbedingungen
- `trace.py`: vollständige Schrittdaten
- `validation.py`: unabhängige Abschlussprüfung
- `cli.py` / `app.py`: Oberflächen

Pro Schritt wird zuerst die Grammatikmaske und erst danach Temperature/Top-p angewendet. Eine leere erlaubte Menge ist ein Fehler und führt niemals zu unbeschränktem Sampling. EOS gehört bei Outlines Core erst im akzeptierenden Zustand zur erlaubten Menge; die Schleife beendet das Dokument bereits kontrolliert beim Erreichen dieses Zustands. Prompt-Tokens gelangen nicht in den Grammatikzustand.

`allowed_mass` ist die Summe der ursprünglichen Modellwahrscheinlichkeiten aller erlaubten Tokens. Die Top-20 stammen jeweils aus der vollständigen Verteilung und werden nicht separat normalisiert. Bei `temperature=0` ist die finale Auswahlverteilung als One-Hot-Verteilung protokolliert.

## Tests

```bash
uv sync
uv run pytest
```

`tests/test_examples.py` enthält drei echte lokale End-to-End-Tests. Prompts und Pydantic-Modellklassen stehen direkt im Testcode; die JSON-Schemata werden mit `schema_from_pydantic()` daraus erzeugt. Die Tests lesen keine Beispieldateien. Es werden keine Modell- oder Tokenizer-Mocks verwendet: Das MLX-Modell wird einmal geladen, jeder Prompt wird verarbeitet, constrained decoding wird ausgeführt und das Ergebnis mit `json.loads`, `jsonschema` sowie dem ursprünglichen Pydantic-Modell validiert. Entsprechend benötigen die Tests Apple Silicon, Modell-RAM und beim ersten Lauf den Modelldownload.

Standardmäßig wird `mlx-community/Qwen3-8B-4bit` verwendet. Ein anderes lokales MLX-Modell kann für die Tests gesetzt werden:

```bash
TEST_MODEL_ID=mlx-community/Qwen3-4B-4bit uv run pytest
```

## Bekannte Grenzen

- Nur der oben dokumentierte Schemaausschnitt ist implementiert.
- Objektfelder haben eine feste Reihenfolge.
- Strukturelle Gültigkeit garantiert keine sachlich richtigen Werte.
- Ein gültiges Präfix muss nicht innerhalb eines vorgegebenen Tokenbudgets fertig werden; der Status lautet dann `unvollständig`.
- Der erste Prototyp verwendet keine Repetition Penalty und keinen Logit Bias.
- Der Modell-Prefill wird als ein MLX-Aufruf ausgeführt; sehr lange Prompts sind nicht aufgeteilt.
