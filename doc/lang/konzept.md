# SIC-Sprachkonzept

SIC beschreibt zuerst das Ergebnis und danach dessen Voraussetzungen. Dateien enden auf `.sic`.

## Programm

```sic
goal total: Int = area * price
let area = width * height
```

- Genau ein `goal`, immer zuerst.
- `let` definiert nur einen zuvor verwendeten Namen.
- Werte sind unveränderlich; Abhängigkeiten bestimmen die Ausführung.
- Freie Namen sind Eingaben. Typen entstehen aus Verwendung und Annotationen.
- Typen: `Bool`, `Int`, `Float`, `String`, `T?`, `List<T>`, `Map<K,V>` und schrittweise inferierte Records.
- Keine impliziten Typumwandlungen.

Lokale Bindung:

```sic
let subtotal = orders.map(x => x.amount).sum()
in subtotal * 2
```

## Ausdrücke

Unterstützt werden Literale, Records, Maps, Listen, Index- und Feldzugriffe, Lambdas, `if`, `match`, `null`, `??`, `?.`, Arithmetik, Vergleiche sowie `not`, `and`, `or`.

Mehrzeilige String-Builder verwenden dreifache Anführungszeichen und betten
Ausdrücke mit `{...}` ein. Doppelte Klammern erzeugen literale Klammern:

```sic
goal greeting: String = """Hallo {name}!
Du hast {messages.count()} Nachrichten.
{{name}} bleibt Text."""
```

Listen bieten `filter`, `map`, `flat_map`, `fold`, `sum`, `find`, `any`, `all`, `count`, `take`, `drop`, `first`, `last`, `sort` und `group_by`. Maps bieten `keys`, `values`, `get` und `contains_key`.

## Berechnungen

Jedes `goal` und `let` ist mit benannten Argumenten aufrufbar:

```sic
goal result: Int = a + b
let b = c * d
```

```sic
result(a: 10, b: 5)
result(a: 10, c: 2, d: 3)
b(c: 2, d: 3)
```

Ein übergebener Zwischenwert ersetzt seine Berechnung und Voraussetzungen. Alle Zweige müssen statisch versorgt sein; zur Laufzeit wird nur der gewählte Zweig ausgeführt.

## Module

Herkunftspfade ersetzen Imports:

```sic
invoice::prices::shop(base: 100.0, discount: 20.0)
```

`goal` und `let` sind exportierbar. Auflösung führt keinen Anwendungscode aus. Das Register speichert Typen, Abhängigkeiten, Ersetzungen, Zwischenwerte und Fehler. Python registriert Quellen mit `ModuleRegistry.register_source(modul, paket, source)`.

## Fehler

```sic
err divideByZero
err divideByZero = "Divisor ist null"
```

`err` beendet nur den ausgewerteten Weg. Einfache Aufrufe propagieren Fehler automatisch. Überschriebene Berechnungen und nicht gewählte Zweige lösen keinen Fehler aus. Fehleridentitäten enthalten ihre Modulherkunft.

Benannte Ausgänge werden gemeinsam entnommen:

```sic
invoice, discount, divideByZero =
    invoice::prices::shop(base: 100.0, points: 20.0, divisor: 0.0)
```

Nur entnommene Fehler sind abgefangen. Dann gelten:

- Ergebnis und Zwischenwerte: `T?`
- Fehler mit Nutzwert: `T?`
- Fehler ohne Nutzwert: `Bool?` (`true` vorhanden, `null` abwesend)

Verfügbarkeit wird mit `!= null`, `match` oder `??` nachgewiesen. Ungeprüfte Verwendung fehlbarer Werte ist ein Typfehler. Umbenennung wird noch nicht unterstützt; gleichnamige Fehler verschiedener Herkunft sind bei mehrdeutiger Entnahme unzulässig.

## Verarbeitung

1. Tokenweise Syntax-, Namens- und Typprüfung.
2. Abschlussprüfung von Definitionen, Typen und Zyklen.
3. Bedarfsgesteuerte Ausführung mit Laufzeitvalidierung externer Werte.

## Web-Prototyp

Ein dünner FastAPI-Host kann SIC-Goals als HTTP-Endpunkte ausführen und
Page-Goals zu HTML, CSS und JavaScript bauen. Konfiguration und Beispiel sind
in [web.md](../web.md) beschrieben.
