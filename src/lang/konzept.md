# Sprachkonzept: Vom Ziel zu den Voraussetzungen

## Philosophie

**Das gewünschte Ergebnis steht zuerst.** Danach beschreibt das Programm, welche Werte, Berechnungen und Daten dafür benötigt werden. Die Ausführungsreihenfolge ergibt sich aus den Abhängigkeiten.

Dieses Prinzip gilt auch für externe Funktionen: Zuerst wird benannt, **was** benötigt wird, danach, **woher** es stammt.

```text
read_json::files::std("customers.json")
```

Das bedeutet: Verwende `read_json` aus dem Modul `files` des Pakets `std`.

**Typen werden bei der ersten Verwendung festgelegt.** Separate Typdefinitionsblöcke entfallen. Typinformationen stehen direkt am Ausdruck oder werden aus eindeutigen Anforderungen abgeleitet.

```text
(customers: List<Customer>)
```

führt `customers` als Liste von `Customer`-Werten ein.

```text
(customer.orders: List<Order>)
```

legt fest, dass `Customer` ein Feld `orders` vom Typ `List<Order>` besitzt.

```text
not order.paid
```

legt ohne zusätzliche Annotation fest, dass `Order.paid` ein `Bool` ist.

Einmal festgelegte Anforderungen bleiben verbindlich. Spätere Verwendungen und Definitionen müssen dazu passen.

## Ziel

Die Sprache soll Menschen und LLMs ermöglichen, Programme **vom Ergebnis ausgehend** zu entwickeln. Gleichzeitig soll sie tokenweise eingeschränkt generiert werden können.

Ein Decoder verwaltet dafür:

- den aktuellen Syntaxzustand,
- bekannte Namen und Gültigkeitsbereiche,
- Typanforderungen und Funktionssignaturen,
- offene Definitionen,
- verfügbare Module und deren Exporte.

Der Decoder sperrt Fortsetzungen, die bereits bekannten Regeln widersprechen. Anforderungen an später definierte Werte bleiben offen, bis ihre Definition vorliegt.

Ein Programm ist vollständig, wenn alle benötigten Namen aufgelöst, alle Typanforderungen erfüllt und keine unzulässigen Abhängigkeitszyklen vorhanden sind.

Die erste Version konzentriert sich auf unveränderliche Werte, Records, Listen, Entscheidungen, Funktionen und Datenpipelines. Veränderlicher Zustand und UI-Effekte können später ergänzt werden.

## Syntax

### 1. Ziele und Wertdefinitionen

`goal` benennt das Ergebnis. `let` definiert einen unveränderlichen Wert.

```text
goal total: Int =
    paintable_area * price_per_square_meter

let paintable_area: Int =
    wall_area - window_area

let wall_area: Int = 6 * 3
let window_area: Int = 2 * 1
let price_per_square_meter: Int = 1200
```

Namen dürfen vor ihrer Wertdefinition verwendet werden. Ihre Typen müssen dabei ausdrücklich angegeben oder aus dem Kontext ableitbar sein.

Ein Name wird pro Gültigkeitsbereich nur einmal definiert. Lokale Werte werden als Ausdruck mit `in` gebunden:

```text
let subtotal = orders.map(order => order.amount).sum()
in subtotal * 2
```

Der lokale Name gilt ausschließlich im Ausdruck hinter `in`.

### 2. Typangaben bei der Verwendung

Eine Annotation legt den Typ eines Ausdrucks fest:

```text
(customers: List<Customer>)
(order.amount: Int)
```

Bei einem bisher unbekannten Feld entsteht dadurch eine Anforderung an den zugehörigen Record-Typ.

Benannte Record-Typen wie `Customer` benötigen keinen separaten Definitionsblock. Ihre bekannten Felder werden schrittweise gesammelt:

| Verwendung | Entstehende Anforderung |
|---|---|
| `(customer.name: String)` | `Customer` besitzt `name: String` |
| `(customer.orders: List<Order>)` | `Customer` besitzt `orders: List<Order>` |
| `(order.amount: Int)` | `Order` besitzt `amount: Int` |
| `not order.paid` | `Order` besitzt `paid: Bool` |

Weitere Felder sind erlaubt. Widersprüchliche Anforderungen an dasselbe Feld sind Fehler.

### 3. Typinferenz

Eine explizite Annotation entfällt, wenn der Kontext den Typ eindeutig bestimmt.

```text
orders.filter(order => order.paid)
```

`order` erhält seinen Typ aus der Liste. `paid` muss ein `Bool` sein, weil `filter` einen booleschen Ausdruck erwartet.

```text
not order.paid
```

erzwingt ebenfalls `paid: Bool`.

Erwartete Ergebnistypen können rückwärts durch Ausdrücke propagiert werden. Bleiben mehrere Typen möglich, verlangt die Sprache eine Annotation, statt stillschweigend einen Typ auszuwählen.

Die Grundtypen der ersten Version sind:

| Typ | Bedeutung |
|---|---|
| `Bool` | Wahrheitswert |
| `Int` | Ganze Zahl |
| `Float` | Gleitkommazahl |
| `String` | Zeichenfolge |
| `T?` | Wert vom Typ `T` oder `null` |
| `List<T>` | Liste mit Elementtyp `T` |
| `Map<K, V>` | Schlüssel-Wert-Zuordnung |

Implizite Typumwandlungen sind zunächst ausgeschlossen. Nullable Werte werden mit `?` markiert und mit `null`, `??`, sicheren Zugriffen oder Typverengung behandelt:

```text
let score: Int? = scores.get("Ada")
in if score != null then score + 1 else 0
```

```text
customer?.address?.city ?? "unbekannt"
```

### 4. Records und Listen

Records enthalten benannte Felder:

```text
let customer = {
    name: "Ada",
    orders: [
        { amount: 8000, paid: false },
        { amount: 4500, paid: true }
    ]
}
```

Feldzugriffe verwenden einen Punkt:

```text
customer.name
```

Records müssen alle zuvor geforderten Felder mit passenden Typen bereitstellen. Zusätzliche Felder sind zulässig.

Listen verwenden eckige Klammern:

```text
let names = ["Ada", "Linus", "Grace"]
```

Maps verwenden Record-Klammern mit String-Schlüsseln:

```text
let scores: Map<String, Int> = {"Ada": 10, "Grace": 12}
```

Listen, Strings und Maps unterstützen Indexzugriffe wie `names[0]` und `scores["Ada"]`.

Für leere Listen muss der Elementtyp aus dem Kontext oder einer Annotation bekannt sein:

```text
let names: List<String> = []
```

### 5. Entscheidungen und Operatoren

`if` ist ein Ausdruck, der einen Wert liefert:

```text
goal message: String =
    if is_member
    then "Willkommen zurück!"
    else "Willkommen!"

let is_member: Bool = true
```

Die Bedingung muss `Bool` sein. Beide Zweige müssen zum erwarteten Ergebnistyp passen.

`match` prüft Werte durch geordnete Muster. Ein Binding- oder `_`-Muster schließt die Fälle ab:

```text
match score {
    null => 0,
    value => value + 1
}
```

Die Sprache unterstützt zunächst:

| Bereich | Operatoren |
|---|---|
| Arithmetik | `+`, `-`, `*`, `/`, `%` |
| Vergleiche | `==`, `!=`, `<`, `<=`, `>`, `>=` |
| Boolesche Ausdrücke | `not`, `and`, `or` |
| Null-Behandlung | `??`, `?.`, Vergleich mit `null` |
| Strings | `+` zur Verkettung |

Bei überladenen Operatoren wie `+` muss der Kontext die Bedeutung eindeutig bestimmen.

### 6. Pipelines und Schleifen

Listen werden mit typisierten Pipelines verarbeitet:

```text
orders
    .filter(order => not order.paid)
    .map(order => order.amount)
    .sum()
```

| Operation | Bedeutung |
|---|---|
| `.filter(predicate)` | Passende Elemente behalten |
| `.map(transform)` | Elemente in neue Werte umwandeln |
| `.sum()` | Zahlen addieren; leere Liste ergibt `0` |
| `.fold(initial, reducer)` | Elemente mit einem Akkumulator zusammenführen |
| `.flat_map(transform)` | Ergebnislisten zusammenführen |
| `.find(predicate)` | Ersten Treffer als optionalen Wert liefern |
| `.any(predicate)` / `.all(predicate)` | Prädikat über die Liste prüfen |
| `.count()` | Anzahl der Elemente liefern |
| `.take(n)` / `.drop(n)` | Listenpräfix nehmen oder überspringen |
| `.first()` / `.last()` | Optionales Rand-Element liefern |
| `.sort()` | Vergleichbare Werte sortieren |
| `.group_by(key)` | Elemente in einer `Map` gruppieren |

Eine komplexere Schleife wird als `fold` ausgedrückt:

```text
orders.fold(0, (total, order) =>
    if order.paid
    then total
    else total + order.amount
)
```

Der Akkumulator startet mit `0`. Jeder Durchlauf liefert seinen nächsten Wert. Eine leere Liste liefert den Startwert.

Lambda-Parameter erhalten ihre Typen durch die Listenoperation und gelten ausschließlich innerhalb des Lambda-Ausdrucks.

### 7. Externe Funktionen und Herkunftspfade

Externe Namen werden direkt bei ihrer Verwendung aufgelöst. Die Schreibweise beginnt mit dem gewünschten Symbol und nennt anschließend seine Herkunft:

```text
read_json::files::std("customers.json")
```

| Bestandteil | Bedeutung |
|---|---|
| `read_json` | Gewünschte Funktion |
| `files` | Enthaltendes Modul |
| `std` | Enthaltendes Paket |

Ein vollständiger Herkunftspfad ist eine zusammengehörige Namensreferenz. Der Aufruf mit `(...)` bezieht sich auf das aufgelöste Symbol.

Ein separater Importblock entfällt.

```text
let customers =
    read_json::files::std("customers.json")
```

Die Trennzeichen haben unterschiedliche Aufgaben:

- `::` beschreibt die Herkunft eines Symbols.
- `.` greift auf ein Feld oder eine Operation eines Werts zu.

**Auflösung während der Generierung:**

1. Das Projekt stellt ein Register verfügbarer Pakete, Module und Exporte bereit.
2. Nach einem Symbolnamen wie `read_json` bleiben die passenden Herkunftspfade als Kandidaten übrig.
3. Weitere Pfadsegmente schränken diese Kandidaten ein.
4. Sobald der Pfad eindeutig ist, steht die exportierte Signatur fest.

Bei Mehrdeutigkeit dürfen Typanforderungen nur verwendet werden, wenn sie für alle verbleibenden Kandidaten gelten. Ein vollständiger Pfad muss eindeutig auflösbar sein.

Das Auflösen einer Referenz führt keinen Anwendungscode aus. Effekte entstehen erst durch die spätere Ausführung des Aufrufs.

### 8. Aufrufbare Berechnungen

Jedes `goal` und jedes `let` benennt eine aufrufbare Berechnung. Freie Namen bilden ihre Eingaben. Ein lokal definierter Zwischenwert kann beim Aufruf überschrieben werden; dann werden seine eigenen Voraussetzungen weder verlangt noch ausgewertet.

```text
goal ergebnis: Int = a + b
let b = c * d
```

Mögliche Aufrufe sind:

```text
ergebnis(a: 10, b: 5)
ergebnis(a: 10, c: 2, d: 3)
b(c: 2, d: 3)
```

Für `ergebnis` gilt damit die Voraussetzung `a und (b oder (c und d))`. Argumente sind benannt. Typen werden aus der Verwendung in der Berechnung und aus den Argumenten inferiert. Lambdas bleiben für lokale Transformationen wie `map` verfügbar, sind aber nicht mehr erforderlich, um eine Top-Level-Berechnung zu definieren.

Die Voraussetzungen aller Zweige eines `if` müssen statisch versorgt sein. Zur Laufzeit wird weiterhin nur der gewählte Zweig ausgeführt.

### 9. Vollständiges Beispiel

```text
goal outstanding: List<Int> =
    (customers: List<Customer>)
        .map(customer =>
            (customer.orders: List<Order>)
                .filter(order => not order.paid)
                .map(order => (order.amount: Int))
                .sum()
        )

let customers = [
    {
        name: "Ada",
        orders: [
            { amount: 8000, paid: false },
            { amount: 4500, paid: true }
        ]
    },
    {
        name: "Grace",
        orders: [
            { amount: 2500, paid: false }
        ]
    }
]
```

Das Programm legt beim Lesen schrittweise fest:

| Ausdruck | Festlegung |
|---|---|
| `customers: List<Customer>` | Typ der Datenquelle |
| `customer.orders: List<Order>` | Bestellliste eines Kunden |
| `not order.paid` | Boolesches Feld `paid` |
| `order.amount: Int` | Ganzzahliges Feld `amount` |
| `.sum()` | Aggregation zu einem ganzzahligen Betrag |

Die späteren Daten erfüllen diese Anforderungen. Das Ergebnis ist:

```text
[8000, 2500]
```

Alternativ kann die Datenquelle aus einem Modul stammen:

```text
let customers =
    read_json::files::std("customers.json")
```

Für diese Variante muss `read_json` laut seiner Modulsignatur den erwarteten Datentyp unterstützen und die eingelesenen Daten zur Laufzeit dagegen prüfen.

### 10. Prüfung und Ausführung

Die Verarbeitung erfolgt in drei Phasen:

1. **Generierung:** Syntax prüfen, Namen einführen, Typanforderungen sammeln und widersprüchliche Tokens sperren.
2. **Abschlussprüfung:** Offene Definitionen auflösen, Typen prüfen und unzulässige Abhängigkeitszyklen erkennen.
3. **Ausführung:** Werte entsprechend ihren Abhängigkeiten berechnen und externe Daten zur Laufzeit validieren.

Typkorrektheit garantiert weder vorhandene Dateien noch fachlich richtige Ergebnisse. Diese Prüfungen gehören zur Laufzeit beziehungsweise zu Tests.

## Aufrufbare Module

Ein Herkunftspfad kann außer einer externen Host-Funktion auch ein `goal` oder `let` aus einer anderen Sprachdatei bezeichnen:

```text
goal gesamt: Float =
    rechnungsbetrag::preise::shop(
        grundpreis: 100.0,
        punkte: 20.0,
        divisor: 2.0
    )
```

Das Modulregister hält dafür nicht nur eine flache Funktionssignatur, sondern Ergebnistypen, Eingabetypen, den Abhängigkeitsgraphen, überschreibbare Zwischenwerte und Fehlerausgänge. Sowohl das Goal als auch seine Lets sind exportierbar. Das Auflösen des Pfads führt keinen Anwendungscode aus.

In der Python-Integration registriert `ModuleRegistry.register_source(modul, paket, quelltext)` eine Sprachdatei. Die Zuordnung realer Dateien zu Modul- und Paketnamen bleibt Aufgabe des Projekts.

## Benannte Fehler

`err` beendet nur den tatsächlich ausgewerteten Berechnungsweg. Der optionale Ausdruck hinter `=` ist der Nutzwert des Fehlers:

```text
err divideByZeroError
err divideByZeroError = "Der Divisor darf nicht null sein"
```

Ein einfacher Aufruf propagiert einen Fehler automatisch. Es gibt keine `Ok`-Wrapper und kein erforderliches `?`. Wird die Berechnung, in der die Fehlerstelle liegt, durch ein Argument überschrieben oder liegt sie in einem nicht gewählten Zweig, entsteht der Fehler nicht.

Fehleridentitäten enthalten intern ihre Modulherkunft. Gleichnamige Fehler verschiedener Module werden daher nicht vereinigt. Kann ein Name bei der Entnahme mehrere Identitäten bezeichnen, lehnt die Typprüfung ihn als mehrdeutig ab.

## Mehrfachbindung und Fehlerbehandlung

Ergebnis, Zwischenwerte und Fehler können namensbasiert aus genau einem Aufruf entnommen werden:

```text
rechnungsbetrag, rabatt, divideByZeroError =
    rechnungsbetrag::preise::shop(
        grundpreis: 100.0,
        punkte: 20.0,
        divisor: 0.0
    )
```

Die Mehrfachbindung ist keine veränderliche Zuweisung. Sie steht wie ein `let` nach der ersten Verwendung ihrer Ausgänge. Nur ausdrücklich aufgeführte Fehler werden abgefangen; alle anderen propagieren weiter.

Für die erste Version wird die bislang offene Präsenzsyntax durch Nullable-Werte konkretisiert:

- normale Ergebnis- und Zwischenausgänge haben beim Abfangen eines Fehlers den Typ `T?`;
- ein Fehler mit Nutzwert hat den Typ `T?`;
- ein Fehler ohne Nutzwert hat den Typ `Bool?`: `true` bedeutet vorhanden, `null` bedeutet nicht vorhanden;
- `!= null`, `if`, `match` und `??` dienen als vorhandene Präsenzprüfung und Typverengung;
- Umbenennung bei der Mehrfachbindung wird zunächst nicht unterstützt.

Damit ist eine ungeprüfte Verwendung eines möglicherweise fehlenden Ergebnisses ein Typfehler. Der folgende Ausdruck ist unzulässig, solange `rechnungsbetrag: Float?` gilt:

```text
rechnungsbetrag + 1.0
```

Eine zulässige Behandlung ist beispielsweise:

```text
if divideByZeroError != null
then divideByZeroError
else "ok"
```

Wenn mehrere unabhängige Auswertungen scheitern könnten, bestimmt die normale, reproduzierbare Auswertungsreihenfolge des Ausdrucks den zuerst propagierten Fehler. Nicht gewählte Zweige und durch Overrides abgeschnittene Abhängigkeiten werden nicht ausgewertet.