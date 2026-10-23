# SIC Web (experimentell)

Die Webschicht ist absichtlich ein dünner FastAPI-Host. Konfiguration, Seiten
und Endpunktlogik bleiben SIC-Programme. Ein vollständiges Beispiel liegt in
`examples/todo`.

## Start

```bash
sic build examples/todo/web.sic
sic serve examples/todo/web.sic
```

`build` wertet alle Page-Goals aus und erzeugt HTML mit eingebettetem CSS,
JavaScript und dem SIC-Browser-Client. `serve` baut die Seiten ebenfalls und
startet FastAPI mit Host und Port aus der SIC-Konfiguration.

## Konfiguration

`web.sic` liefert einen Record:

```sic
goal app = {
    title: "Meine App",
    host: "127.0.0.1",
    port: 8000,
    output: "dist",
    pages: ["page.sic"],
    endpoints: [
        {name: "todos", method: "GET", path: "/api/todos", source: "todos.sic"}
    ]
}
```

Alle Pfade sind relativ zur Konfiguration und dürfen deren Projektverzeichnis
nicht verlassen. Änderungen an der Konfiguration oder den Routen benötigen
aktuell einen Neustart. Endpunktprogramme werden dagegen anhand ihres
Änderungszeitpunkts vor einer Anfrage automatisch neu kompiliert.

## Seiten

Ein Page-Goal liefert `output`, `title`, `html`, `css` und `js`:

```sic
goal page = {
    output: "index.html",
    title: "Todos",
    html: "<button data-sic-call=\"todos\">Laden</button><pre data-sic-output=\"todos\"></pre>",
    css: "button { font: inherit; }",
    js: "sic.call(\"todos\").then(console.log);"
}
```

Im Browser steht `sic.call(name, daten)` zur Verfügung. Pfadparameter wie
`{id}` werden aus `daten` eingesetzt. GET-Daten werden Query-Parameter, bei
anderen Methoden wird JSON gesendet.

Zusätzlich unterstützt der Klebecode:

- `data-sic-call="goal"` auf Buttons
- `data-sic-submit="goal"` auf Formularen
- `data-sic-output="goal"` für die einfache Text-/JSON-Ausgabe
- die Events `sic:result` und `sic:error`

In mehrzeiligen SIC-String-Buildern müssen literale JavaScript- und
CSS-Klammern als `{{` und `}}` geschrieben werden.

## Endpunkte

Freie Namen eines Endpunktgoals werden aus JSON-Body, Query- und
Pfadparametern versorgt:

```sic
goal response: String = "Hallo " + (name: String)
```

Zusätzlich ist `request` mit den Feldern `method`, `path`, `query`,
`pathParams`, `headers` und `body` verfügbar. HTTP-Strings werden anhand der
statisch ermittelten SIC-Eingabetypen nach `Int`, `Float` oder `Bool`
konvertiert.

Normale Ergebnisse werden als JSON zurückgegeben. Ein Response-Record kann
Status und Body festlegen:

```sic
goal response = {status: 201, body: {ok: true}}
```

`err` wird vorläufig zu HTTP 422. Dateioperationen laufen unter einem
prozessweiten Lock, damit die JSON-Demo nicht durch parallele Requests
beschädigt wird. Das ist noch kein Transaktions- oder Effektsystem.
