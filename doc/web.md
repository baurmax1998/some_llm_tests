# SIC Web (experimentell)

Die Webschicht ist absichtlich ein dünner FastAPI-Host. Konfiguration, Seiten
und Endpunktlogik bleiben SIC-Programme. Ein vollständiges Beispiel liegt in
`examples/todo`.

## Start

```bash
sic build examples/todo/app.sic
sic serve examples/todo/app.sic
```

`build` wertet alle Page-Goals aus und erzeugt HTML mit eingebettetem CSS,
JavaScript und dem SIC-Browser-Client. `serve` baut die Seiten ebenfalls und
startet FastAPI mit Host und Port aus der SIC-Konfiguration.

## Konfiguration

`app.sic` liefert den Standardbibliothekstyp `App::web`. Erforderlich sind nur
Port, Seiten und Endpunkte:

```sic
goal app: App::web = {
    port: 8000,
    pages: ["frontend/app.sic"],
    endpoints: [
        {name: "listTodos", method: "GET", path: "/api/todos", source: "backend/list_todos.sic"}
    ]
}
```

Alle Pfade sind relativ zur Konfiguration und dürfen deren Projektverzeichnis
nicht verlassen. Titel (`"SIC App"`), Host (`"127.0.0.1"`) und Ausgabeordner
(`.out`) besitzen Defaults. Jeder Endpunkt wird automatisch unter
`name::backend` registriert; eine zusätzliche `modules`-Liste ist dafür nicht
nötig. Änderungen an Konfiguration oder Routen benötigen aktuell einen
Neustart. Endpunktprogramme werden anhand ihres Änderungszeitpunkts vor einer
Anfrage automatisch neu kompiliert.

## Komponenten

HTML wird über zwei kleine, sichere Bausteine erzeugt:

```sic
text::web("Text <wird escaped>")
element::web("p", {"class": "intro"}, [child])
```

`element` nimmt Tag, `Map<String, String>` mit Attributen und `List<Html::web>`
mit bereits gerenderten Kindern entgegen und liefert selbst `Html::web`.
Dadurch kann HTML nicht mit beliebigen Strings verwechselt werden. Attributwerte
werden ebenfalls escaped. Void-Elemente wie `input` benötigen eine leere Kinderliste.

Eine wiederverwendbare Komponente ist ein normales SIC-Modul:

```sic
goal card: Html::web = element::web("article", {"class": "card"}, [heading])
let heading = element::web("h2", {"class": "title"}, [
    text::web(title: String)
])
```

Nach der Registrierung als Modul `card` im Paket `components` wird sie mit
benannten, statisch geprüften Argumenten zusammengesetzt:

```sic
goal pageHtml: Html::web = card::frontend(title: "Hallo")
```

Jedes `let` eines Moduls ist ebenfalls exportierbar. Dadurch können kleine
Komponenten intern bleiben oder gezielt von anderen Modulen zusammengesetzt
werden. Das Todo-Beispiel trennt Layout, Todo-Oberfläche und Browserverhalten
unter `examples/todo/frontend/`. Wenn Modul und Goal gleich heißen, genügt der
kurze Paketpfad `layout::frontend` statt `layout::layout::frontend`.

## Seiten

Ein Page-Goal hat den Typ `Page::web` aus der Web-Standardbibliothek. Dieser
Typ verlangt die Felder `output`, `title`, `html`, `css` und `js` als Strings:

```sic
goal page: Page::web = {
    output: "index.html",
    title: "Todos",
    html: todoPage::frontend(),
    css: "",
    js: "sic.call(\"todos\").then(console.log);"
}
```

Im Browser steht `sic.call(name, daten)` zur Verfügung. Der Aufruf wird über
die WebSocket-Verbindung der View übertragen; das gilt auch für `POST`, `PUT`,
`PATCH` und `DELETE`. Pfadparameter wie `{id}` werden serverseitig aus `daten`
eingesetzt. Die deklarierte HTTP-Methode und der aufgelöste Pfad stehen dem
SIC-Goal weiterhin über `request` zur Verfügung. Die konfigurierten HTTP-Routen
bleiben für externe Clients und direkte API-Aufrufe parallel erreichbar.

Zusätzlich unterstützt der Klebecode:

- `data-sic-call="goal"` auf Buttons
- `data-sic-submit="goal"` auf Formularen
- `data-sic-output="goal"` für die einfache Text-/JSON-Ausgabe
- die Events `sic:result` und `sic:error`

In mehrzeiligen SIC-String-Buildern müssen literale JavaScript- und
CSS-Klammern als `{{` und `}}` geschrieben werden.

## Deklarative Apps

Alternativ zu `Page::web` kann eine Frontend-Datei direkt `View::web` liefern.
Der Build verwendet dann `index.html` und den Projekttitel. Views werden
unmittelbar zusammengesetzt; eigene State-, Action- oder Funktionsdeklarationen
sind nicht nötig:

```sic
goal view: View::web =
    Column::web([
        Input::web(
            placeholder: "Neue Aufgabe",
            on_submit: title => createTodo::backend(id: nextId, title: title)
        ),
        ListView::web(todos.map(todo =>
            Row::web([
                Checkbox::web(
                    todo.done,
                    on_change: _ => toggleTodo::backend(id: todo.id)
                ),
                Text::web(todo.title),
                Button::web(
                    "Löschen",
                    on_click: _ => deleteTodo::backend(id: todo.id)
                )
            ])
        ))
    ])

let todos = listTodos::backend()
let nextId = todos.count() + 1
```

Event-Lambdas werden beim Build als Browseraktionen erfasst, nicht ausgeführt.
Eine aktive View hält eine WebSocket-Verbindung. Darüber laufen sowohl ihre
Aktionen als auch reaktive Aktualisierungen. Passende Effektmeldungen markieren
die View als veraltet; die Runtime wertet sie neu aus und ersetzt den
View-Wurzelknoten, einschließlich aktualisierter Event-Bindungen.

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
beschädigt wird. Lesezugriffe abonnieren automatisch den technischen
`FileChanged::files::std`-Effekt, Schreibzugriffe lösen ihn aus. Diese
prozesslokale Effektverteilung ist noch kein Transaktionssystem.
