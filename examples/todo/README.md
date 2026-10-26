# SIC Todo-Webdemo

```bash
sic build examples/todo/app.sic
sic serve examples/todo/app.sic
```

Danach: <http://127.0.0.1:8000>

Die statisch abgeleiteten Schnittstellen aller SIC-Dateien der Codebasis
lassen sich separat erzeugen:

```bash
sic describe examples/todo/app.sic
```

Dabei entstehen unter `.out/i/` gespiegelte, C-ähnliche `.i`-Dateien,
zum Beispiel `.out/i/backend/create_todo.i`. Verwendete Imports erhalten
ebenfalls eine auflösbare Datei, etwa `.out/i/std/files/write_json.i`. Die
Beschreibungen enthalten für jedes Goal und `let` die freien Eingaben,
überschreibbaren Zwischenwerte, typisierten Ausgänge sowie mögliche Fehler.

## Struktur

```text
app.sic                 # Webprojekt und Routen
frontend/
  app.sic               # deklarative App::web
backend/
  list_todos.sic
  create_todo.sic
  toggle_todo.sic
  delete_todo.sic
  todos.json
.out/                    # generierte Ausgabe (ignoriert)
```

Das typisierte `App::web` in `app.sic` enthält nur `port`, `pages` und
`endpoints`; Host, Titel, Ausgabeordner und Backend-Registrierung entstehen
automatisch. `frontend/app.sic` liefert direkt `View::web`. `todos` und `nextId`
sind normale `let`-Abhängigkeiten. Die Todo-Zeilen entstehen mit `map`; Event-
Lambdas rufen unmittelbar die Goals aus dem Paket `backend` auf. Nach einer
Mutation lädt die kleine Browser-Runtime die neu ausgewertete App.

Die UI verwendet die Standardbibliothekstypen `App::web` und `View::web` sowie
`Column::web`, `Input::web`, `ListView::web`, `Row::web`, `Checkbox::web`,
`Text::web` und `Button::web`.
