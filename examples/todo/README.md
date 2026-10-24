# SIC Todo-Webdemo

```bash
sic build examples/todo/app.sic
sic serve examples/todo/app.sic
```

Danach: <http://127.0.0.1:8000>

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
