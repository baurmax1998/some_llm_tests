# SIC Todo-Webdemo

```bash
sic build examples/todo/web.sic
sic serve examples/todo/web.sic
```

Danach: <http://127.0.0.1:8000>

- `web.sic` konfiguriert App, Seiten und Endpunkte vollständig in SIC.
- `page.sic` wird beim Build zu `dist/index.html` inklusive Browser-Kleber kompiliert.
- Jedes API-Goal wird einmal geprüft und danach nur bei einer Dateiänderung neu kompiliert.
- Die JSON-Datei ist absichtlich nur eine vorläufige, pro Prozess serialisierte Persistenz.
