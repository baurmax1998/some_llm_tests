# ASCII Snake

Live-Konsolenspiel ohne zusätzliche Abhängigkeiten. Es lässt sich gleichzeitig
über die Tastatur und über eine lokale TCP-API steuern.

## Start

Einmalig installieren:

```bash
pip install -e .
```

Danach im ersten Terminal:

```bash
snake
```

Die API lauscht fest und ausschließlich auf `127.0.0.1:8765`.

## Tastatur

- Pfeiltasten oder WASD: steuern
- Leertaste/Enter: starten bzw. fortsetzen
- `p`: Pause
- `r`: zurücksetzen
- `?` oder `h`: Hilfe
- `q`: beenden

## TCP-API

Das einfache TCP-Protokoll verwendet UTF-8 und genau eine Anfrage/Antwort pro
Zeile. Anfragen sind JSON-Objekte oder einfache Textbefehle. Antworten sind
immer JSON. Verfügbare Befehle: `start`, `pause`, `reset`, `up`, `down`, `left`,
`right`, `state`, `screenshot`, `help` und `quit`. `screenshot` liefert die
aktuelle ASCII-Ansicht im Feld `screenshot` zurück.

```bash
printf '%s\n' '{"command":"start"}' '{"command":"direction","direction":"up"}' '{"command":"state"}' | nc 127.0.0.1 8765

# Kurzform:
printf 'left\n' | nc 127.0.0.1 8765
```

Beispielantwort:

```json
{"ok": true, "state": {"status": "running", "score": 0, "width": 30, "height": 18, "direction": "right", "snake": [[15, 9], [14, 9], [13, 9]], "food": [4, 3]}}
```

`quit` beendet sowohl Server als auch Spiel.

## Lokaler LLM-Gamer

Während Snake in einem Terminal läuft, kann in einem zweiten Terminal ein
lokales MLX-Modell die Steuerung übernehmen:

```bash
gamer
```

Der Gamer lässt Snake auch während der Modellberechnung ohne Pause weiterlaufen.
Das Modell muss den durch seine Antwortzeit entstehenden Versatz anhand des
Prompts, des Spieltempos und vorheriger Laufzeiten einschätzen. Ein
Schema-Constraint erlaubt ausschließlich gemappte Aktionslisten, die danach über
TCP ausgeführt werden. Prompt-Vorlage: `snake_gamer.md`.

Für jeden Lauf entsteht unter `snake_gamer_logs/` eine normale, gut lesbare
Textdatei mit UTC-Zeitstempeln, Prompts, Modellantworten, Aktionen, Zuständen und
Screenshots. Beide Programme besitzen bewusst keine Kommandozeilenparameter.
