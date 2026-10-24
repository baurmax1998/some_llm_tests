# ASCII Snake

Live-Konsolenspiel ohne zusätzliche Abhängigkeiten. Es lässt sich gleichzeitig
über die Tastatur und über eine lokale TCP-API steuern.

## Start

```bash
python -m src.games.snake
# oder nach `pip install -e .`:
sic-snake
```

Optionen zeigt `python -m src.games.snake --help`. Standardmäßig lauscht die
API ausschließlich auf `127.0.0.1:8765`.

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
`right`, `state`, `help` und `quit`.

```bash
printf '%s\n' '{"command":"start"}' '{"command":"direction","direction":"up"}' '{"command":"state"}' | nc 127.0.0.1 8765

# Kurzform:
printf 'left\n' | nc 127.0.0.1 8765
```

Beispielantwort:

```json
{"ok": true, "state": {"status": "running", "score": 0, "width": 30, "height": 18, "direction": "right", "snake": [[15, 9], [14, 9], [13, 9]], "food": [4, 3]}}
```

`quit` beendet sowohl Server als auch Spiel. Mit `--host 0.0.0.0` wäre die API
im Netzwerk erreichbar; ohne Authentifizierung sollte sie besser lokal bleiben.
