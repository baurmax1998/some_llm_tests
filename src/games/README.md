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

Für ein langsameres Spiel kann die Tick-Dauer erhöht werden, beispielsweise auf
eine halbe Sekunde pro Schritt:

```bash
snake --tick-time 0.5
```

`--speed 0.5` ist ein alternativer Name. Größere Werte bewegen die Schlange
langsamer. Die API lauscht fest und ausschließlich auf `127.0.0.1:8765`.

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
gewählte Ansicht im Feld `screenshot` zurück. Ohne Angabe ist sie ASCII. Für
eine reine Koordinatenansicht:

```bash
printf '%s\n' '{"command":"screenshot","mode":"coordinates"}' | nc 127.0.0.1 8765
```

Die Koordinatenansicht enthält Spielfeldgrenzen, Tickdauer, Richtung,
Kopfposition, sämtliche Körperpositionen und die Futterposition.

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

Der erste Plan wird erstellt, bevor Snake gestartet wird. Danach lässt der Gamer
Snake auch während jeder Modellberechnung ohne Pause weiterlaufen; Planung und
Ausführung überlappen sich noch nicht. Aktionen werden in echten Spielticks
ausgeführt. Vor der Ausführung wird der Plan auf den aktuellen Live-Zustand
rebasiert, auf Kollisionen simuliert und bei zu kurzem Endkorridor um eine
Sicherheitsabbiegung ergänzt. Eine fortlaufende Chatsitzung übernimmt
den MLX-KV-Cache zwischen den Anfragen. Das Modell muss den verbleibenden Versatz
anhand des Prompts, des Spieltempos und vorheriger Laufzeiten einschätzen. Ein
Schema-Constraint erlaubt ausschließlich gemappte Aktionslisten, die danach über
TCP ausgeführt werden. Prompt-Vorlage: `snake_gamer.md`.

Für jeden Lauf entsteht unter `snake_gamer_logs/` eine normale, gut lesbare
Textdatei mit UTC-Zeitstempeln, Prompts, Modellantworten, Cache-Metriken,
Aktionen, Zuständen und Screenshots.

Der Decoder unterstützt `json` und das Kurzformat `compact`, zum Beispiel
`u,01,l,03,p!`. Die Codes stehen für `up`, `left` und `screenshot`; Zeitwerte
bestehen aus genau zwei Tick-Ziffern und `!` beendet den Plan. Der Gamer wartet
auf echte Spielticks statt auf ungenaue Millisekunden. Das Format wird beim
Start ausgewählt:

```bash
gamer --format json
gamer --format compact
```

Die Ansicht für das LLM ist unabhängig davon konfigurierbar:

```bash
gamer --format compact --screenshot-mode coordinates
gamer --format json --screenshot-mode ascii
```

`--view coordinates` ist die Kurzalternative zu `--screenshot-mode`. Ohne
Parameter bleiben `json` und `ascii` die Standardwerte. `--output-format` ist
ein alternativer Name für `--format`.

Der reale Vergleich von jeweils fünf Modellanfragen mit beiden Formaten und mit
bzw. ohne KV-Cache läuft über:

```bash
pytest -q -s tests/test_gamer_cache_performance.py
```

Ein separater Fähigkeitstest lässt das echte Modell ohne Zeitdruck zehn feste
Snake-Positionen lösen. Er prüft Decoderformat, Kollisionen und Fortschritt zum
Futter und schlägt fehl, sobald eine Position nicht korrekt gelöst wird:

```bash
pytest -q -s tests/test_gamer_llm_positions.py
```
