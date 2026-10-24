"""Lokaler LLM-Agent für das per TCP erreichbare ASCII-Snake."""

from __future__ import annotations

import argparse
import json
import math
import os
import pprint
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.llm import DEFAULT_MODEL, MLXModelBackend, generate
from src.mask import OutlinesRegexConstraint, OutlinesSchemaConstraint

TERMINAL_STATES = {"game_over", "won"}
ACTION_COMMANDS = {"up", "down", "left", "right", "wait", "screenshot"}
DIRECTION_VECTORS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
OPPOSITE_DIRECTION = {"up": "down", "down": "up", "left": "right", "right": "left"}
HOST = "127.0.0.1"
PORT = 8765
MODEL = DEFAULT_MODEL
PROMPT_PATH = Path(__file__).with_name("snake_gamer.md")
LOG_DIR = Path("snake_gamer_logs")
MAX_TURNS = 200
MAX_ACTION_TICKS = 50
SAFETY_TICK_MARGIN = 2
TCP_TIMEOUT = 5.0
TEMPERATURE = 0.0
TOP_P = 1.0
SEED = 0
MAX_TOKENS = 256
DEFAULT_OUTPUT_FORMAT = "json"

# Drei Ziffern verhindern, dass der Regex-Guide schon nach der ersten Ziffer
# einen vermeintlich vollständigen Plan meldet. `!` beendet den gesamten Plan.
COMPACT_ACTION_PATTERN = r"(?:[udlrw],[0-9][0-9]|p)(?:,(?:[udlrw],[0-9][0-9]|p))*!"
COMPACT_CODES = {
    "u": "up",
    "d": "down",
    "l": "left",
    "r": "right",
    "w": "wait",
    "p": "screenshot",
}
JSON_FORMAT_PROMPT = """Antworte ausschließlich als JSON-Objekt im Schema des Decoders.
`ticks` ist die Anzahl vollständiger Spielschritte, keine Zeit in Millisekunden.
Beispiel: {"actions":[{"command":"up","ticks":1},{"command":"screenshot","ticks":0}]}
"""
COMPACT_FORMAT_PROMPT = """Ausgabe ausschließlich als kurze Liste ohne Leerzeichen.
Codes: u=hoch, d=runter, l=links, r=rechts, w=Richtung halten, p=Screenshot.
Auf u/d/l/r/w folgen genau zwei Ziffern für die Anzahl Spielticks (00 bis 99), keine Millisekunden.
p bedeutet NICHT Start; der Gamer startet automatisch. Planende ist ! Beispiel: u,01,l,03,p!
"""

ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": sorted(ACTION_COMMANDS),
                    },
                    "ticks": {"type": "integer"},
                },
                "required": ["command", "ticks"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["actions"],
    "additionalProperties": False,
}


class SnakeClient:
    """Client für das JSON-lines-Protokoll des Snake-Servers."""

    def __init__(self, host: str, port: int, timeout: float = 5.0) -> None:
        self.socket = socket.create_connection((host, port), timeout=timeout)
        self._reader = self.socket.makefile("r", encoding="utf-8", newline="\n")
        self._writer = self.socket.makefile("w", encoding="utf-8", newline="\n")

    def request(self, command: str, **values: Any) -> dict[str, Any]:
        request = {"command": command, **values}
        self._writer.write(json.dumps(request, ensure_ascii=False) + "\n")
        self._writer.flush()
        line = self._reader.readline()
        if not line:
            raise ConnectionError("Der Snake-Server hat die Verbindung geschlossen.")
        response = json.loads(line)
        if not isinstance(response, dict):
            raise ConnectionError("Ungültige Antwort vom Snake-Server.")
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error", "Unbekannter Serverfehler")))
        return response

    def close(self) -> None:
        self._reader.close()
        self._writer.close()
        self.socket.close()

    def __enter__(self) -> SnakeClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class RunLogger:
    """Schreibt einen gut lesbaren Text-Log pro Agentenlauf."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.path = directory / f"snake_gamer_{stamp}_{os.getpid()}.txt"
        self._file = self.path.open("x", encoding="utf-8")
        self._started = time.monotonic()
        self._lock = threading.Lock()

    def write(self, event: str, **data: Any) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        elapsed = time.monotonic() - self._started
        lines = ["=" * 80, f"[{timestamp} | +{elapsed:.3f}s] {event.upper()}"]
        for key, value in data.items():
            rendered = value if isinstance(value, str) else pprint.pformat(value, width=100, sort_dicts=False)
            lines.extend((f"\n{key}:", rendered))
        lines.append("")
        with self._lock:
            self._file.write("\n".join(lines) + "\n")
            self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def system_prompt_for_format(system_prompt: str, output_format: str) -> str:
    format_prompt = COMPACT_FORMAT_PROMPT if output_format == "compact" else JSON_FORMAT_PROMPT
    return system_prompt.rstrip() + "\n\n" + format_prompt


def parse_compact_actions(text: str) -> list[dict[str, int | str]]:
    """Mappt beispielsweise ``u,01,l,03,p!`` auf Gamer-Aktionen."""
    if not text.endswith("!"):
        raise ValueError("Ein kompakter Aktionsplan muss mit ! enden.")
    parts = text[:-1].split(",")
    actions: list[dict[str, int | str]] = []
    index = 0
    while index < len(parts):
        code = parts[index]
        if code not in COMPACT_CODES:
            raise ValueError(f"Unbekannter kompakter Aktionscode: {code}")
        command = COMPACT_CODES[code]
        if command == "screenshot":
            actions.append({"command": command, "ticks": 0})
            index += 1
            continue
        if (
            index + 1 >= len(parts)
            or len(parts[index + 1]) != 2
            or not parts[index + 1].isdigit()
        ):
            raise ValueError(f"Nach {code} werden genau zwei Tick-Ziffern erwartet.")
        actions.append({"command": command, "ticks": int(parts[index + 1])})
        index += 2
    return actions


class GamerCommandDecoder:
    """Erzwingt das Aktionsformat und hält optional eine gecachte Chatsitzung."""

    def __init__(
        self,
        backend: MLXModelBackend,
        *,
        use_cache: bool = True,
        output_format: str = "json",
    ) -> None:
        if output_format not in {"json", "compact"}:
            raise ValueError("output_format muss 'json' oder 'compact' sein")
        self.backend = backend
        self.use_cache = use_cache
        self.output_format = output_format
        self._messages: list[dict[str, str]] = []
        self._system_prompt: str | None = None
        self._last_response: str | None = None
        self._last_response_token_ids: list[int] = []

    def generate_actions(
        self,
        user_prompt: str,
        system_prompt: str,
        *,
        temperature: float,
        top_p: float,
        seed: int,
        max_tokens: int,
    ) -> tuple[list[dict[str, int | str]], str, dict[str, Any]]:
        # Der Modell-KV-Cache darf weiterleben; der zustandsbehaftete
        # Ausgabe-Constraint muss dagegen für jede Antwort neu entstehen.
        if self.output_format == "json":
            constraint = OutlinesSchemaConstraint(
                ACTION_SCHEMA, self.backend.tokenizer, self.backend.vocab_size
            )
            schema: dict[str, Any] | None = ACTION_SCHEMA
            output_validator = None
        else:
            constraint = OutlinesRegexConstraint(
                COMPACT_ACTION_PATTERN, self.backend.tokenizer, self.backend.vocab_size
            )
            schema = None
            output_validator = parse_compact_actions
        effective_system_prompt = system_prompt_for_format(system_prompt, self.output_format)
        if not self.use_cache or self._system_prompt != effective_system_prompt:
            self._messages = [{"role": "system", "content": effective_system_prompt}]
            self._system_prompt = effective_system_prompt
            self._last_response = None
            self._last_response_token_ids = []
        self._messages.append({"role": "user", "content": user_prompt})
        cache_reused = self.backend.prepare_messages(
            self._messages,
            reuse_cache=self.use_cache,
            previous_response=self._last_response,
            previous_response_token_ids=self._last_response_token_ids,
        )
        prompt_tokens = self.backend.last_prompt_token_count
        trace = generate(
            backend=self.backend,
            constraint=constraint,
            schema=schema,
            prompt=user_prompt,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            max_new_tokens=max_tokens,
            output_validator=output_validator,
            prepare_backend=False,
        )
        if trace.status != "abgeschlossen":
            raise RuntimeError(trace.error or f"LLM-Antwort unvollständig: {trace.text}")
        if self.output_format == "json":
            value = json.loads(trace.text)
            actions = value["actions"]
            if not isinstance(actions, list):
                raise ValueError("Der Decoder erhielt keine Aktionsliste.")
        else:
            actions = parse_compact_actions(trace.text)
        if self.use_cache:
            historical_response = self.backend.last_generation_prefix + trace.text
            self._messages.append({"role": "assistant", "content": historical_response})
            self._last_response = trace.text
            self._last_response_token_ids = [step.selected_token_id for step in trace.steps]
        else:
            self._messages.clear()
        metrics = {
            "cache_reused": cache_reused,
            "cache_status": self.backend.last_cache_miss_reason,
            "prompt_tokens_processed": prompt_tokens,
            "generated_tokens": len(trace.steps),
            "model_seconds": sum(step.model_seconds for step in trace.steps),
            "mask_seconds": sum(step.mask_seconds for step in trace.steps),
        }
        return actions, trace.text, metrics


def _state_facts(
    state: dict[str, Any], previous_inference_seconds: float | None
) -> str:
    head_x, head_y = state["snake"][0]
    food_x, food_y = state["food"]
    width, height = state["width"], state["height"]
    direction = state["direction"]
    wall_spaces = {
        "left": head_x,
        "right": width - 1 - head_x,
        "up": head_y,
        "down": height - 1 - head_y,
    }
    vectors = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
    opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}
    occupied = {tuple(point) for point in state["snake"][:-1]}
    legal = []
    for candidate, (dx, dy) in vectors.items():
        next_point = (head_x + dx, head_y + dy)
        if (
            candidate != opposite[direction]
            and 0 <= next_point[0] < width
            and 0 <= next_point[1] < height
            and next_point not in occupied
        ):
            legal.append(candidate)

    horizontal = "rechts" if food_x > head_x else "links" if food_x < head_x else "gleiche X-Spalte"
    vertical = "unten" if food_y > head_y else "oben" if food_y < head_y else "gleiche Y-Zeile"
    timing = "LLM_DELAY_TICKS=unbekannt (erster Plan vor Spielstart)."
    if previous_inference_seconds is not None:
        expected_ticks = max(1, math.ceil(previous_inference_seconds / state["tick_seconds"]))
        spaces = wall_spaces[direction]
        if expected_ticks > spaces:
            projection = f"KOLLISION nach spätestens {spaces + 1} Ticks"
        else:
            dx, dy = vectors[direction]
            remaining = spaces - expected_ticks
            projection = (
                f"Kopf ungefähr bei ({head_x + dx * expected_ticks},{head_y + dy * expected_ticks}), "
                f"danach nur noch {remaining} Felder bis zur Wand"
            )
            if remaining <= 2:
                projection += "; IN DIESER ANTWORT ABBIEGEN, NICHT GERADEAUS WEITERFAHREN"
        timing = (
            f"LLM_DELAY_TICKS={expected_ticks} (letzte Antwort {previous_inference_seconds:.3f}s). "
            f"Benötigter Sicherheitskorridor am Planende={expected_ticks + SAFETY_TICK_MARGIN} Ticks. "
            f"Ohne Richtungswechsel: {projection}."
        )

    return (
        f"Kopf=({head_x},{head_y}); Richtung={direction}; Futter=({food_x},{food_y}) "
        f"→ {abs(food_x-head_x)} Felder {horizontal}, {abs(food_y-head_y)} Felder {vertical}.\n"
        f"Freie Felder bis zur Wand: links={wall_spaces['left']}, rechts={wall_spaces['right']}, "
        f"oben={wall_spaces['up']}, unten={wall_spaces['down']}.\n"
        f"Sofort erlaubte Richtungen: {', '.join(legal)}. {timing}"
    )


def _brief_feedback(feedback: list[dict[str, Any]]) -> str:
    if not feedback:
        return "Keine (erster Zug)."
    return "; ".join(
        f"{item['command']} {item['ticks']} Ticks"
        + (f" (angenommen={item['direction_accepted']})" if "direction_accepted" in item else "")
        for item in feedback
    )


def _turn_prompt(
    turn: int,
    screenshot: str,
    screenshot_mode: str,
    state: dict[str, Any],
    feedback: list[dict[str, Any]],
    previous_inference_seconds: float | None,
) -> str:
    timing_note = (
        "Das Spiel ist noch nicht gestartet; dieser Plan wird vor dem Start erzeugt."
        if state["status"] == "ready"
        else "Das Spiel läuft während der Berechnung ohne Pause weiter."
    )
    return f"""AKTUELLER ZUG {turn}. Nur dieser Zustand ist maßgeblich.

Exakte Zustandsanalyse:
{_state_facts(state, previous_inference_seconds)}

TCP-Ansicht zum selben Zeitpunkt (Modus={screenshot_mode}):
```text
{screenshot}
```

Vorher ausgeführt: {_brief_feedback(feedback)}
{timing_note}

Wähle eine Richtung zum Futter, aber vermeide zuerst die Wand. Wenn du bewusst
geradeaus weiterfahren willst, verwende `w`/`wait` statt die aktuelle Richtung
noch einmal auszugeben. Bei Kollisionsprognose musst du jetzt eine andere
erlaubte Richtung wählen. `p`/`screenshot` beobachtet nur
und startet das Spiel nicht. Antworte exakt im vorgegebenen Ausgabeformat.
"""


def _advance_simulation(
    snake: list[tuple[int, int]],
    direction: str,
    width: int,
    height: int,
    food: tuple[int, int] | None,
) -> tuple[bool, tuple[int, int] | None]:
    dx, dy = DIRECTION_VECTORS[direction]
    new_head = (snake[0][0] + dx, snake[0][1] + dy)
    eating = new_head == food
    occupied = snake if eating else snake[:-1]
    if not (0 <= new_head[0] < width and 0 <= new_head[1] < height) or new_head in occupied:
        return False, food
    snake.insert(0, new_head)
    if eating:
        food = None
    else:
        snake.pop()
    return True, food


def _straight_runway(
    snake: list[tuple[int, int]], direction: str, width: int, height: int
) -> int:
    simulated = list(snake)
    runway = 0
    while runway < width * height:
        moved, _ = _advance_simulation(simulated, direction, width, height, None)
        if not moved:
            break
        runway += 1
    return runway


def apply_safety_condition(
    actions: list[dict[str, int | str]],
    state: dict[str, Any],
    required_runway_ticks: int,
) -> tuple[list[dict[str, int | str]], list[str]]:
    """Rebasiert den LLM-Plan auf den Live-Zustand und sichert sein Ende ab."""
    snake = [tuple(point) for point in state["snake"]]
    direction = str(state["direction"])
    food = tuple(state["food"]) if state["food"] is not None else None
    width, height = int(state["width"]), int(state["height"])
    safe: list[dict[str, int | str]] = []
    screenshots: list[dict[str, int | str]] = []
    notes: list[str] = []
    danger = False

    for action in actions:
        command = str(action["command"])
        ticks = max(0, min(int(action["ticks"]), MAX_ACTION_TICKS))
        if command == "screenshot":
            screenshots.append({"command": command, "ticks": 0})
            continue
        requested_direction = direction if command == "wait" else command
        if requested_direction == OPPOSITE_DIRECTION[direction]:
            notes.append(f"{command} verworfen: direkte Umkehr von {direction}")
            continue
        if command in DIRECTION_VECTORS and ticks == 0:
            ticks = 1
            notes.append(f"{command} auf 1 Tick angehoben, damit die Richtung wirksam wird")
        completed = 0
        for _ in range(ticks):
            moved, food = _advance_simulation(
                snake, requested_direction, width, height, food
            )
            if not moved:
                danger = True
                notes.append(f"{command} nach {completed} sicheren Ticks gekürzt: Kollision voraus")
                break
            direction = requested_direction
            completed += 1
        if completed:
            safe.append({"command": command, "ticks": completed})
        if danger:
            break

    runway = _straight_runway(snake, direction, width, height)
    if danger or runway < required_runway_ticks:
        candidates = [
            candidate
            for candidate in DIRECTION_VECTORS
            if candidate != OPPOSITE_DIRECTION[direction]
        ]
        best = max(
            candidates,
            key=lambda candidate: _straight_runway(snake, candidate, width, height),
        )
        best_runway = _straight_runway(snake, best, width, height)
        if best != direction and best_runway > 0:
            safe.append({"command": best, "ticks": 1})
            notes.append(
                f"Sicherheitsabbiegung {best}, weil Endkorridor {runway} < "
                f"benötigte {required_runway_ticks} Ticks (neuer Korridor {best_runway - 1})"
            )
        elif runway < required_runway_ticks:
            notes.append(
                f"Kein besserer Endkorridor gefunden: {runway} < {required_runway_ticks} Ticks"
            )
    return [*safe, *screenshots], notes


def _wait_game_ticks(
    client: SnakeClient, ticks: int, stop_event: threading.Event
) -> dict[str, Any]:
    state = client.request("state")["state"]
    target = int(state["tick_count"]) + ticks
    poll_seconds = min(0.05, max(0.005, float(state["tick_seconds"]) / 10))
    while int(state["tick_count"]) < target and not stop_event.is_set():
        if state["status"] in TERMINAL_STATES:
            break
        time.sleep(poll_seconds)
        state = client.request("state")["state"]
    return state


def run_agent(
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    screenshot_mode: str = "ascii",
) -> int:
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    effective_system_prompt = system_prompt_for_format(system_prompt, output_format)
    stop_event = threading.Event()

    with RunLogger(LOG_DIR) as logger:
        print(f"Logdatei: {logger.path}")
        logger.write(
            "run_started",
            host=HOST,
            port=PORT,
            model=MODEL,
            output_format=output_format,
            screenshot_mode=screenshot_mode,
            prompt_file=str(PROMPT_PATH),
            system_prompt=effective_system_prompt,
        )
        try:
            with SnakeClient(HOST, PORT, TCP_TIMEOUT) as client:
                backend = MLXModelBackend(MODEL)
                decoder = GamerCommandDecoder(backend, output_format=output_format)
                feedback: list[dict[str, Any]] = []
                previous_inference_seconds: float | None = None

                # Der erste Plan entsteht noch vor dem Start. Danach läuft das
                # Spiel ohne Pause; eine parallele Planerzeugung gibt es bewusst
                # noch nicht.
                for turn in range(1, MAX_TURNS + 1):
                    state = client.request("state")["state"]
                    if state["status"] in TERMINAL_STATES:
                        break

                    observation = client.request("screenshot", mode=screenshot_mode)
                    screenshot = observation["screenshot"]
                    state = observation["state"]
                    logger.write("screenshot", turn=turn, source="turn_start", screenshot=screenshot, state=state)
                    print(f"\n--- Zug {turn} ---\n{screenshot}")

                    user_prompt = _turn_prompt(
                        turn,
                        screenshot,
                        screenshot_mode,
                        state,
                        feedback,
                        previous_inference_seconds,
                    )
                    logger.write(
                        "prompt",
                        turn=turn,
                        system_prompt=effective_system_prompt,
                        user_prompt=user_prompt,
                    )
                    inference_started = time.monotonic()
                    actions, raw_response, generation_metrics = decoder.generate_actions(
                        user_prompt,
                        system_prompt,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        seed=SEED + turn - 1,
                        max_tokens=MAX_TOKENS,
                    )
                    inference_seconds = time.monotonic() - inference_started
                    logger.write(
                        "llm_response",
                        turn=turn,
                        inference_seconds=inference_seconds,
                        generation_metrics=generation_metrics,
                        raw_response=raw_response,
                        actions=actions,
                    )
                    print(f"LLM ({inference_seconds:.3f}s): {raw_response}")
                    previous_inference_seconds = inference_seconds

                    # Die Schlange ist während der Inferenz weitergelaufen. Vor
                    # dem Ausführen nur das Spielende prüfen, nicht pausieren.
                    live_state = client.request("state")["state"]
                    if live_state["status"] in TERMINAL_STATES:
                        final_shot = client.request("screenshot", mode=screenshot_mode)
                        logger.write(
                            "screenshot",
                            turn=turn,
                            source="game_end_during_inference",
                            screenshot=final_shot["screenshot"],
                            state=final_shot["state"],
                        )
                        state = final_shot["state"]
                        break
                    required_runway_ticks = (
                        math.ceil(inference_seconds / float(live_state["tick_seconds"]))
                        + SAFETY_TICK_MARGIN
                    )
                    original_actions = actions
                    actions, safety_notes = apply_safety_condition(
                        actions, live_state, required_runway_ticks
                    )
                    logger.write(
                        "safety_check",
                        turn=turn,
                        required_runway_ticks=required_runway_ticks,
                        original_actions=original_actions,
                        safe_actions=actions,
                        adjustments=safety_notes or ["keine"],
                        live_state=live_state,
                    )

                    if live_state["status"] in {"ready", "paused"}:
                        live_state = client.request("start")["state"]
                        logger.write("game_started", turn=turn, state=live_state)

                    feedback = []
                    for index, action in enumerate(actions):
                        command = str(action["command"])
                        requested_ticks = int(action["ticks"])
                        ticks = max(0, min(requested_ticks, MAX_ACTION_TICKS))
                        action_log: dict[str, Any] = {
                            "index": index,
                            "command": command,
                            "requested_ticks": requested_ticks,
                            "ticks": ticks,
                        }

                        if command in {"up", "down", "left", "right"}:
                            response = client.request(command)
                            action_log["direction_accepted"] = response.get("direction_accepted")
                            current_state = _wait_game_ticks(client, ticks, stop_event)
                        elif command == "wait":
                            current_state = _wait_game_ticks(client, ticks, stop_event)
                        elif command == "screenshot":
                            shot = client.request("screenshot", mode=screenshot_mode)
                            current_state = shot["state"]
                            action_log["screenshot"] = shot["screenshot"]
                            logger.write(
                                "screenshot",
                                turn=turn,
                                source="agent_action",
                                action_index=index,
                                screenshot=shot["screenshot"],
                                state=current_state,
                            )
                        else:  # Das Schema verhindert dies; zusätzliche Laufzeitgrenze.
                            raise ValueError(f"Nicht gemappter LLM-Befehl: {command}")

                        action_log["state"] = current_state
                        feedback.append(action_log)
                        logger.write("action", turn=turn, **action_log)
                        if current_state["status"] in TERMINAL_STATES:
                            break

                    final_state = client.request("state")["state"]
                    if final_state["status"] in TERMINAL_STATES:
                        final_shot = client.request("screenshot", mode=screenshot_mode)
                        logger.write(
                            "screenshot",
                            turn=turn,
                            source="game_end",
                            screenshot=final_shot["screenshot"],
                            state=final_shot["state"],
                        )
                        state = final_shot["state"]
                        break
                else:
                    state = client.request("state")["state"]
                    logger.write("max_turns_reached", turns=MAX_TURNS, state=state)

                logger.write("run_finished", state=state)
                print(f"Beendet: Status={state['status']}, Punkte={state['score']}")
                return 0
        except KeyboardInterrupt:
            stop_event.set()
            logger.write("interrupted")
            print("\nAgent abgebrochen.", file=sys.stderr)
            return 130
        except Exception as exc:
            logger.write("error", error_type=type(exc).__name__, message=str(exc))
            print(f"Snake-Gamer fehlgeschlagen: {exc}", file=sys.stderr)
            return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Lokaler LLM-Agent für TCP-Snake")
    parser.add_argument(
        "--format",
        "--output-format",
        dest="output_format",
        choices=("json", "compact"),
        default=DEFAULT_OUTPUT_FORMAT,
        help="Ausgabeformat des Command-Decoders (Standard: json)",
    )
    parser.add_argument(
        "--screenshot-mode",
        "--view",
        choices=("ascii", "coordinates"),
        default="ascii",
        help="Spielansicht für das LLM (Standard: ascii)",
    )
    args = parser.parse_args()
    return run_agent(args.output_format, args.screenshot_mode)


if __name__ == "__main__":
    raise SystemExit(main())
