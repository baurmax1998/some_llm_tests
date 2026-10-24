"""Lokaler LLM-Agent für das per TCP erreichbare ASCII-Snake."""

from __future__ import annotations

import json
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
from src.mask import OutlinesSchemaConstraint

TERMINAL_STATES = {"game_over", "won"}
ACTION_COMMANDS = {"up", "down", "left", "right", "wait", "screenshot"}
HOST = "127.0.0.1"
PORT = 8765
MODEL = DEFAULT_MODEL
PROMPT_PATH = Path(__file__).with_name("snake_gamer.md")
LOG_DIR = Path("snake_gamer_logs")
MAX_TURNS = 200
MAX_WAIT_MS = 2000
TCP_TIMEOUT = 5.0
TEMPERATURE = 0.0
TOP_P = 1.0
SEED = 0
MAX_TOKENS = 256

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
                    "milliseconds": {"type": "integer"},
                },
                "required": ["command", "milliseconds"],
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


class GamerCommandDecoder:
    """Erzwingt das Aktionsformat tokenweise und mappt JSON auf Befehle."""

    def __init__(self, backend: MLXModelBackend) -> None:
        self.backend = backend

    def generate_actions(
        self,
        user_prompt: str,
        system_prompt: str,
        *,
        temperature: float,
        top_p: float,
        seed: int,
        max_tokens: int,
    ) -> tuple[list[dict[str, int | str]], str]:
        # Constraints sind zustandsbehaftet und müssen je Antwort neu entstehen.
        constraint = OutlinesSchemaConstraint(
            ACTION_SCHEMA, self.backend.tokenizer, self.backend.vocab_size
        )
        trace = generate(
            backend=self.backend,
            constraint=constraint,
            schema=ACTION_SCHEMA,
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            max_new_tokens=max_tokens,
        )
        if trace.status != "abgeschlossen":
            raise RuntimeError(trace.error or f"LLM-Antwort unvollständig: {trace.text}")
        value = json.loads(trace.text)
        actions = value["actions"]
        if not isinstance(actions, list):
            raise ValueError("Der Decoder erhielt keine Aktionsliste.")
        return actions, trace.text


def _turn_prompt(
    turn: int,
    screenshot: str,
    feedback: list[dict[str, Any]],
    previous_inference_seconds: float | None,
) -> str:
    previous = json.dumps(feedback, ensure_ascii=False, indent=2) if feedback else "Keine (erster Zug)."
    delay = (
        f"{previous_inference_seconds:.3f} Sekunden"
        if previous_inference_seconds is not None
        else "noch unbekannt (erster Zug)"
    )
    return f"""Plane Zug {turn}.

Aktueller TCP-Screenshot (nur der Zustand im Moment der Aufnahme):
```text
{screenshot}
```

Ergebnisse der Aktionen des vorherigen Zugs:
{previous}

Dauer der vorherigen LLM-Berechnung: {delay}

WICHTIG: Das Spiel läuft auch während deiner Berechnung ohne Pause weiter. Wenn
deine Antwort ausgeführt wird, ist der Screenshot bereits ungefähr um die
übliche LLM-Berechnungsdauer veraltet. Schätze die Bewegung anhand von Richtung,
Spieltempo und bisheriger Berechnungsdauer voraus. Erzeuge jetzt eine kurze
Aktionsliste als JSON.
"""


def _sleep_while_alive(
    client: SnakeClient, milliseconds: int, stop_event: threading.Event
) -> dict[str, Any]:
    deadline = time.monotonic() + milliseconds / 1000
    state: dict[str, Any] = {}
    while not stop_event.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(remaining, 0.1))
        state = client.request("state")["state"]
        if state["status"] in TERMINAL_STATES:
            break
    return state or client.request("state")["state"]


def run_agent() -> int:
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    stop_event = threading.Event()

    with RunLogger(LOG_DIR) as logger:
        print(f"Logdatei: {logger.path}")
        logger.write(
            "run_started",
            host=HOST,
            port=PORT,
            model=MODEL,
            prompt_file=str(PROMPT_PATH),
            system_prompt=system_prompt,
        )
        try:
            with SnakeClient(HOST, PORT, TCP_TIMEOUT) as client:
                backend = MLXModelBackend(MODEL)
                decoder = GamerCommandDecoder(backend)
                feedback: list[dict[str, Any]] = []
                previous_inference_seconds: float | None = None

                state = client.request("state")["state"]
                if state["status"] in {"ready", "paused"}:
                    # Ab hier läuft das Spiel ohne künstliche Denkpausen durch.
                    state = client.request("start")["state"]

                for turn in range(1, MAX_TURNS + 1):
                    state = client.request("state")["state"]
                    if state["status"] in TERMINAL_STATES:
                        break

                    observation = client.request("screenshot")
                    screenshot = observation["screenshot"]
                    state = observation["state"]
                    logger.write("screenshot", turn=turn, source="turn_start", screenshot=screenshot, state=state)
                    print(f"\n--- Zug {turn} ---\n{screenshot}")

                    user_prompt = _turn_prompt(
                        turn, screenshot, feedback, previous_inference_seconds
                    )
                    logger.write("prompt", turn=turn, system_prompt=system_prompt, user_prompt=user_prompt)
                    inference_started = time.monotonic()
                    actions, raw_response = decoder.generate_actions(
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
                        raw_response=raw_response,
                        actions=actions,
                    )
                    print(f"LLM ({inference_seconds:.3f}s): {raw_response}")
                    previous_inference_seconds = inference_seconds

                    # Die Schlange ist während der Inferenz weitergelaufen. Vor
                    # dem Ausführen nur das Spielende prüfen, nicht pausieren.
                    live_state = client.request("state")["state"]
                    if live_state["status"] in TERMINAL_STATES:
                        final_shot = client.request("screenshot")
                        logger.write(
                            "screenshot",
                            turn=turn,
                            source="game_end_during_inference",
                            screenshot=final_shot["screenshot"],
                            state=final_shot["state"],
                        )
                        state = final_shot["state"]
                        break

                    feedback = []
                    for index, action in enumerate(actions):
                        command = str(action["command"])
                        requested_ms = int(action["milliseconds"])
                        milliseconds = max(0, min(requested_ms, MAX_WAIT_MS))
                        action_log: dict[str, Any] = {
                            "index": index,
                            "command": command,
                            "requested_milliseconds": requested_ms,
                            "milliseconds": milliseconds,
                        }

                        if command in {"up", "down", "left", "right"}:
                            response = client.request(command)
                            action_log["direction_accepted"] = response.get("direction_accepted")
                            current_state = _sleep_while_alive(client, milliseconds, stop_event)
                        elif command == "wait":
                            current_state = _sleep_while_alive(client, milliseconds, stop_event)
                        elif command == "screenshot":
                            shot = client.request("screenshot")
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
                        final_shot = client.request("screenshot")
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
    return run_agent()


if __name__ == "__main__":
    raise SystemExit(main())
