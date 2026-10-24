"""Live-ASCII-Snake mit Tastatur- und lokaler TCP-Steuerung.

Starten mit::

    python -m src.games.snake

Das TCP-Protokoll besteht aus einer UTF-8-JSON-Anfrage pro Zeile und einer
JSON-Antwort pro Zeile. Einfache Textbefehle wie ``left`` sind ebenfalls
zulässig.
"""

from __future__ import annotations

import curses
import json
import random
import socketserver
import threading
import time
from dataclasses import dataclass
from typing import Any

Point = tuple[int, int]
DIRECTIONS: dict[str, Point] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
HELP = {
    "keyboard": {
        "arrows/WASD": "Richtung",
        "space/enter": "Start oder weiter",
        "p": "Pause",
        "r": "Neustart",
        "?/h": "Hilfe ein/aus",
        "q": "Beenden",
    },
    "tcp": {
        "format": "eine JSON-Anfrage je Zeile; alternativ einfacher Textbefehl",
        "commands": ["start", "pause", "reset", "up", "down", "left", "right", "state", "screenshot", "help", "quit"],
        "examples": [
            {"command": "start"},
            {"command": "direction", "direction": "up"},
            {"command": "state"},
        ],
    },
}


@dataclass(frozen=True)
class GameConfig:
    width: int = 30
    height: int = 18
    tick_seconds: float = 0.20

    def __post_init__(self) -> None:
        if self.width < 8 or self.height < 6:
            raise ValueError("Das Spielfeld muss mindestens 8x6 groß sein.")
        if self.tick_seconds <= 0:
            raise ValueError("tick_seconds muss größer als 0 sein.")


class SnakeGame:
    """Thread-sicherer, von UI oder API steuerbarer Spielzustand."""

    def __init__(self, config: GameConfig | None = None, *, seed: int | None = None) -> None:
        self.config = config or GameConfig()
        self._random = random.Random(seed)
        self._lock = threading.RLock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            x = self.config.width // 2
            y = self.config.height // 2
            self.snake: list[Point] = [(x, y), (x - 1, y), (x - 2, y)]
            self.direction = "right"
            self._pending_direction = "right"
            self.food: Point | None = None
            self.score = 0
            self.status = "ready"
            self._place_food()

    def start(self) -> None:
        with self._lock:
            if self.status in {"ready", "paused"}:
                self.status = "running"

    def pause(self) -> None:
        with self._lock:
            if self.status == "running":
                self.status = "paused"

    def set_direction(self, direction: str) -> bool:
        direction = direction.lower()
        if direction not in DIRECTIONS:
            raise ValueError(f"Unbekannte Richtung: {direction}")
        with self._lock:
            # Gegenüberliegende Richtungen würden direkt in den eigenen Hals führen.
            if direction == OPPOSITE[self.direction]:
                return False
            self._pending_direction = direction
            return True

    def tick(self) -> None:
        with self._lock:
            if self.status != "running":
                return
            self.direction = self._pending_direction
            dx, dy = DIRECTIONS[self.direction]
            head_x, head_y = self.snake[0]
            new_head = (head_x + dx, head_y + dy)
            eating = new_head == self.food

            outside = not (0 <= new_head[0] < self.config.width and 0 <= new_head[1] < self.config.height)
            occupied = self.snake if eating else self.snake[:-1]
            if outside or new_head in occupied:
                self.status = "game_over"
                return

            self.snake.insert(0, new_head)
            if eating:
                self.score += 1
                self._place_food()
            else:
                self.snake.pop()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self.status,
                "score": self.score,
                "width": self.config.width,
                "height": self.config.height,
                "tick_seconds": self.config.tick_seconds,
                "direction": self.direction,
                "snake": [[x, y] for x, y in self.snake],
                "food": list(self.food) if self.food is not None else None,
            }

    def ascii_snapshot(self) -> str:
        """Liefert dieselbe Spielfeldansicht wie die Konsole als Text."""
        state = self.snapshot()
        width, height = state["width"], state["height"]
        snake = {tuple(point) for point in state["snake"]}
        head = tuple(state["snake"][0])
        food = tuple(state["food"]) if state["food"] is not None else None
        lines = [
            f"SNAKE  Punkte: {state['score']}  Status: {state['status']}  Richtung: {state['direction']}  Tick: {state['tick_seconds']:.3f}s",
            "+" + "-" * width + "+",
        ]
        for y in range(height):
            cells = []
            for x in range(width):
                point = (x, y)
                cells.append("@" if point == head else "o" if point in snake else "*" if point == food else " ")
            lines.append("|" + "".join(cells) + "|")
        lines.append("+" + "-" * width + "+")
        return "\n".join(lines)

    def _place_food(self) -> None:
        free = [
            (x, y)
            for y in range(self.config.height)
            for x in range(self.config.width)
            if (x, y) not in self.snake
        ]
        if not free:
            self.food = None
            self.status = "won"
        else:
            self.food = self._random.choice(free)


class SnakeController:
    def __init__(self, game: SnakeGame, stop_event: threading.Event) -> None:
        self.game = game
        self.stop_event = stop_event

    def execute(self, request: str | dict[str, Any]) -> dict[str, Any]:
        try:
            if isinstance(request, str):
                command = request.strip().lower()
                payload: dict[str, Any] = {}
            elif isinstance(request, dict):
                payload = request
                command = str(payload.get("command", "")).lower()
            else:
                raise ValueError("Anfrage muss ein JSON-Objekt oder Textbefehl sein.")

            if command in DIRECTIONS:
                accepted = self.game.set_direction(command)
                return self._ok(direction_accepted=accepted)
            if command == "direction":
                direction = str(payload.get("direction", "")).lower()
                accepted = self.game.set_direction(direction)
                return self._ok(direction_accepted=accepted)
            if command == "start":
                self.game.start()
                return self._ok()
            if command == "pause":
                self.game.pause()
                return self._ok()
            if command == "reset":
                self.game.reset()
                return self._ok()
            if command == "state":
                return self._ok()
            if command == "screenshot":
                return {"ok": True, "screenshot": self.game.ascii_snapshot(), "state": self.game.snapshot()}
            if command == "help":
                return {"ok": True, "help": HELP, "state": self.game.snapshot()}
            if command == "quit":
                self.stop_event.set()
                return self._ok()
            raise ValueError(f"Unbekannter Befehl: {command or '<leer>'}")
        except (ValueError, TypeError) as exc:
            return {"ok": False, "error": str(exc), "state": self.game.snapshot()}

    def _ok(self, **extra: Any) -> dict[str, Any]:
        return {"ok": True, **extra, "state": self.game.snapshot()}


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        controller: SnakeController = self.server.controller  # type: ignore[attr-defined]
        for raw_line in self.rfile:
            text = raw_line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                request: str | dict[str, Any] = json.loads(text)
            except json.JSONDecodeError:
                request = text
            response = controller.execute(request)
            self.wfile.write((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
            self.wfile.flush()


class SnakeTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], controller: SnakeController) -> None:
        self.controller = controller
        super().__init__(address, _RequestHandler)


def _safe_addstr(window: Any, y: int, x: int, text: str, attributes: int = 0) -> None:
    try:
        window.addstr(y, x, text, attributes)
    except curses.error:
        pass


def _draw(screen: Any, state: dict[str, Any], host: str, port: int, show_help: bool) -> None:
    screen.erase()
    width, height = state["width"], state["height"]
    required_rows, required_cols = height + 6, width + 2
    rows, cols = screen.getmaxyx()
    if rows < required_rows or cols < required_cols:
        _safe_addstr(screen, 0, 0, f"Terminal zu klein: benötigt {required_cols}x{required_rows}, aktuell {cols}x{rows}")
        screen.refresh()
        return

    _safe_addstr(screen, 0, 0, f"SNAKE  Punkte: {state['score']}  Status: {state['status']}  TCP: {host}:{port}", curses.A_BOLD)
    _safe_addstr(screen, 1, 0, "+" + "-" * width + "+")
    snake = {tuple(point) for point in state["snake"]}
    head = tuple(state["snake"][0])
    food = tuple(state["food"]) if state["food"] is not None else None
    for y in range(height):
        cells = []
        for x in range(width):
            point = (x, y)
            cells.append("@" if point == head else "o" if point in snake else "*" if point == food else " ")
        _safe_addstr(screen, y + 2, 0, "|" + "".join(cells) + "|")
    _safe_addstr(screen, height + 2, 0, "+" + "-" * width + "+")
    _safe_addstr(screen, height + 3, 0, "Pfeile/WASD bewegen | Leertaste Start | p Pause | r Reset | ? Hilfe | q Ende")

    if show_help:
        lines = [
            "HILFE",
            "Pfeile oder WASD: steuern",
            "Leertaste/Enter: starten/weiter, p: Pause, r: Neustart, q: Ende",
            f"TCP auf {host}:{port}; pro Zeile JSON oder Text, z.B. {{\"command\":\"left\"}}",
            "Befehle: start pause reset up down left right state screenshot help quit",
            "Beliebige Taste schließt diese Hilfe.",
        ]
        box_width = min(max(map(len, lines)) + 4, cols - 2)
        top = max(1, (rows - len(lines) - 2) // 2)
        left = max(1, (cols - box_width) // 2)
        for index, line in enumerate(lines):
            _safe_addstr(screen, top + index, left, line[:box_width], curses.A_REVERSE)
    screen.refresh()


def _run_ui(screen: Any, game: SnakeGame, controller: SnakeController, host: str, port: int) -> None:
    curses.curs_set(0)
    screen.nodelay(True)
    screen.keypad(True)
    show_help = False
    last_tick = time.monotonic()
    key_directions = {
        curses.KEY_UP: "up", ord("w"): "up",
        curses.KEY_DOWN: "down", ord("s"): "down",
        curses.KEY_LEFT: "left", ord("a"): "left",
        curses.KEY_RIGHT: "right", ord("d"): "right",
    }

    while not controller.stop_event.is_set():
        state = game.snapshot()
        _draw(screen, state, host, port, show_help)
        key = screen.getch()
        if key != -1:
            if show_help:
                show_help = False
            elif key in key_directions:
                game.set_direction(key_directions[key])
            elif key in (ord(" "), 10, 13):
                game.start()
            elif key in (ord("p"), ord("P")):
                game.pause()
            elif key in (ord("r"), ord("R")):
                game.reset()
            elif key in (ord("?"), ord("h"), ord("H")):
                show_help = True
            elif key in (ord("q"), ord("Q"), 27):
                controller.stop_event.set()
                break

        now = time.monotonic()
        if now - last_tick >= game.config.tick_seconds:
            game.tick()
            last_tick = now
        time.sleep(0.01)


def main() -> int:
    host = "127.0.0.1"
    port = 8765
    try:
        game = SnakeGame(GameConfig())
        stop_event = threading.Event()
        controller = SnakeController(game, stop_event)
        server = SnakeTCPServer((host, port), controller)
    except (ValueError, OSError) as exc:
        print(f"Snake konnte nicht gestartet werden: {exc}")
        return 2

    actual_port = int(server.server_address[1])
    server_thread = threading.Thread(target=server.serve_forever, name="snake-tcp", daemon=True)
    server_thread.start()
    try:
        curses.wrapper(_run_ui, game, controller, host, actual_port)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
