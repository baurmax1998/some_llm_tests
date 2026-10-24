"""Echter Fähigkeitstest des lokalen Snake-LLMs an festen Positionen."""

from __future__ import annotations

import time

import pytest

from src.games.gamer import (
    DIRECTION_VECTORS,
    OPPOSITE_DIRECTION,
    PROMPT_PATH,
    GamerCommandDecoder,
    _advance_simulation,
    _turn_prompt,
)
from src.games.snake import GameConfig, SnakeGame


# Name, Richtung, Kopf, Futter, akzeptable erste Bewegungsabsicht.
CASES = [
    ("gerade nach rechts", "right", (4, 4), (8, 4), {"right", "wait"}),
    ("von rechts nach oben", "right", (4, 4), (4, 1), {"up"}),
    ("von rechts nach unten", "right", (4, 3), (4, 6), {"down"}),
    ("gerade nach links", "left", (5, 4), (1, 4), {"left", "wait"}),
    ("von links nach oben", "left", (5, 4), (5, 1), {"up"}),
    ("gerade nach oben", "up", (4, 5), (4, 1), {"up", "wait"}),
    ("von oben nach rechts", "up", (4, 5), (8, 5), {"right"}),
    ("gerade nach unten", "down", (4, 2), (4, 6), {"down", "wait"}),
    ("rechte Wand, Ziel hinten", "right", (9, 4), (1, 4), {"up", "down"}),
    ("obere Wand, Ziel hinten", "up", (4, 0), (4, 7), {"left", "right"}),
]


def _state(direction: str, head: tuple[int, int], food: tuple[int, int]) -> tuple[SnakeGame, dict]:
    game = SnakeGame(GameConfig(width=10, height=8, tick_seconds=1.0), seed=0)
    dx, dy = DIRECTION_VECTORS[direction]
    game.snake = [
        head,
        (head[0] - dx, head[1] - dy),
        (head[0] - 2 * dx, head[1] - 2 * dy),
    ]
    game.direction = direction
    game._pending_direction = direction
    game.food = food
    return game, game.snapshot()


def _simulate_collision_free(
    actions: list[dict], state: dict
) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    snake = [tuple(point) for point in state["snake"]]
    direction = state["direction"]
    food = tuple(state["food"])

    for action in actions:
        command = action["command"]
        if command == "screenshot":
            continue
        requested_direction = direction if command == "wait" else command
        assert requested_direction != OPPOSITE_DIRECTION[direction], (
            f"direkte Umkehr von {direction} nach {requested_direction}"
        )
        ticks = max(0, min(int(action["ticks"]), 50))
        if command in DIRECTION_VECTORS and ticks == 0:
            ticks = 1
        for _ in range(ticks):
            moved, food = _advance_simulation(
                snake, requested_direction, state["width"], state["height"], food
            )
            assert moved, f"Plan kollidiert bei {requested_direction}"
            direction = requested_direction
    return snake, food


@pytest.mark.integration
def test_llm_solves_ten_snake_positions(backend):
    """Ohne Zeitdruck müssen alle zehn Modellpläne sinnvoll und sicher sein."""
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    decoder = GamerCommandDecoder(backend, use_cache=False, output_format="compact")
    results = []
    failures = []

    for index, (name, direction, head, food, expected) in enumerate(CASES, start=1):
        game, state = _state(direction, head, food)
        prompt = _turn_prompt(
            index,
            game.coordinate_snapshot(),
            "coordinates",
            state,
            feedback=[],
            previous_inference_seconds=None,
        )
        started = time.perf_counter()
        actions, raw, _metrics = decoder.generate_actions(
            prompt,
            system_prompt,
            temperature=0.0,
            top_p=1.0,
            seed=index,
            max_tokens=64,
        )
        seconds = time.perf_counter() - started

        movement_actions = [a for a in actions if a["command"] != "screenshot"]
        assert movement_actions, f"{name}: keine Bewegungsaktion in {raw!r}"
        first_command = movement_actions[0]["command"]
        try:
            final_snake, remaining_food = _simulate_collision_free(actions, state)
            initial_distance = abs(head[0] - food[0]) + abs(head[1] - food[1])
            final_distance = abs(final_snake[0][0] - food[0]) + abs(final_snake[0][1] - food[1])
            if index <= 8:
                assert remaining_food is None or final_distance < initial_distance, (
                    f"kein Fortschritt zum Futter: Distanz {initial_distance} -> {final_distance}"
                )
            else:
                # Liegt das Futter genau hinter der Schlange, muss zunächst eine
                # sichere Seitwärtskurve statt der verbotenen Umkehr kommen.
                assert first_command in expected, (
                    f"erste Aktion {first_command!r}, erwartet {sorted(expected)}"
                )
            status = "OK"
        except AssertionError as exc:
            status = f"FEHLER: {exc}"
            failures.append(f"{name}: {exc}; Antwort={raw!r}")
        results.append((name, raw, first_command, seconds, status))

    print("\n\nSnake-LLM: 10 feste Positionen")
    for name, raw, first_command, seconds, status in results:
        print(f"- {name}: {raw} -> {first_command} ({seconds:.2f}s) [{status}]")
    assert not failures, "\n" + "\n".join(failures)
