import threading

from src.games.gamer import _state_facts, apply_safety_condition, parse_compact_actions
from src.games.snake import GameConfig, SnakeController, SnakeGame


def test_game_moves_and_rejects_reverse_direction():
    game = SnakeGame(GameConfig(width=10, height=8), seed=1)
    original_head = game.snake[0]

    game.start()
    assert game.set_direction("left") is False
    game.tick()

    assert game.snake[0] == (original_head[0] + 1, original_head[1])
    assert game.snapshot()["status"] == "running"


def test_controller_supports_commands_and_errors():
    game = SnakeGame(GameConfig(width=10, height=8), seed=1)
    stop_event = threading.Event()
    controller = SnakeController(game, stop_event)

    assert controller.execute({"command": "start"})["state"]["status"] == "running"
    assert controller.execute("up")["direction_accepted"] is True
    screenshot = controller.execute("screenshot")
    assert screenshot["ok"] is True
    assert screenshot["screenshot_mode"] == "ascii"
    assert "@" in screenshot["screenshot"]
    assert "*" in screenshot["screenshot"]
    coordinates = controller.execute({"command": "screenshot", "mode": "coordinates"})
    assert coordinates["screenshot_mode"] == "coordinates"
    assert "head=(5, 4)" in coordinates["screenshot"]
    assert "body=[(4, 4), (3, 4)]" in coordinates["screenshot"]
    assert "food=" in coordinates["screenshot"]
    assert controller.execute({"command": "unknown"})["ok"] is False
    assert controller.execute("quit")["ok"] is True
    assert stop_event.is_set()


def test_compact_action_format_is_mapped():
    assert parse_compact_actions("u,01,l,03,p!") == [
        {"command": "up", "ticks": 1},
        {"command": "left", "ticks": 3},
        {"command": "screenshot", "ticks": 0},
    ]


def test_state_facts_warn_about_projected_wall_collision():
    game = SnakeGame(GameConfig(width=10, height=8, tick_seconds=0.5), seed=1)
    game.snake = [(8, 4), (7, 4), (6, 4)]
    game.food = (6, 1)

    facts = _state_facts(game.snapshot(), previous_inference_seconds=2.0)

    assert "Kopf=(8,4)" in facts
    assert "2 Felder links" in facts
    assert "3 Felder oben" in facts
    assert "KOLLISION" in facts
    assert "up" in facts and "down" in facts


def test_safety_condition_rebases_and_avoids_unsafe_plan_end():
    game = SnakeGame(GameConfig(width=10, height=8, tick_seconds=0.5), seed=1)
    game.snake = [(7, 4), (6, 4), (5, 4)]
    game.food = (1, 1)
    plan = [{"command": "right", "ticks": 5}]

    safe, notes = apply_safety_condition(plan, game.snapshot(), required_runway_ticks=4)

    assert safe[0] == {"command": "right", "ticks": 2}
    assert safe[-1]["command"] in {"up", "down"}
    assert any("Kollision" in note for note in notes)


def test_wall_collision_ends_game():
    game = SnakeGame(GameConfig(width=8, height=6), seed=1)
    game.start()

    for _ in range(10):
        game.tick()

    assert game.snapshot()["status"] == "game_over"
