import threading

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
    assert "@" in screenshot["screenshot"]
    assert "*" in screenshot["screenshot"]
    assert controller.execute({"command": "unknown"})["ok"] is False
    assert controller.execute("quit")["ok"] is True
    assert stop_event.is_set()


def test_wall_collision_ends_game():
    game = SnakeGame(GameConfig(width=8, height=6), seed=1)
    game.start()

    for _ in range(10):
        game.tick()

    assert game.snapshot()["status"] == "game_over"
