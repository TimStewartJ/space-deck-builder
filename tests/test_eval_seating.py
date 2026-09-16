from __future__ import annotations

from types import SimpleNamespace

from src.ai.random_agent import RandomAgent
from src.config import DataConfig
from src.engine.game import Game
from src.ppo.elo_tournament import _play_builtin_games
from src.ppo.ppo_eval import evaluate


def make_game() -> Game:
    data = DataConfig()
    cards = data.load_cards()
    registry = data.build_registry(cards)
    game = Game(cards, card_index_map=registry.card_index_map)
    game.add_player("A", RandomAgent("A"))
    game.add_player("B", RandomAgent("B"))
    return game


def test_game_can_select_each_first_player():
    first = make_game()
    first.start_game(first_player_name="A")
    assert first.current_player.name == "A"
    assert first.first_player_name == "A"
    assert len(first.players[0].hand) == 3
    assert len(first.players[1].hand) == 5

    second = make_game()
    second.start_game(first_player_name="B")
    assert second.current_player.name == "B"
    assert second.first_player_name == "B"
    assert len(second.players[0].hand) == 3
    assert len(second.players[1].hand) == 5


def test_game_rejects_unknown_first_player():
    game = make_game()
    try:
        game.start_game(first_player_name="missing")
    except ValueError as exc:
        assert "exactly one player" in str(exc)
    else:
        raise AssertionError("unknown first player was accepted")


def test_evaluate_records_balanced_seats(monkeypatch):
    captured = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            self.eval_seat_counts = {}

        def run_eval(self, num_games, *, alternate_seats, temperature):
            captured["alternate_seats"] = alternate_seats
            captured["temperature"] = temperature
            self.eval_seat_counts = {"first": 2, "second": 2}
            return 3, 1, 40

    monkeypatch.setattr("src.ppo.ppo_eval.BatchRunner", FakeRunner)
    result = evaluate(
        SimpleNamespace(),
        data_cfg=DataConfig(),
        device="cpu",
        opponents="random",
        eval_games=4,
        num_concurrent=4,
        num_workers=1,
    )

    assert captured["alternate_seats"] is True
    assert captured["temperature"] == 1.0
    assert result.per_opponent[0].first_player_games == 2
    assert result.per_opponent[0].second_player_games == 2


def test_builtin_elo_pairing_alternates_seats(monkeypatch):
    data = DataConfig()
    cards = data.load_cards()
    registry = data.build_registry(cards)
    starts = []
    original = Game.start_game

    def recording_start(game, first_player_name=None):
        starts.append(first_player_name)
        return original(game, first_player_name=first_player_name)

    monkeypatch.setattr(Game, "start_game", recording_start)
    wins_a, wins_b = _play_builtin_games(
        "random",
        "simple",
        cards,
        registry.card_index_map,
        4,
    )

    assert wins_a + wins_b == 4
    assert starts == ["Random_A", "Simple_B", "Random_A", "Simple_B"]
