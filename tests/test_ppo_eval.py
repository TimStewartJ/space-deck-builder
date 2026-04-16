"""Tests for the standalone evaluation module (ppo_eval)."""
import pytest
from unittest.mock import patch, MagicMock
from src.ppo.ppo_eval import _distribute_games, EvalResult, OpponentResult


class TestDistributeGames:
    """Verify game distribution across opponent types."""

    def test_even_split(self):
        assert _distribute_games(100, 2) == [50, 50]

    def test_remainder_spread(self):
        result = _distribute_games(100, 3)
        assert sum(result) == 100
        assert result == [34, 33, 33]

    def test_single_bucket(self):
        assert _distribute_games(100, 1) == [100]

    def test_more_buckets_than_games(self):
        result = _distribute_games(2, 5)
        assert sum(result) == 2
        assert result == [1, 1, 0, 0, 0]

    def test_zero_games(self):
        assert _distribute_games(0, 3) == [0, 0, 0]

    def test_min_per_bucket_enforced(self):
        result = _distribute_games(1, 3, min_per_bucket=1)
        assert result == [1, 1, 1]

    def test_min_per_bucket_no_effect_when_already_sufficient(self):
        result = _distribute_games(100, 2, min_per_bucket=1)
        assert result == [50, 50]


class TestEvalCLIParsing:
    """Verify the eval subcommand CLI argument parsing."""

    def test_eval_parser_defaults(self):
        from src.__main__ import main
        import argparse

        # Build parser the same way main() does
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        from src.__main__ import _build_eval_parser
        _build_eval_parser(sub)

        args = parser.parse_args(["eval"])
        assert args.command == "eval"
        assert args.model is None
        assert args.opponents == "random,heuristic,simple"
        assert args.games > 0

    def test_eval_parser_custom_args(self):
        import argparse
        from src.__main__ import _build_eval_parser

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        _build_eval_parser(sub)

        args = parser.parse_args([
            "eval",
            "--opponents", "random,heuristic",
            "--games", "50",
            "--simulation-device", "cpu",
            "--num-workers", "2",
        ])
        assert args.opponents == "random,heuristic"
        assert args.games == 50
        assert args.simulation_device == "cpu"
        assert args.num_workers == 2


class TestEvalResultDataclasses:
    """Verify result dataclasses hold expected data."""

    def test_opponent_result(self):
        r = OpponentResult(opponent="random", wins=80, games=100,
                           win_rate=0.8, avg_steps=42.5)
        assert r.opponent == "random"
        assert r.win_rate == 0.8

    def test_eval_result(self):
        opp = OpponentResult("random", 80, 100, 0.8, 42.5)
        result = EvalResult(
            per_opponent=[opp],
            total_wins=80,
            total_games=100,
            overall_win_rate=0.8,
            elapsed_seconds=1.5,
        )
        assert len(result.per_opponent) == 1
        assert result.total_wins == 80
