"""Tests for train CLI defaults that define the baseline curriculum."""
import argparse

from src.__main__ import _build_train_parser
from src.config import RunConfig


def _parse_train_args(argv: list[str]):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    _build_train_parser(sub)
    return parser.parse_args(["train", *argv])


def test_run_config_default_curriculum_uses_fixed_opponents():
    cfg = RunConfig()

    assert cfg.self_play is False
    assert cfg.opponents == "random,heuristic,simple"


def test_train_cli_defaults_match_run_config_curriculum():
    args = _parse_train_args([])
    cfg = RunConfig()

    assert args.self_play == cfg.self_play
    assert args.opponents == cfg.opponents


def test_train_cli_self_play_keeps_fixed_opponent_pool():
    args = _parse_train_args(["--self-play"])

    assert args.self_play is True
    assert args.opponents == "random,heuristic,simple"


def test_train_cli_can_reproduce_legacy_random_only_training():
    args = _parse_train_args(["--opponents", "random"])

    assert args.self_play is False
    assert args.opponents == "random"
