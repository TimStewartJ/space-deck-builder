"""End-to-end scalar-versus-Numba differential traces."""
from __future__ import annotations

import pytest

from src.config import DataConfig
from src.vector_engine.differential import (
    make_supported_game,
    run_differential_trace,
)
from src.vector_engine.rules import compile_rules


@pytest.fixture(scope="module")
def setup():
    data = DataConfig()
    cards = data.load_cards()
    registry = data.build_registry(cards)
    rules = compile_rules(cards, registry)
    return cards, registry, rules


def test_four_games_match_after_every_decision(setup):
    cards, registry, rules = setup
    games = [
        make_supported_game(cards, registry, rules, seed)
        for seed in range(4)
    ]
    result = run_differential_trace(
        games,
        rules,
        registry,
        max_steps=250,
    )
    assert result.steps >= 100
    assert result.action_counts["play"] > 0
    assert result.action_counts["buy"] > 0
    assert result.action_counts["end"] > 0
    assert result.action_counts["attack_player"] > 0


def test_same_seed_has_reproducible_trace_summary(setup):
    cards, registry, rules = setup
    summaries = []
    for _ in range(2):
        games = [
            make_supported_game(cards, registry, rules, seed)
            for seed in (101, 102)
        ]
        summaries.append(
            run_differential_trace(
                games, rules, registry, max_steps=100
            )
        )
    assert summaries[0] == summaries[1]
