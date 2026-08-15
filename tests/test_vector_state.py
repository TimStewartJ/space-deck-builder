"""Scalar Game to vector SoA conversion and canonical comparison tests."""
from __future__ import annotations

import random

import numpy as np
import pytest

from src.ai.random_agent import RandomAgent
from src.config import DataConfig
from src.engine.game import Game
from src.vector_engine.rules import compile_rules
from src.vector_engine.state import (
    ZONE_BASES,
    ZONE_PLAYED,
    assert_canonical_equal,
    canonical_scalar_snapshot,
    canonical_vector_snapshot,
    from_games,
)


@pytest.fixture(scope="module")
def setup():
    data = DataConfig()
    cards = data.load_cards()
    registry = data.build_registry(cards)
    rules = compile_rules(cards, registry)
    return cards, registry, rules


def make_game(cards, registry, seed=0):
    random.seed(seed)
    game = Game(cards, card_index_map=registry.card_index_map)
    game.add_player("A", RandomAgent("A"))
    game.add_player("B", RandomAgent("B"))
    game.start_game()
    return game


def test_started_game_converts_exactly(setup):
    cards, registry, rules = setup
    games = [make_game(cards, registry, seed) for seed in range(4)]
    state = from_games(games, rules, training_players=[0, 1, 0, 1])
    for index, game in enumerate(games):
        assert_canonical_equal(
            canonical_scalar_snapshot(game, rules),
            canonical_vector_snapshot(state, index),
        )
    assert state.training_player.tolist() == [0, 1, 0, 1]


def test_newly_played_base_is_in_played_and_bases(setup):
    cards, registry, rules = setup
    game = make_game(cards, registry)
    player = game.current_player
    base = next(
        card.clone()
        for card in cards
        if card.is_base() and rules.supported[card.index]
    )
    player.hand.append(base)
    assert player.play_card(base)

    state = from_games([game], rules)
    player_index = game.players.index(player)
    assert state.zones[0, player_index, ZONE_PLAYED, base.index] == 1
    assert state.zones[0, player_index, ZONE_BASES, base.index] == 1
    assert_canonical_equal(
        canonical_scalar_snapshot(game, rules),
        canonical_vector_snapshot(state, 0),
    )


def test_deck_order_is_intentionally_not_part_of_canonical_state(setup):
    cards, registry, rules = setup
    game = make_game(cards, registry)
    before = canonical_scalar_snapshot(game, rules)
    game.players[0].deck.reverse()
    after = canonical_scalar_snapshot(game, rules)
    assert np.array_equal(
        before["players"][0]["deck"],
        after["players"][0]["deck"],
    )


def test_canonical_comparison_reports_first_zone_difference(setup):
    cards, registry, rules = setup
    game = make_game(cards, registry)
    state = from_games([game], rules)
    vector = canonical_vector_snapshot(state, 0)
    vector["players"][0]["hand"][0] += 1
    with pytest.raises(
        AssertionError, match=r"player 0 hand diverged at \(0,\)"
    ):
        assert_canonical_equal(
            canonical_scalar_snapshot(game, rules),
            vector,
        )


def test_requires_two_players(setup):
    cards, registry, rules = setup
    game = Game(cards, card_index_map=registry.card_index_map)
    game.add_player("A", RandomAgent("A"))
    with pytest.raises(ValueError, match="exactly two players"):
        from_games([game], rules)
