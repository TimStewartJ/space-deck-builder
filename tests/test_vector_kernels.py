"""Rules-correct Gate 2 Numba kernel tests."""
from __future__ import annotations

import random

import numpy as np
import pytest

from src.ai.random_agent import RandomAgent
from src.config import DataConfig
from src.encoding.action_context import build_action_context
from src.encoding.action_encoder import END_TURN_INDEX
from src.encoding.state_encoder import encode_state
from src.engine.game import Game
from src.vector_engine.kernels import (
    ACTION_DIM,
    ATTACK_BASE_OFFSET,
    ATTACK_PLAYER_INDEX,
    BUY_OFFSET,
    EFFECT_NONSCRAP_OFFSET,
    EFFECT_SCRAP_OFFSET,
    MAX_FORCED_DRAWS,
    PLAY_OFFSET,
    VectorEngine,
)
from src.vector_engine.rules import EffectTrigger, compile_rules
from src.vector_engine.state import (
    INSTANCE_ACTIVE,
    VectorState,
    ZONE_BASES,
    ZONE_DECK,
    ZONE_DISCARD,
    ZONE_HAND,
    ZONE_PLAYED,
    from_games,
)


@pytest.fixture(scope="module")
def setup():
    data = DataConfig()
    cards = data.load_cards()
    registry = data.build_registry(cards)
    rules = compile_rules(cards, registry)
    return cards, registry, rules


def empty_engine(rules, games=1):
    state = VectorState.empty(games, rules.num_cards)
    state.health[:] = 50
    state.hand_size[:] = 5
    return VectorEngine(state, rules)


def _bundle(rules, card, trigger):
    return next(
        bundle
        for bundle in range(int(rules.bundle_count[card]))
        if rules.bundle_trigger[card, bundle] == int(trigger)
    )


def test_mask_uses_real_action_layout_and_outpost_priority(setup):
    _, registry, rules = setup
    engine = empty_engine(rules)
    state = engine.state
    scout = registry.card_index_map["Scout"]
    trade_pod = registry.card_index_map["Trade Pod"]
    outpost = registry.card_index_map["Space Station"]
    base = registry.card_index_map["Blob Wheel"]

    state.zones[0, 0, ZONE_HAND, scout] = 1
    state.trade[0, 0] = 3
    state.trade_row[0, 0] = trade_pod
    state.combat[0, 0] = 10
    state.zones[0, 1, ZONE_BASES, outpost] = 1
    state.zones[0, 1, ZONE_BASES, base] = 1

    mask = engine.build_masks()[0]
    assert mask[PLAY_OFFSET + scout]
    assert mask[BUY_OFFSET + trade_pod]
    assert mask[ATTACK_BASE_OFFSET + outpost]
    assert not mask[ATTACK_BASE_OFFSET + base]
    assert not mask[ATTACK_PLAYER_INDEX]
    assert mask[END_TURN_INDEX]
    assert mask.shape == (ACTION_DIM,)


def test_play_applies_auto_bundle(setup):
    _, registry, rules = setup
    engine = empty_engine(rules)
    scout = registry.card_index_map["Scout"]
    engine.state.zones[0, 0, ZONE_HAND, scout] = 1

    engine.step(np.asarray([PLAY_OFFSET + scout], dtype=np.int32))

    assert engine.state.zones[0, 0, ZONE_HAND, scout] == 0
    assert engine.state.zones[0, 0, ZONE_PLAYED, scout] == 1
    assert engine.state.trade[0, 0] == 1


def test_parent_auto_bundle_and_forced_draw(setup):
    _, registry, rules = setup
    engine = empty_engine(rules)
    mothership = registry.card_index_map["Mothership"]
    scout = registry.card_index_map["Scout"]
    engine.state.zones[0, 0, ZONE_HAND, mothership] = 1
    engine.state.zones[0, 0, ZONE_DECK, scout] = 1
    draws = np.full((1, MAX_FORCED_DRAWS), -1, dtype=np.int16)
    draws[0, 0] = scout

    engine.step(
        np.asarray([PLAY_OFFSET + mothership], dtype=np.int32),
        forced_draws=draws,
        forced_draw_count=np.asarray([1], dtype=np.int8),
    )

    assert engine.state.combat[0, 0] == 6
    assert engine.state.zones[0, 0, ZONE_HAND, scout] == 1
    assert engine.state.zones[0, 0, ZONE_DECK, scout] == 0


def test_ally_bundle_is_masked_and_applied(setup):
    _, registry, rules = setup
    engine = empty_engine(rules)
    trade_pod = registry.card_index_map["Trade Pod"]
    blob_fighter = registry.card_index_map["Blob Fighter"]
    ally_bundle = _bundle(rules, trade_pod, EffectTrigger.ALLY)
    engine.state.zones[0, 0, ZONE_PLAYED, trade_pod] = 1
    engine.state.effect_state[0, 0, trade_pod, 0] = (
        int(INSTANCE_ACTIVE) | (1 << ally_bundle)
    )

    assert not engine.build_masks()[0, EFFECT_NONSCRAP_OFFSET + trade_pod]

    engine.state.zones[0, 0, ZONE_PLAYED, blob_fighter] = 1
    assert engine.build_masks()[0, EFFECT_NONSCRAP_OFFSET + trade_pod]
    engine.step(
        np.asarray(
            [EFFECT_NONSCRAP_OFFSET + trade_pod], dtype=np.int32
        )
    )
    assert engine.state.combat[0, 0] == 2
    assert (
        engine.state.effect_state[0, 0, trade_pod, 0]
        == int(INSTANCE_ACTIVE)
    )


def test_scrap_bundle_removes_card_and_returns_explorer(setup):
    _, registry, rules = setup
    engine = empty_engine(rules)
    explorer = registry.card_index_map["Explorer"]
    scrap_bundle = _bundle(rules, explorer, EffectTrigger.SCRAP)
    engine.state.zones[0, 0, ZONE_PLAYED, explorer] = 1
    engine.state.effect_state[0, 0, explorer, 0] = (
        int(INSTANCE_ACTIVE) | (1 << scrap_bundle)
    )

    assert engine.build_masks()[0, EFFECT_SCRAP_OFFSET + explorer]
    engine.step(
        np.asarray([EFFECT_SCRAP_OFFSET + explorer], dtype=np.int32)
    )

    assert engine.state.combat[0, 0] == 2
    assert engine.state.zones[0, 0, ZONE_PLAYED, explorer] == 0
    assert engine.state.explorer_count[0] == 1


def test_buy_uses_forced_market_refill(setup):
    _, registry, rules = setup
    engine = empty_engine(rules)
    trade_pod = registry.card_index_map["Trade Pod"]
    blob_fighter = registry.card_index_map["Blob Fighter"]
    engine.state.trade[0, 0] = 5
    engine.state.trade_row[0, 0] = trade_pod
    engine.state.trade_deck[0, blob_fighter] = 1

    engine.step(
        np.asarray([BUY_OFFSET + trade_pod], dtype=np.int32),
        forced_refill=np.asarray([blob_fighter], dtype=np.int16),
    )

    assert engine.state.trade[0, 0] == 3
    assert engine.state.zones[0, 0, ZONE_DISCARD, trade_pod] == 1
    assert engine.state.trade_row[0, 0] == blob_fighter
    assert engine.state.trade_deck[0, blob_fighter] == 0


def test_attack_base_then_player(setup):
    _, registry, rules = setup
    engine = empty_engine(rules)
    outpost = registry.card_index_map["Space Station"]
    engine.state.zones[0, 1, ZONE_BASES, outpost] = 1
    engine.state.combat[0, 0] = 7

    engine.step(
        np.asarray([ATTACK_BASE_OFFSET + outpost], dtype=np.int32)
    )
    assert engine.state.zones[0, 1, ZONE_BASES, outpost] == 0
    assert engine.state.zones[0, 1, ZONE_DISCARD, outpost] == 1
    assert engine.state.combat[0, 0] == 3

    engine.step(np.asarray([ATTACK_PLAYER_INDEX], dtype=np.int32))
    assert engine.state.health[0, 1] == 47
    assert engine.state.combat[0, 0] == 0


def test_end_turn_discards_ships_preserves_bases_and_draws(setup):
    _, registry, rules = setup
    engine = empty_engine(rules)
    scout = registry.card_index_map["Scout"]
    base = registry.card_index_map["Blob Wheel"]
    viper = registry.card_index_map["Viper"]
    engine.state.zones[0, 0, ZONE_HAND, scout] = 1
    engine.state.zones[0, 0, ZONE_PLAYED, scout] = 1
    engine.state.zones[0, 0, ZONE_PLAYED, base] = 1
    engine.state.zones[0, 0, ZONE_BASES, base] = 1
    engine.state.zones[0, 0, ZONE_DECK, viper] = 5
    draws = np.full((1, MAX_FORCED_DRAWS), -1, dtype=np.int16)
    draws[0, :5] = viper

    engine.step(
        np.asarray([END_TURN_INDEX], dtype=np.int32),
        forced_draws=draws,
        forced_draw_count=np.asarray([5], dtype=np.int8),
    )

    assert engine.state.zones[0, 0, ZONE_DISCARD, scout] == 2
    assert engine.state.zones[0, 0, ZONE_BASES, base] == 1
    assert engine.state.zones[0, 0, ZONE_PLAYED].sum() == 0
    assert engine.state.zones[0, 0, ZONE_HAND, viper] == 5
    assert engine.state.current_player[0] == 1
    assert engine.state.turns[0] == 1


def test_initial_scalar_mask_and_observation_match(setup):
    cards, registry, rules = setup
    supported_cards = [card for card in cards if rules.supported[card.index]]
    random.seed(5)
    game = Game(supported_cards, card_index_map=registry.card_index_map)
    game.add_player("A", RandomAgent("A"))
    game.add_player("B", RandomAgent("B"))
    game.start_game()
    current = game.players.index(game.current_player)
    state = from_games([game], rules, training_players=[current])
    engine = VectorEngine(state, rules)

    vector_mask = engine.build_masks()[0]
    context = build_action_context(
        game,
        game.current_player,
        registry.card_index_map,
        ACTION_DIM,
    )
    assert np.array_equal(vector_mask, context.mask)

    vector_obs = engine.materialize_observations()[0]
    scalar_obs = encode_state(
        game,
        is_current_player_training=True,
        cards=registry.card_names,
        card_index_map=registry.card_index_map,
        can_buy=context.can_buy,
        has_actions=context.has_meaningful,
        return_numpy=True,
    )
    assert np.array_equal(vector_obs, scalar_obs)
