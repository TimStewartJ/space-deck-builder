"""Targeted scalar/vector tests for every Gate 3 pending-state family."""
from __future__ import annotations

import numpy as np
import pytest

from src.ai.random_agent import RandomAgent
from src.cards.card import Card, CardType
from src.cards.factions import Faction
from src.config import DataConfig
from src.encoding.action_context import build_action_context
from src.encoding.action_encoder import (
    END_TURN_INDEX,
    get_action_space_size,
    get_action_space_layout,
)
from src.engine.game import Game
from src.engine.actions import ActionType, get_available_actions
from src.vector_engine.differential import (
    _capture_scalar_step,
    assert_masks_equal,
)
from src.vector_engine.kernels import (
    EFFECT_NONSCRAP_OFFSET,
    MAX_FORCED_DRAWS,
    VectorEngine,
)
from src.vector_engine.rules import compile_rules
from src.vector_engine.state import (
    NO_CARD,
    assert_canonical_equal,
    canonical_scalar_snapshot,
    canonical_vector_snapshot,
    from_games,
)


DATA = DataConfig()
CARDS = DATA.load_cards()
REGISTRY = DATA.build_registry(CARDS)
RULES = compile_rules(CARDS, REGISTRY)
LAYOUT = get_action_space_layout(REGISTRY.num_cards)
OFFSETS = {name: offset for name, offset, _ in LAYOUT["card_groups"]}
TEMPLATES = {card.name: card for card in CARDS}


def starter(name):
    game = Game(CARDS, card_index_map=REGISTRY.card_index_map)
    if name == "Explorer":
        game.setup_explorer_pile()
        return game.explorer_pile[-1]
    return next(c for c in game.create_starting_deck() if c.name == name)


def controlled(card_name):
    game = Game([], card_index_map=REGISTRY.card_index_map)
    game.add_player("P0", RandomAgent("P0"))
    game.add_player("P1", RandomAgent("P1"))
    for player in game.players:
        game.stats.add_player(player.name)
    game.current_player = game.players[0]
    game.current_turn = 0
    game.first_player_name = "P0"
    game.is_running = True
    card = TEMPLATES[card_name].clone()
    game.players[0].hand.append(card)
    game.players[0].deck.extend(starter("Viper").clone() for _ in range(12))
    return game, card


def make_engine(game):
    return VectorEngine(
        from_games([game], RULES, training_players=[0]), RULES
    )


def step_pair(game, engine, index):
    context = build_action_context(
        game,
        game.current_player,
        REGISTRY.card_index_map,
        get_action_space_size(REGISTRY.card_names),
    )
    assert_masks_equal(
        context.mask, engine.build_masks()[0], game_index=0, step=0
    )
    assert context.mask[index]
    drawn, refills = _capture_scalar_step(
        game, context.resolvers[index]
    )
    draw_tape = np.full((1, MAX_FORCED_DRAWS), NO_CARD, np.int16)
    draw_tape[0, : len(drawn)] = drawn
    refill_tape = np.full((1, 8), NO_CARD, np.int16)
    refill_tape[0, : len(refills)] = refills
    engine.step(
        np.asarray([index], np.int32),
        forced_draws=draw_tape,
        forced_draw_count=np.asarray([len(drawn)], np.int8),
        forced_refill=refill_tape,
    )
    assert_canonical_equal(
        canonical_scalar_snapshot(game, RULES),
        canonical_vector_snapshot(engine.state, 0),
    )


def test_trade_row_scrap_pending_refills_on_completion():
    game, card = controlled("Battle Pod")
    target = TEMPLATES["Blob Fighter"].clone()
    refill = TEMPLATES["Mothership"].clone()
    game.trade_row.append(target)
    game.trade_deck.append(refill)
    engine = make_engine(game)
    # Battle Pod's primary ability opens its trade-row scrap as it is played.
    step_pair(game, engine, OFFSETS["play_card"] + card.index)
    assert game.current_player.pending_action_sets
    step_pair(game, engine, OFFSETS["scrap_trade"] + target.index)
    assert not game.current_player.pending_action_sets
    assert game.trade_row[0].name == "Mothership"


def test_ally_effect_queues_destroy_then_trade_scrap():
    game, card = controlled("Blob Destroyer")
    ally = Card("Dummy Blob", card.index, 0, [], CardType.SHIP, faction=Faction.BLOB)
    game.current_player.played_cards.append(ally)
    base = TEMPLATES["Blob Wheel"].clone()
    game.players[1].bases.append(base)
    target = TEMPLATES["Trade Pod"].clone()
    game.trade_row.append(target)
    engine = make_engine(game)
    step_pair(game, engine, OFFSETS["play_card"] + card.index)
    step_pair(game, engine, EFFECT_NONSCRAP_OFFSET + card.index)
    assert len(game.current_player.pending_action_sets) == 2
    step_pair(game, engine, OFFSETS["destroy_base"] + base.index)
    step_pair(game, engine, OFFSETS["scrap_trade"] + target.index)
    assert not game.current_player.pending_action_sets


def test_deferred_discard_materializes_on_opponent_turn():
    game, card = controlled("Imperial Fighter")
    game.players[1].hand.append(starter("Scout").clone())
    engine = make_engine(game)
    step_pair(game, engine, OFFSETS["play_card"] + card.index)
    assert game.players[1].pending_start_of_turn_discards == 1
    step_pair(game, engine, END_TURN_INDEX)
    assert game.current_player is game.players[1]
    assert game.current_player.pending_action_sets
    target = game.current_player.hand[0]
    step_pair(game, engine, OFFSETS["discard"] + target.index)
    assert not game.current_player.pending_action_sets


def test_scrap_draw_completion_draws_resolved_count():
    game, card = controlled("Brain World")
    game.current_player.hand.extend(
        [starter("Scout").clone(), starter("Scout").clone()]
    )
    engine = make_engine(game)
    step_pair(game, engine, OFFSETS["play_card"] + card.index)
    step_pair(game, engine, EFFECT_NONSCRAP_OFFSET + card.index)
    for _ in range(2):
        target = game.current_player.hand[0]
        step_pair(game, engine, OFFSETS["scrap_hand"] + target.index)
    assert not game.current_player.pending_action_sets
    assert len(game.current_player.hand) == 2


def test_pending_source_snapshot_and_skip_preserve_resolved_draw():
    game, card = controlled("Brain World")
    hand_target = starter("Scout").clone()
    discard_target = starter("Viper").clone()
    game.current_player.hand.append(hand_target)
    game.current_player.discard_pile.append(discard_target)
    engine = make_engine(game)
    step_pair(game, engine, OFFSETS["play_card"] + card.index)
    step_pair(game, engine, EFFECT_NONSCRAP_OFFSET + card.index)
    step_pair(game, engine, OFFSETS["scrap_hand"] + hand_target.index)

    scalar_pending = game.current_player.pending_action_sets[0]
    assert {
        action.card_source.value for action in scalar_pending.actions
    } == {"discard"}
    assert engine.state.pending_sources[0, 0, 0] == 2

    # Skip the remaining discard target; completion draws exactly the one
    # card already scrapped.
    step_pair(game, engine, LAYOUT["global_indices"]["skip"])
    assert not game.current_player.pending_action_sets
    assert len(game.current_player.hand) == 1


@pytest.mark.parametrize("group", ["effect_nonscrap", "effect_alt"])
def test_or_option_applies_one_option_and_uses_the_ability(group):
    game, card = controlled("Blob World")
    ally = TEMPLATES["Trade Pod"].clone()
    for effect in ally.effects:
        effect.applied = True
    game.current_player.played_cards.append(ally)
    engine = make_engine(game)
    step_pair(game, engine, OFFSETS["play_card"] + card.index)
    player = game.current_player
    first = OFFSETS["effect_nonscrap"] + card.index
    second = OFFSETS["effect_alt"] + card.index
    context = build_action_context(
        game, player, REGISTRY.card_index_map,
        get_action_space_size(REGISTRY.card_names),
    )
    assert context.mask[first] and context.mask[second]
    hand = len(player.hand)

    step_pair(game, engine, OFFSETS[group] + card.index)
    if group == "effect_nonscrap":
        assert (player.combat, len(player.hand)) == (5, hand)
    else:
        # One card for each Blob card played this turn: Trade Pod and Blob World.
        assert (player.combat, len(player.hand)) == (0, hand + 2)
    context = build_action_context(
        game, player, REGISTRY.card_index_map,
        get_action_space_size(REGISTRY.card_names),
    )
    assert not context.mask[first] and not context.mask[second]
    masks = engine.build_masks()[0]
    assert not masks[first] and not masks[second]


def test_random_opponent_weights_each_or_option_of_each_copy():
    game, card = controlled("Trading Post")
    established = TEMPLATES["Trading Post"].clone()
    game.current_player.bases.append(established)
    engine = make_engine(game)
    step_pair(game, engine, OFFSETS["play_card"] + card.index)
    engine.build_masks()
    weights = engine.build_action_weights()[0]
    assert weights[OFFSETS["effect_nonscrap"] + card.index] == 2
    assert weights[OFFSETS["effect_alt"] + card.index] == 2

    available = [
        action
        for action in get_available_actions(game, game.current_player)
        if action.type == ActionType.APPLY_EFFECT
        and action.card_effect.option_index == 1
    ]
    assert [action.card for action in available] == [card, established]
    # Variant 1 of effect_alt is the established copy's second option.
    drawn, _ = _capture_scalar_step(game, available[1])
    assert drawn == []
    engine.step(
        np.asarray([OFFSETS["effect_alt"] + card.index], np.int32),
        action_variants=np.asarray([1], np.int16),
    )
    assert_canonical_equal(
        canonical_scalar_snapshot(game, RULES),
        canonical_vector_snapshot(engine.state, 0),
    )
    assert game.current_player.trade == 1
    assert established.effects[0].any_child_effects_used()
    assert not card.effects[0].any_child_effects_used()
