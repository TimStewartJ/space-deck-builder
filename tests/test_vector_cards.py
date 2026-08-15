"""One scalar-versus-vector differential case for every supported card."""
from __future__ import annotations

import numpy as np
import pytest

from src.ai.random_agent import RandomAgent
from src.cards.card import Card, CardType
from src.cards.factions import Faction
from src.config import DataConfig
from src.encoding.action_context import build_action_context
from src.encoding.action_encoder import get_action_space_size
from src.encoding.action_encoder import END_TURN_INDEX
from src.engine.game import Game
from src.vector_engine.differential import (
    _capture_scalar_step,
    assert_masks_equal,
)
from src.vector_engine.kernels import (
    EFFECT_NONSCRAP_OFFSET,
    EFFECT_SCRAP_OFFSET,
    MAX_FORCED_DRAWS,
    PLAY_OFFSET,
    VectorEngine,
)
from src.vector_engine.rules import compile_rules
from src.vector_engine.state import (
    NO_CARD,
    ZONE_HAND,
    assert_canonical_equal,
    canonical_scalar_snapshot,
    canonical_vector_snapshot,
    from_games,
)


DATA = DataConfig()
CARDS = DATA.load_cards()
REGISTRY = DATA.build_registry(CARDS)
RULES = compile_rules(CARDS, REGISTRY)
TEMPLATES = {card.name: card for card in CARDS}


def _starter(name):
    game = Game(CARDS, card_index_map=REGISTRY.card_index_map)
    if name == "Explorer":
        game.setup_explorer_pile()
        return game.explorer_pile[-1]
    return next(
        card
        for card in game.create_starting_deck()
        if card.name == name
    )


def _template(name):
    return TEMPLATES.get(name) or _starter(name)


def _dummy_ally(card):
    faction = card.faction
    if not faction and card.ally_factions:
        faction = card.ally_factions
    if not faction:
        faction = Faction.BLOB
    return Card(
        name=f"Dummy ally for {card.name}",
        index=card.index,
        cost=0,
        effects=[],
        card_type=CardType.SHIP,
        faction=faction,
    )


def _controlled_pair(card_name):
    game = Game([], card_index_map=REGISTRY.card_index_map)
    game.add_player("P0", RandomAgent("P0"))
    game.add_player("P1", RandomAgent("P1"))
    for player in game.players:
        game.stats.add_player(player.name)
    game.current_turn = 0
    game.current_player = game.players[0]
    game.first_player_name = "P0"
    game.is_running = True

    card = _template(card_name).clone()
    player = game.current_player
    player.hand.append(card)
    scout = _starter("Scout")
    player.deck.extend(scout.clone() for _ in range(16))

    # Make every ally threshold eligible without introducing extra effects.
    if any(effect.faction_requirement for effect in card.effects):
        player.played_cards.append(_dummy_ally(card))

    # Exercise Embassy Yacht's deterministic two-base condition.
    if card_name == "Embassy Yacht":
        for _ in range(2):
            dummy = Card(
                "Dummy base",
                REGISTRY.card_index_map["Mech World"],
                0,
                [],
                CardType.BASE,
                defense=1,
            )
            player.bases.append(dummy)

    engine = VectorEngine(
        from_games([game], RULES, training_players=[0]),
        RULES,
    )
    return game, engine, card


def _step_pair(game, engine, action_index, action):
    context = build_action_context(
        game,
        game.current_player,
        REGISTRY.card_index_map,
        get_action_space_size(REGISTRY.card_names),
    )
    assert_masks_equal(
        context.mask,
        engine.build_masks()[0],
        game_index=0,
        step=0,
    )
    assert context.mask[action_index]

    drawn, refill = _capture_scalar_step(game, action)
    tape = np.full((1, MAX_FORCED_DRAWS), NO_CARD, dtype=np.int16)
    tape[0, : len(drawn)] = drawn
    engine.step(
        np.asarray([action_index], dtype=np.int32),
        forced_draws=tape,
        forced_draw_count=np.asarray([len(drawn)], dtype=np.int8),
        forced_refill=np.asarray([refill], dtype=np.int16),
    )
    assert_canonical_equal(
        canonical_scalar_snapshot(game, RULES),
        canonical_vector_snapshot(engine.state, 0),
    )


@pytest.mark.parametrize("card_name", RULES.supported_names)
def test_supported_card_matches_scalar_engine(card_name):
    game, engine, card = _controlled_pair(card_name)
    play_index = PLAY_OFFSET + card.index
    context = build_action_context(
        game,
        game.current_player,
        REGISTRY.card_index_map,
        get_action_space_size(REGISTRY.card_names),
    )
    _step_pair(game, engine, play_index, context.resolvers[play_index])

    # Consume every explicit ally/base/scrap action this card exposes.
    for _ in range(4):
        context = build_action_context(
            game,
            game.current_player,
            REGISTRY.card_index_map,
            get_action_space_size(REGISTRY.card_names),
        )
        candidates = (
            EFFECT_NONSCRAP_OFFSET + card.index,
            EFFECT_SCRAP_OFFSET + card.index,
        )
        action_index = next(
            (index for index in candidates if context.mask[index]),
            None,
        )
        if action_index is None:
            break
        _step_pair(
            game,
            engine,
            action_index,
            context.resolvers[action_index],
        )


def test_multiple_identical_ally_effects_are_consumed_one_copy_at_a_time():
    game, engine, card = _controlled_pair("Trade Pod")
    game.current_player.hand.append(card.clone())
    engine.state.zones[0, 0, ZONE_HAND, card.index] += 1

    for _ in range(2):
        context = build_action_context(
            game,
            game.current_player,
            REGISTRY.card_index_map,
            get_action_space_size(REGISTRY.card_names),
        )
        index = PLAY_OFFSET + card.index
        _step_pair(game, engine, index, context.resolvers[index])

    effect_index = EFFECT_NONSCRAP_OFFSET + card.index
    for expected_combat in (2, 4):
        context = build_action_context(
            game,
            game.current_player,
            REGISTRY.card_index_map,
            get_action_space_size(REGISTRY.card_names),
        )
        assert context.mask[effect_index]
        _step_pair(
            game, engine, effect_index, context.resolvers[effect_index]
        )
        assert game.current_player.combat == expected_combat
    context = build_action_context(
        game,
        game.current_player,
        REGISTRY.card_index_map,
        get_action_space_size(REGISTRY.card_names),
    )
    assert not context.mask[effect_index]
    assert not engine.build_masks()[0, effect_index]


def test_scrapping_one_duplicate_preserves_survivors_effects():
    game, engine, card = _controlled_pair("Ram")
    game.current_player.hand.append(card.clone())
    engine.state.zones[0, 0, ZONE_HAND, card.index] += 1

    for _ in range(2):
        context = build_action_context(
            game,
            game.current_player,
            REGISTRY.card_index_map,
            get_action_space_size(REGISTRY.card_names),
        )
        index = PLAY_OFFSET + card.index
        _step_pair(game, engine, index, context.resolvers[index])

    ally_index = EFFECT_NONSCRAP_OFFSET + card.index
    scrap_index = EFFECT_SCRAP_OFFSET + card.index
    context = build_action_context(
        game,
        game.current_player,
        REGISTRY.card_index_map,
        get_action_space_size(REGISTRY.card_names),
    )
    _step_pair(game, engine, ally_index, context.resolvers[ally_index])

    context = build_action_context(
        game,
        game.current_player,
        REGISTRY.card_index_map,
        get_action_space_size(REGISTRY.card_names),
    )
    _step_pair(game, engine, scrap_index, context.resolvers[scrap_index])

    # The surviving Ram still owns its ally bundle.
    context = build_action_context(
        game,
        game.current_player,
        REGISTRY.card_index_map,
        get_action_space_size(REGISTRY.card_names),
    )
    assert context.mask[ally_index]
    assert engine.build_masks()[0, ally_index]
    _step_pair(game, engine, ally_index, context.resolvers[ally_index])


def test_established_base_auto_effect_reactivates_next_turn():
    game, engine, card = _controlled_pair("Blob Wheel")
    context = build_action_context(
        game,
        game.current_player,
        REGISTRY.card_index_map,
        get_action_space_size(REGISTRY.card_names),
    )
    _step_pair(
        game,
        engine,
        PLAY_OFFSET + card.index,
        context.resolvers[PLAY_OFFSET + card.index],
    )
    assert game.current_player.combat == 1

    # End both players' turns so the base owner is current again.
    for _ in range(2):
        context = build_action_context(
            game,
            game.current_player,
            REGISTRY.card_index_map,
            get_action_space_size(REGISTRY.card_names),
        )
        _step_pair(
            game,
            engine,
            END_TURN_INDEX,
            context.resolvers[END_TURN_INDEX],
        )

    effect_index = EFFECT_NONSCRAP_OFFSET + card.index
    context = build_action_context(
        game,
        game.current_player,
        REGISTRY.card_index_map,
        get_action_space_size(REGISTRY.card_names),
    )
    assert context.mask[effect_index]
    _step_pair(
        game, engine, effect_index, context.resolvers[effect_index]
    )
    assert game.current_player.combat == 1


def test_new_duplicate_base_precedes_established_copy_until_turn_end():
    game, _, card = _controlled_pair("Blob Wheel")
    established = card.clone()
    game.current_player.bases.append(established)
    # Leave both established effects unapplied. Playing the new copy applies
    # only its own auto effect; scalar iteration then orders new before old.
    engine = VectorEngine(
        from_games([game], RULES, training_players=[0]),
        RULES,
    )

    context = build_action_context(
        game,
        game.current_player,
        REGISTRY.card_index_map,
        get_action_space_size(REGISTRY.card_names),
    )
    _step_pair(
        game,
        engine,
        PLAY_OFFSET + card.index,
        context.resolvers[PLAY_OFFSET + card.index],
    )

    scrap_index = EFFECT_SCRAP_OFFSET + card.index
    context = build_action_context(
        game,
        game.current_player,
        REGISTRY.card_index_map,
        get_action_space_size(REGISTRY.card_names),
    )
    _step_pair(game, engine, scrap_index, context.resolvers[scrap_index])

    # The established copy survived with its auto bundle still ready.
    effect_index = EFFECT_NONSCRAP_OFFSET + card.index
    context = build_action_context(
        game,
        game.current_player,
        REGISTRY.card_index_map,
        get_action_space_size(REGISTRY.card_names),
    )
    assert context.mask[effect_index]
    assert engine.build_masks()[0, effect_index]
    _step_pair(
        game, engine, effect_index, context.resolvers[effect_index]
    )


def test_turn_cap_winner_matches_scalar_get_winner():
    game, engine, _ = _controlled_pair("Scout")
    game.config.turn_cap = 0
    engine.state.turn_cap[0] = 0
    context = build_action_context(
        game,
        game.current_player,
        REGISTRY.card_index_map,
        get_action_space_size(REGISTRY.card_names),
    )
    _step_pair(
        game,
        engine,
        END_TURN_INDEX,
        context.resolvers[END_TURN_INDEX],
    )
    assert game.is_game_over
    assert game.stats.winner is None
    assert game.get_winner() == game.players[0].name
    assert engine.state.winner[0] == 0
