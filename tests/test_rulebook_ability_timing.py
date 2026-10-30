"""Abilities resolve when the rulebook says they do, in both engines.

Star Realms Box Set rulebook 1.1E, pages 4 and 5, and decision 0016
(docs/decisions/0016-rulebook-ability-timing.md):

- an OR ability gives exactly one of its options, and either option uses the
  ability for the turn.

Every step is applied to the scalar engine and to the vector engine; the
helper checks that both offer the same legal actions before the step and
reach the same canonical state after it.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.ai.random_agent import RandomAgent
from src.config import DataConfig
from src.encoding.action_context import build_action_context
from src.encoding.action_encoder import (
    END_TURN_INDEX,
    SKIP_INDEX,
    get_action_offsets,
    get_action_space_size,
)
from src.engine.game import Game
from src.vector_engine.differential import _capture_scalar_step, assert_masks_equal
from src.vector_engine.kernels import MAX_FORCED_DRAWS, VectorEngine
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
OFFSETS = get_action_offsets(REGISTRY.num_cards)
ACTION_DIM = get_action_space_size(REGISTRY.card_names)
TEMPLATES = {card.name: card for card in CARDS}


def _starter(name):
    return next(
        card
        for card in Game([], card_index_map=REGISTRY.card_index_map).create_starting_deck()
        if card.name == name
    )


class Pair:
    """One controlled scalar game and its vector twin, stepped together."""

    def __init__(self, hand=(), deck=12, bases=(), opponent_bases=(), row=()):
        game = Game([], card_index_map=REGISTRY.card_index_map)
        game.add_player("P0", RandomAgent("P0"))
        game.add_player("P1", RandomAgent("P1"))
        for player in game.players:
            game.stats.add_player(player.name)
        game.current_player = game.players[0]
        game.current_turn = 0
        game.first_player_name = "P0"
        game.is_running = True
        player = game.players[0]
        player.hand.extend(self._card(name) for name in hand)
        player.deck.extend(_starter("Viper") for _ in range(deck))
        player.bases.extend(self._card(name) for name in bases)
        game.players[1].bases.extend(self._card(name) for name in opponent_bases)
        game.trade_row.extend(self._card(name) for name in row)
        self.game = game
        self.player = player
        self.engine = VectorEngine(
            from_games([game], RULES, training_players=[0]), RULES
        )

    @staticmethod
    def _card(name):
        template = TEMPLATES.get(name)
        return template.clone() if template is not None else _starter(name)

    def context(self):
        context = build_action_context(
            self.game, self.game.current_player, REGISTRY.card_index_map, ACTION_DIM
        )
        assert_masks_equal(
            context.mask, self.engine.build_masks()[0], game_index=0, step=0
        )
        return context

    def legal(self) -> set[int]:
        return {int(index) for index in np.flatnonzero(self.context().mask)}

    def step(self, index: int) -> None:
        context = self.context()
        assert context.mask[index], index
        drawn, refills = _capture_scalar_step(self.game, context.resolvers[index])
        draws = np.full((1, MAX_FORCED_DRAWS), NO_CARD, np.int16)
        draws[0, : len(drawn)] = drawn
        refill = np.full((1, 8), NO_CARD, np.int16)
        refill[0, : len(refills)] = refills
        self.engine.step(
            np.asarray([index], np.int32),
            forced_draws=draws,
            forced_draw_count=np.asarray([len(drawn)], np.int8),
            forced_refill=refill,
        )
        assert_canonical_equal(
            canonical_scalar_snapshot(self.game, RULES),
            canonical_vector_snapshot(self.engine.state, 0),
        )

    def play(self, name: str) -> int:
        index = REGISTRY.card_index_map[name]
        self.step(OFFSETS.play_card + index)
        return index

    def resources(self) -> tuple[int, int, int, int]:
        player = self.player
        return player.trade, player.combat, player.health, len(player.hand)


def _first(index):
    return OFFSETS.effect_nonscrap + index


def _second(index):
    return OFFSETS.effect_alt + index


# (card, (trade, combat, authority, cards drawn) of option 1, ... of option 2)
OR_ABILITIES = [
    ("Trading Post", (0, 0, 1, 0), (1, 0, 0, 0)),
    ("Barter World", (0, 0, 2, 0), (2, 0, 0, 0)),
    ("Defense Center", (0, 0, 3, 0), (0, 2, 0, 0)),
    ("Blob World", (0, 5, 0, 0), (0, 0, 0, 1)),
    ("Recycling Station", (1, 0, 0, 0), None),
]


@pytest.mark.parametrize("option", [1, 2])
@pytest.mark.parametrize(("name", "first", "second"), OR_ABILITIES)
def test_base_or_ability_gives_exactly_one_option(name, first, second, option):
    pair = Pair(hand=(name, "Scout", "Scout"))
    card = pair.play(name)
    legal = pair.legal()
    assert {_first(card), _second(card)} <= legal
    before = pair.resources()

    pair.step(_first(card) if option == 1 else _second(card))
    expected = first if option == 1 else second
    if expected is None:
        # Recycling Station's second option discards up to two cards, then
        # draws that many: an optional choice, opened now.
        scout = REGISTRY.card_index_map["Scout"]
        assert pair.legal() == {OFFSETS.discard + scout, SKIP_INDEX}
        pair.step(OFFSETS.discard + scout)
        pair.step(SKIP_INDEX)
        expected = (0, 0, 0, 0)
    trade, combat, health, hand = pair.resources()
    assert (
        trade - before[0], combat - before[1], health - before[2], hand - before[3]
    ) == expected
    legal = pair.legal()
    assert _first(card) not in legal and _second(card) not in legal


@pytest.mark.parametrize(("name", "first", "second"), OR_ABILITIES)
def test_base_or_ability_is_offered_again_next_turn(name, first, second):
    pair = Pair(hand=(name,))
    card = pair.play(name)
    pair.step(_second(card))
    while pair.player.get_current_pending_set() is not None:
        pair.step(SKIP_INDEX)
    pair.step(END_TURN_INDEX)
    pair.step(END_TURN_INDEX)
    legal = pair.legal()
    assert {_first(card), _second(card)} <= legal
    pair.step(_first(card))
    legal = pair.legal()
    assert _first(card) not in legal and _second(card) not in legal


@pytest.mark.parametrize("option", [1, 2])
def test_patrol_mech_gives_exactly_one_option(option):
    pair = Pair(hand=("Patrol Mech",))
    card = pair.play("Patrol Mech")
    pair.step(_first(card) if option == 1 else _second(card))
    assert pair.resources()[:2] == ((3, 0) if option == 1 else (0, 5))
    legal = pair.legal()
    assert _first(card) not in legal and _second(card) not in legal


def test_two_copies_of_an_or_base_each_give_one_option():
    pair = Pair(hand=("Trading Post",), bases=("Trading Post",))
    card = pair.play("Trading Post")
    pair.step(_second(card))
    assert pair.legal() >= {_first(card), _second(card)}
    pair.step(_first(card))
    legal = pair.legal()
    assert _first(card) not in legal and _second(card) not in legal
    assert pair.resources()[:3] == (1, 0, 51)


def test_discard_option_whose_targets_run_out_offers_only_skip():
    """Recycling Station may discard two cards; with one in hand, SKIP ends it."""
    pair = Pair(hand=("Recycling Station", "Scout"))
    card = pair.play("Recycling Station")
    pair.step(_second(card))
    scout = REGISTRY.card_index_map["Scout"]
    pair.step(OFFSETS.discard + scout)
    assert pair.player.get_current_pending_set() is not None
    assert pair.legal() == {SKIP_INDEX}
    pair.step(SKIP_INDEX)
    assert len(pair.player.hand) == 1
