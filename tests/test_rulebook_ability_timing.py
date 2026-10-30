"""Abilities resolve when the rulebook says they do, in both engines.

Star Realms Box Set rulebook 1.1E, pages 4 and 5, and decision 0016
(docs/decisions/0016-rulebook-ability-timing.md):

- an OR ability gives exactly one of its options, and either option uses the
  ability for the turn;
- a ship's primary ability happens immediately as the ship is played, so its
  choices are pending decisions made before anything else; ally and scrap
  abilities may be used at any time during the main phase.

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

    def __init__(self, hand=(), deck=12, discard=(), bases=(), opponent_bases=(), row=()):
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
        player.discard_pile.extend(self._card(name) for name in discard)
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


def _index(name):
    return REGISTRY.card_index_map[name]


# Ships whose primary ability holds a choice: the resources the rest of the
# ability adds, and the choice's targets once played with a Scout in hand, a
# Viper in the discard pile, a Cutter in the trade row and an opposing
# Blob Wheel.
SHIP_CHOICES = [
    ("Trade Bot", (1, 0), {("scrap_hand", "Scout"), ("scrap_discard", "Viper")}),
    ("Missile Bot", (0, 2), {("scrap_hand", "Scout"), ("scrap_discard", "Viper")}),
    ("Supply Bot", (2, 0), {("scrap_hand", "Scout"), ("scrap_discard", "Viper")}),
    ("Battle Mech", (0, 4), {("scrap_hand", "Scout"), ("scrap_discard", "Viper")}),
    ("Battle Pod", (0, 4), {("scrap_trade", "Cutter")}),
    ("Missile Mech", (0, 6), {("destroy_base", "Blob Wheel")}),
]


def _choice_pair(name, *, targets=True):
    if not targets:
        return Pair(hand=(name,))
    return Pair(
        hand=(name, "Scout"),
        discard=("Viper",),
        row=("Cutter",),
        opponent_bases=("Blob Wheel",),
    )


def _targets(choices):
    return {getattr(OFFSETS, group) + _index(card) for group, card in choices}


@pytest.mark.parametrize(("name", "gains", "choices"), SHIP_CHOICES)
def test_ship_choice_is_pending_as_soon_as_the_ship_is_played(name, gains, choices):
    pair = _choice_pair(name)
    card = pair.play(name)
    assert pair.resources()[:2] == gains
    # Only the choice's targets and SKIP ("you may"): no card play, purchase,
    # attack, other ability or END_TURN until it is made.
    assert pair.legal() == _targets(choices) | {SKIP_INDEX}

    pair.step(min(_targets(choices)))
    assert pair.player.get_current_pending_set() is None
    legal = pair.legal()
    assert END_TURN_INDEX in legal
    assert _first(card) not in legal


@pytest.mark.parametrize(("name", "gains", "choices"), SHIP_CHOICES)
def test_ship_choice_may_be_skipped_and_is_then_used(name, gains, choices):
    pair = _choice_pair(name)
    card = pair.play(name)
    pair.step(SKIP_INDEX)
    assert pair.player.get_current_pending_set() is None
    assert _first(card) not in pair.legal()
    assert pair.resources()[:2] == gains


@pytest.mark.parametrize(("name", "gains", "choices"), SHIP_CHOICES)
def test_ship_choice_without_a_target_opens_no_decision(name, gains, choices):
    pair = _choice_pair(name, targets=False)
    card = pair.play(name)
    assert pair.player.get_current_pending_set() is None
    assert pair.resources()[:2] == gains
    legal = pair.legal()
    assert END_TURN_INDEX in legal
    assert _first(card) not in legal


def test_ship_ally_abilities_stay_usable_after_the_choice():
    pair = Pair(hand=("Trade Bot", "Missile Bot"), discard=("Viper",))
    trade_bot = pair.play("Trade Bot")
    pair.step(SKIP_INDEX)
    missile_bot = pair.play("Missile Bot")
    pair.step(SKIP_INDEX)
    # Machine Cult allies: each ship's "{Gain 2 Combat}" ally ability.
    assert {_first(trade_bot), _first(missile_bot)} <= pair.legal()
    pair.step(_first(missile_bot))
    pair.step(_first(trade_bot))
    assert pair.resources()[:2] == (1, 6)


@pytest.mark.parametrize("option", [1, 2])
def test_patrol_mech_option_is_chosen_as_it_is_played(option):
    pair = Pair(hand=("Patrol Mech", "Scout"))
    card = pair.play("Patrol Mech")
    # One option must be taken: no SKIP, nothing else.
    assert pair.legal() == {_first(card), _second(card)}
    pair.step(_first(card) if option == 1 else _second(card))
    assert pair.resources()[:2] == ((3, 0) if option == 1 else (0, 5))
    legal = pair.legal()
    assert _first(card) not in legal and _second(card) not in legal
    assert OFFSETS.play_card + _index("Scout") in legal


def test_patrol_mech_ally_ability_stays_usable_after_the_choice():
    pair = Pair(hand=("Patrol Mech", "Trade Bot", "Scout"), discard=("Viper",))
    patrol = pair.play("Patrol Mech")
    pair.step(_first(patrol))
    pair.play("Trade Bot")
    pair.step(SKIP_INDEX)
    # "{Machine Cult Ally}: Scrap a card in your hand or discard pile."
    assert _first(patrol) in pair.legal()
    assert _second(patrol) not in pair.legal()
    pair.step(_first(patrol))
    assert pair.player.get_current_pending_set() is not None


def test_stealth_needle_copy_of_a_ship_resolves_its_choice_at_once():
    pair = Pair(hand=("Trade Bot", "Stealth Needle", "Scout"), discard=("Viper",))
    trade_bot = pair.play("Trade Bot")
    pair.step(SKIP_INDEX)
    pair.play("Stealth Needle")
    pair.step(_first(trade_bot))
    # The copied primary ability gains its trade and opens its scrap now.
    assert pair.resources()[0] == 2
    assert pair.legal() == {
        OFFSETS.scrap_hand + _index("Scout"),
        OFFSETS.scrap_discard + _index("Viper"),
        SKIP_INDEX,
    }


def test_stealth_needle_copy_of_patrol_mech_chooses_an_option_at_once():
    pair = Pair(hand=("Patrol Mech", "Stealth Needle"))
    patrol = pair.play("Patrol Mech")
    pair.step(_first(patrol))
    needle = pair.play("Stealth Needle")
    pair.step(_first(patrol))
    # The needle now has Patrol Mech's OR ability; its own actions choose.
    assert pair.legal() == {_first(needle), _second(needle)}
    pair.step(_second(needle))
    assert pair.resources()[:2] == (3, 5)
    # The OR ability is used; the needle's effect_nonscrap action is now the
    # copied Machine Cult ally ability.
    assert _second(needle) not in pair.legal()
    pair.step(_first(needle))
    assert pair.player.get_current_pending_set() is None
    assert pair.resources()[:2] == (3, 5)


def test_each_copy_of_patrol_mech_opens_its_own_choice():
    pair = Pair(hand=("Patrol Mech", "Patrol Mech"))
    card = pair.play("Patrol Mech")
    pair.step(_second(card))
    pair.play("Patrol Mech")
    assert pair.legal() == {_first(card), _second(card)}
    pair.step(_first(card))
    assert pair.resources()[:2] == (3, 5)


def test_random_opponent_weights_both_options_of_a_pending_choice():
    pair = Pair(hand=("Patrol Mech",))
    card = pair.play("Patrol Mech")
    pair.engine.build_masks()
    weights = pair.engine.build_action_weights()[0]
    assert weights[_first(card)] == 1 and weights[_second(card)] == 1
    assert weights.sum() == 2


@pytest.mark.parametrize("style", ["balanced", "aggro", "economy", "scrapper", "fortress", "blobrush"])
def test_held_out_bots_choose_one_patrol_mech_option_when_played(style):
    from src.ai.held_out_agents import HeldOutAgent

    pair = Pair(hand=("Patrol Mech",), row=("Blob Wheel",))
    pair.play("Patrol Mech")
    game, player = pair.game, pair.player
    agent = HeldOutAgent("P0", style)
    action = agent.make_decision(game)
    pending = player.get_current_pending_set()
    assert any(action is option for option in pending.actions)
    patrol = player.played_cards[0]
    assert action.card_effect is agent._choose_or(
        game, player, game.players[1], patrol.effects[0]
    )


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
