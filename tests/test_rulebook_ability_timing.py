"""Abilities resolve when the rulebook says they do, in both engines.

Star Realms Box Set rulebook 1.1E, pages 4 and 5, and decision 0016
(docs/decisions/0016-rulebook-ability-timing.md):

- an OR ability gives exactly one of its options, and either option uses the
  ability for the turn;
- a ship's primary ability happens immediately as the ship is played, so its
  choices are pending decisions made before anything else; ally and scrap
  abilities may be used at any time during the main phase;
- Machine Base ("Draw a card, then scrap a card from your hand") is one
  activation that draws and then must scrap, also on the turn it is played;
- Blob World's draw option counts every Blob card played this turn, also one
  scrapped since;
- Central Office's top-deck is optional on every turn: its trade comes as it
  is played or first activated, and the top-deck is a separate activation.

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


@pytest.mark.parametrize("played_this_turn", [False, True])
def test_blob_world_draws_for_a_blob_card_scrapped_this_turn(played_this_turn):
    """Blob World draws "a card for each Blob card that you've played this turn".

    Ram is scrapped for its trade before Blob World's draw option; it was
    played this turn, so it counts. Blob World counts itself on the turn it is
    played.
    """
    if played_this_turn:
        pair = Pair(hand=("Blob World", "Ram", "Blob Fighter"))
        world = pair.play("Blob World")
    else:
        pair = Pair(hand=("Ram", "Blob Fighter"), bases=("Blob World",))
        world = _index("Blob World")
    ram = pair.play("Ram")
    pair.play("Blob Fighter")
    pair.step(OFFSETS.effect_scrap + ram)
    assert "Ram" not in [card.name for card in pair.player.played_cards]
    hand = len(pair.player.hand)
    pair.step(_second(world))
    assert len(pair.player.hand) == hand + (3 if played_this_turn else 2)


def test_blob_world_counts_a_stealth_needle_copy_of_a_blob_ship():
    """The copy gives the Needle the Blob faction: it is a Blob card played."""
    pair = Pair(hand=("Ram", "Stealth Needle"), bases=("Blob World",))
    ram = pair.play("Ram")
    needle = pair.play("Stealth Needle")
    pair.step(_first(ram))
    # The copied scrap ability scraps the Needle; it still counts.
    pair.step(OFFSETS.effect_scrap + needle)
    hand = len(pair.player.hand)
    pair.step(_second(_index("Blob World")))
    assert len(pair.player.hand) == hand + 2


def test_blob_world_count_starts_again_each_turn():
    pair = Pair(hand=("Ram",), bases=("Blob World",))
    pair.play("Ram")
    pair.step(END_TURN_INDEX)
    pair.step(END_TURN_INDEX)
    hand = len(pair.player.hand)
    pair.step(_second(_index("Blob World")))
    assert len(pair.player.hand) == hand


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


def _machine_base_activation(pair, base):
    """Activate Machine Base: it draws a Viper, then a card must be scrapped."""
    hand = len(pair.player.hand)
    pair.step(_first(base))
    assert len(pair.player.hand) == hand + 1
    viper, scout = _index("Viper"), _index("Scout")
    # Mandatory: no SKIP, nothing else until a card from the hand is scrapped.
    assert pair.legal() == {OFFSETS.scrap_hand + viper, OFFSETS.scrap_hand + scout}
    pair.step(OFFSETS.scrap_hand + viper)
    assert pair.player.get_current_pending_set() is None
    assert [card.name for card in pair.player.hand] == ["Scout"]
    assert all(card.name != "Viper" for card in pair.player.discard_pile)
    legal = pair.legal()
    assert _first(base) not in legal
    assert END_TURN_INDEX in legal


def test_machine_base_draws_nothing_when_played():
    """Its text, "Draw a card, then scrap a card from your hand", is one activation."""
    pair = Pair(hand=("Machine Base", "Scout"))
    base = pair.play("Machine Base")
    assert pair.player.get_current_pending_set() is None
    assert [card.name for card in pair.player.hand] == ["Scout"]
    legal = pair.legal()
    assert _first(base) in legal
    assert OFFSETS.play_card + _index("Scout") in legal
    assert END_TURN_INDEX in legal
    _machine_base_activation(pair, base)


def test_machine_base_is_the_same_activation_on_later_turns():
    pair = Pair(hand=("Scout",), bases=("Machine Base",))
    base = _index("Machine Base")
    _machine_base_activation(pair, base)
    pair.step(END_TURN_INDEX)
    pair.step(END_TURN_INDEX)
    assert _first(base) in pair.legal()


def test_machine_base_with_nothing_to_draw_or_scrap_opens_no_decision():
    pair = Pair(hand=("Machine Base",), deck=0)
    base = pair.play("Machine Base")
    pair.step(_first(base))
    assert pair.player.get_current_pending_set() is None
    assert pair.player.hand == []
    assert _first(base) not in pair.legal()


@pytest.mark.parametrize("top_deck", [False, True])
@pytest.mark.parametrize("played_this_turn", [True, False])
def test_central_office_top_deck_is_optional_on_every_turn(played_this_turn, top_deck):
    """Central Office: "You may put the next ship you acquire this turn on top".

    Playing it, or on a later turn activating it, gives its 2 Trade; the
    top-deck is a separate optional activation on both turns.
    """
    if played_this_turn:
        pair = Pair(hand=("Central Office",), row=("Cutter",))
        office = pair.play("Central Office")
    else:
        pair = Pair(bases=("Central Office",), row=("Cutter",))
        office = _index("Central Office")
        pair.step(_first(office))
    assert pair.resources()[0] == 2
    assert pair.player.next_ship_to_top == 0
    assert _first(office) in pair.legal()
    if top_deck:
        pair.step(_first(office))
        assert pair.resources()[0] == 2
        assert pair.player.next_ship_to_top == 1
        assert _first(office) not in pair.legal()
    cutter = pair.game.trade_row[0]
    pair.step(OFFSETS.buy_card + cutter.index)
    assert (pair.player.ordered_top == [cutter]) == top_deck
    assert (cutter in pair.player.discard_pile) != top_deck


@pytest.mark.parametrize("played_this_turn", [True, False])
def test_brain_world_scrap_is_one_activation_on_every_turn(played_this_turn):
    if played_this_turn:
        pair = Pair(hand=("Brain World", "Scout"))
        world = pair.play("Brain World")
    else:
        pair = Pair(hand=("Scout",), bases=("Brain World",))
        world = _index("Brain World")
    pair.step(_first(world))
    assert pair.player.get_current_pending_set() is not None
    pair.step(OFFSETS.scrap_hand + _index("Scout"))
    while pair.player.get_current_pending_set() is not None:
        pair.step(SKIP_INDEX)
    # One card scrapped, one drawn.
    assert [card.name for card in pair.player.hand] == ["Viper"]
    assert _first(world) not in pair.legal()
