"""Physical Core Set rules that were formerly represented as no-ops."""
from __future__ import annotations

import pytest

from src.ai.agent import Agent
from src.cards.card import CardType
from src.cards.effects import CardEffectType, Effect
from src.cards.factions import Faction
from src.config import DataConfig
from src.encoding.action_context import build_action_context
from src.encoding.action_encoder import get_action_space_size
from src.engine.actions import Action, ActionType
from src.engine.game import Game


@pytest.fixture(scope="module")
def card_data():
    data = DataConfig()
    cards = data.load_cards()
    registry = data.build_registry(cards)
    templates = {card.name: card for card in cards}
    return data, registry, templates


def _game(registry):
    game = Game([], card_index_map=registry.card_index_map)
    player = game.add_player("P0", Agent("P0"))
    game.add_player("P1", Agent("P1"))
    for participant in game.players:
        game.stats.add_player(participant.name)
    game.current_player = player
    game.current_turn = 0
    game.first_player_name = player.name
    game.is_running = True
    return game, player


def _play(game, player, card):
    player.hand.append(card)
    game.execute_action(
        Action(ActionType.PLAY_CARD, card_id=card.index, card=card)
    )


def _context(game, player, registry):
    return build_action_context(
        game,
        player,
        registry.card_index_map,
        get_action_space_size(registry.card_names),
    )


def _effect_action(game, player, registry, card, *, scrap=False):
    return next(
        action
        for action in _context(game, player, registry).resolvers.values()
        if action.type == ActionType.APPLY_EFFECT
        and action.card is card
        and bool(action.card_effect and action.card_effect.is_scrap_effect)
        == scrap
    )


def test_blob_carrier_acquires_only_ships_free_on_top(card_data):
    _, registry, templates = card_data
    game, player = _game(registry)
    ally = templates["Trade Pod"].clone()
    carrier = templates["Blob Carrier"].clone()
    ship = templates["Battle Blob"].clone()
    base = templates["Blob World"].clone()
    game.trade_row = [ship, base]

    _play(game, player, ally)
    _play(game, player, carrier)
    trade_before = player.trade
    game.execute_action(_effect_action(game, player, registry, carrier))

    pending = player.get_current_pending_set()
    assert pending is not None and pending.mandatory
    assert {action.card for action in pending.actions} == {ship}

    game.execute_action(pending.actions[0])
    assert player.trade == trade_before
    assert player.ordered_top == [ship]
    assert player.deck[-1] is ship
    assert player.get_current_pending_set() is None
    assert player.draw_card() is ship


def test_brain_world_draws_once_per_card_actually_scrapped(card_data):
    _, registry, templates = card_data
    game, player = _game(registry)
    brain = templates["Brain World"].clone()
    hand_card = game.create_starting_deck()[0]
    discard_card = game.create_starting_deck()[1]
    draw_card = game.create_starting_deck()[2]
    player.hand = [brain, hand_card]
    player.discard_pile = [discard_card]
    player.deck = [draw_card]

    game.execute_action(
        Action(ActionType.PLAY_CARD, card_id=brain.index, card=brain)
    )
    assert player.get_current_pending_set() is None
    game.execute_action(_effect_action(game, player, registry, brain))

    pending = player.get_current_pending_set()
    assert pending is not None and not pending.mandatory
    game.execute_action(
        next(action for action in pending.actions if action.card is hand_card)
    )
    skip = next(
        action
        for action in _context(game, player, registry).resolvers.values()
        if action.type == ActionType.SKIP_DECISION
    )
    game.execute_action(skip)

    assert hand_card not in player.hand
    assert draw_card in player.hand
    assert player.get_current_pending_set() is None


def test_mech_world_is_wildcard_ally_not_every_faction(card_data):
    _, registry, templates = card_data
    game, player = _game(registry)
    mech = templates["Mech World"].clone()
    blob = templates["Ram"].clone()
    player.deck = [game.create_starting_deck()[0].clone() for _ in range(3)]
    _play(game, player, mech)
    _play(game, player, blob)

    assert mech.ally_factions == Faction.ALL
    assert player.get_faction_ally_count(Faction.BLOB) == 2
    assert mech.effective_faction == Faction.MACHINE_CULT

    before = len(player.hand)
    Effect(
        CardEffectType.DRAW_PER_FACTION,
        faction_target="Blob",
    ).apply(game, player)
    assert len(player.hand) == before + 1


def test_stealth_needle_copies_rules_faction_and_scraps_itself(card_data):
    _, registry, templates = card_data
    game, player = _game(registry)
    ram = templates["Ram"].clone()
    needle = templates["Stealth Needle"].clone()

    _play(game, player, ram)
    _play(game, player, needle)
    pending = player.get_current_pending_set()
    assert pending is not None
    game.execute_action(next(action for action in pending.actions if action.card is ram))

    assert player.combat == 10
    assert needle.effective_faction == (
        Faction.MACHINE_CULT | Faction.BLOB
    )
    game.execute_action(_effect_action(game, player, registry, needle))
    assert player.combat == 12
    game.execute_action(
        _effect_action(game, player, registry, needle, scrap=True)
    )
    assert player.trade == 3
    assert needle not in player.played_cards
    assert ram in player.played_cards


def test_fleet_hq_triggers_once_per_later_ship_and_stacks(card_data):
    _, registry, templates = card_data
    game, player = _game(registry)
    first = templates["Fleet HQ"].clone()
    second = templates["Fleet HQ"].clone()
    scout = game.create_starting_deck()[0]
    viper = game.create_starting_deck()[-1]

    _play(game, player, first)
    assert player.combat == 0
    _play(game, player, scout)
    assert player.combat == 1
    _play(game, player, second)
    assert player.combat == 1
    _play(game, player, viper)
    assert player.combat == 4


def test_central_office_token_ignores_bases_then_tops_ship(card_data):
    _, registry, templates = card_data
    game, player = _game(registry)
    office = templates["Central Office"].clone()
    base = templates["Blob Wheel"].clone()
    ship = templates["Trade Pod"].clone()
    game.trade_row = [base, ship]

    _play(game, player, office)
    assert player.trade == 2
    game.execute_action(_effect_action(game, player, registry, office))
    assert player.next_ship_to_top == 1
    player.trade = 20
    game.execute_action(
        Action(ActionType.BUY_CARD, card_id=base.index, card=base)
    )
    assert player.next_ship_to_top == 1
    assert base in player.discard_pile
    game.execute_action(
        Action(ActionType.BUY_CARD, card_id=ship.index, card=ship)
    )
    assert player.next_ship_to_top == 0
    assert player.ordered_top == [ship]


def test_freighter_ally_grants_one_optional_top_token(card_data):
    _, registry, templates = card_data
    game, player = _game(registry)
    ally = templates["Federation Shuttle"].clone()
    freighter = templates["Freighter"].clone()

    _play(game, player, ally)
    _play(game, player, freighter)
    assert player.trade == 6
    game.execute_action(_effect_action(game, player, registry, freighter))
    assert player.next_ship_to_top == 1
