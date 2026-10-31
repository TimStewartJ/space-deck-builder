"""End-to-end scalar-versus-Numba differential traces."""
from __future__ import annotations

import random

import pytest

from src.config import DataConfig
from src.vector_engine.differential import (
    make_supported_game,
    run_differential_trace,
)
from src.vector_engine.rules import compile_rules

# Every card whose ability timing decision 0016 changed: the OR abilities,
# the ships whose primary ability holds a choice, Machine Base and Central
# Office. Stealth Needle copies the ships' choices, and Ram's scrap ability
# removes a Blob card played this turn before Blob World counts them.
RULES_CARDS = (
    "Patrol Mech",
    "Trading Post",
    "Barter World",
    "Defense Center",
    "Blob World",
    "Recycling Station",
    "Trade Bot",
    "Missile Bot",
    "Supply Bot",
    "Battle Mech",
    "Battle Pod",
    "Missile Mech",
    "Machine Base",
    "Stealth Needle",
    "Central Office",
    "Ram",
)
OR_CARDS = RULES_CARDS[:6]
CHOICE_SHIPS = RULES_CARDS[6:12]


@pytest.fixture(scope="module")
def setup():
    data = DataConfig()
    cards = data.load_cards()
    registry = data.build_registry(cards)
    rules = compile_rules(cards, registry)
    return cards, registry, rules


def _rules_card_game(cards, registry, rules, seed):
    """A random game whose decks also hold one of every RULES_CARDS card.

    The trade deck's Stealth Needle leaves the game, so no player owns two:
    the Core Set has one, and a Stealth Needle copying another that copied
    nothing would open the same copy choice again forever. Random choices
    can scrap both decks down to nothing with Machine Base, so a lower turn
    cap ends such a stalemate.
    """
    game = make_supported_game(cards, registry, rules, seed)
    game.config.turn_cap = 300
    game.trade_deck = [
        card for card in game.trade_deck if card.name != "Stealth Needle"
    ]
    for slot, card in enumerate(game.trade_row):
        if card.name == "Stealth Needle":
            game.trade_row[slot] = game.trade_deck.pop()
    templates = {card.name: card for card in cards}
    for player in game.players:
        extra = [templates[name].clone() for name in RULES_CARDS]
        for card in extra:
            game._register_card_template(card)
            player.add_owned_card(card)
        player.deck.extend(extra)
        random.shuffle(player.deck)
    return game


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
        max_steps=1500,
    )
    assert result.steps >= 100
    assert result.action_counts["play"] > 0
    assert result.action_counts["buy"] > 0
    assert result.action_counts["end"] > 0
    assert result.action_counts["attack_player"] > 0
    assert result.games_completed == 4


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


def test_rule_timing_cards_match_after_every_decision(setup):
    """Every card whose rules decision 0016 changed, in many random games."""
    cards, registry, rules = setup
    games = [
        _rules_card_game(cards, registry, rules, seed)
        for seed in range(200, 216)
    ]
    result = run_differential_trace(games, rules, registry, max_steps=4000)
    assert result.games_completed == len(games)
    counts = result.card_action_counts

    def used(prefix, names):
        return sum(counts.get(f"{prefix}:{name}", 0) for name in names)

    for name in OR_CARDS:
        # Patrol Mech's options are chosen when it is played; the bases'
        # options are actions of their OR ability.
        assert used("effect", [name]) > 0, name
        assert used("effect_alt", [name]) > 0, name
    for name in ("Patrol Mech",) + CHOICE_SHIPS:
        assert used("play_choice", [name]) > 0, name
    assert used("copy_choice", ("Patrol Mech",) + CHOICE_SHIPS) > 0
    assert used("effect", ["Machine Base"]) > 0
    assert used("effect", ["Central Office"]) > 0
    assert used("scrap", ["Ram"]) > 0
    assert used("play_choice", ["Stealth Needle"]) > 0
    assert result.action_counts["skip"] > 0
    assert result.action_counts["scrap_discard"] > 0
