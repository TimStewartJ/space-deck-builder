"""Gates for search roots written from public counts without copying games."""
from __future__ import annotations

import copy
import random
from dataclasses import fields

import numpy as np
import pytest

from src.ai.random_agent import RandomAgent
from src.config import DataConfig
from src.engine.actions import Action, ActionType, CardSource, get_available_actions
from src.engine.game import Game
from src.search.information_set import (
    InformationSetError,
    assert_public_knowledge_consistent,
    project_information_set,
)
from src.search.vector_determinization import known_hand_counts
from src.search.vector_root import SearchRootBuilder
from src.utils.logger import set_disabled
from src.vector_engine.rules import compile_rules
from src.vector_engine.state import VectorState, from_games

ARRAY_FIELDS = tuple(
    field.name for field in fields(VectorState) if field.name not in ("games", "num_cards")
)


@pytest.fixture(scope="module")
def card_data():
    set_disabled(True)
    data = DataConfig()
    cards = data.load_cards()
    registry = data.build_registry(cards)
    rules = compile_rules(cards, registry)
    return cards, registry, rules


def _make_game(cards, registry, seed: int) -> Game:
    random.seed(seed)
    game = Game(cards, card_names=registry.card_names, card_index_map=registry.card_index_map)
    game.add_player("P0", RandomAgent("P0"))
    game.add_player("P1", RandomAgent("P1"))
    game.start_game(first_player_name="P0" if seed % 2 == 0 else "P1")
    return game


def _assert_root_matches_projection(builder, rules, game) -> None:
    information = project_information_set(game)
    expected = from_games(
        [information.canonical_game], rules, training_players=[information.viewer_index]
    )
    state, roots = builder.build([game])
    for name in ARRAY_FIELDS:
        if name == "rng_state":
            continue
        assert np.array_equal(getattr(state, name), getattr(expected, name)), name
    root = roots[0]
    assert root.viewer == information.viewer_index
    assert np.array_equal(
        root.known_hand,
        known_hand_counts(information, 1 - information.viewer_index, rules.num_cards),
    )


def _hidden_variant(game: Game, seed: int) -> Game:
    memo = {id(player.agent): player.agent for player in game.players}
    variant = copy.deepcopy(game, memo)
    rng = random.Random(seed)
    viewer = variant.current_player
    top = len(viewer.ordered_top)
    unknown = viewer.deck[:-top] if top else list(viewer.deck)
    rng.shuffle(unknown)
    viewer.deck = [*unknown, *viewer.ordered_top]
    opponent = variant.get_opponent(viewer)
    known = {id(card) for card in opponent.public_known_hand}
    unknown_hand = [card for card in opponent.hand if id(card) not in known]
    top = len(opponent.ordered_top)
    pool = [*unknown_hand, *(opponent.deck[:-top] if top else opponent.deck)]
    rng.shuffle(pool)
    opponent.hand = [*opponent.public_known_hand, *pool[: len(unknown_hand)]]
    opponent.deck = [*pool[len(unknown_hand):], *opponent.ordered_top]
    rng.shuffle(variant.trade_deck)
    return variant


def test_roots_equal_canonical_projection_through_complete_games(card_data):
    cards, registry, rules = card_data
    builder = SearchRootBuilder(rules)
    compared = 0
    for seed in range(8):
        game = _make_game(cards, registry, seed)
        step = 0
        while not game.is_game_over and step < 5000:
            if step % 3 == 0:
                _assert_root_matches_projection(builder, rules, game)
                compared += 1
            game.apply_decision(game.current_player.make_decision(game))
            step += 1
        assert game.is_game_over
    assert compared > 300


def test_roots_equal_projection_with_known_hand_and_top_cards(card_data):
    cards, registry, rules = card_data
    builder = SearchRootBuilder(rules)
    game = _make_game(cards, registry, seed=14)
    opponent = game.get_opponent(game.current_player)
    acquired = game.trade_row.pop()
    game._remove_public_trade_card(acquired)
    opponent.add_owned_card(acquired)
    opponent.put_on_top(acquired)
    assert_public_knowledge_consistent(game)
    _assert_root_matches_projection(builder, rules, game)
    assert builder.build([game])[0].ordered_top_count.sum() == 1

    assert opponent.draw_card() is acquired
    assert opponent.public_known_hand == [acquired]
    _assert_root_matches_projection(builder, rules, game)
    root = builder.build([game])[1][0]
    assert root.known_hand[acquired.index] == 1


def test_roots_equal_projection_with_pending_copy_and_free_acquisition(card_data):
    cards, registry, rules = card_data
    builder = SearchRootBuilder(rules)
    game = _make_game(cards, registry, seed=142)
    explorer = game.explorer_pile[-1]
    game.current_player.add_pending_actions(
        [Action(
            ActionType.BUY_CARD,
            card=explorer,
            card_id=explorer.index,
            additional_params={"free": True, "destination": "top"},
        )],
        1,
        mandatory=True,
    )
    _assert_root_matches_projection(builder, rules, game)

    game = _make_game(cards, registry, seed=141)
    for name in ("Ram", "Stealth Needle"):
        for zone in (game.trade_row, game.trade_deck):
            match = next((card for card in zone if card.name == name), None)
            if match is not None:
                zone.remove(match)
                game._remove_public_trade_card(match)
                game.current_player.add_owned_card(match)
                game.current_player.hand.append(match)
                break
    for name in ("Ram", "Stealth Needle"):
        game.apply_decision(next(
            action
            for action in get_available_actions(game, game.current_player)
            if action.type == ActionType.PLAY_CARD and action.card.name == name
        ))
    assert game.current_player.get_current_pending_set() is not None
    _assert_root_matches_projection(builder, rules, game)


def test_roots_ignore_true_hidden_arrangement(card_data):
    cards, registry, rules = card_data
    builder = SearchRootBuilder(rules)
    for seed, steps in ((30, 120), (31, 200)):
        game = _make_game(cards, registry, seed)
        for _ in range(steps):
            if game.is_game_over:
                break
            game.apply_decision(game.current_player.make_decision(game))
        variant = _hidden_variant(game, seed=seed + 1000)
        assert_public_knowledge_consistent(variant)
        first_state, first_roots = builder.build([game])
        second_state, second_roots = builder.build([variant])
        assert first_roots[0].digest == second_roots[0].digest
        for name in ARRAY_FIELDS:
            assert np.array_equal(getattr(first_state, name), getattr(second_state, name)), name


def test_batched_roots_equal_individual_roots(card_data):
    cards, registry, rules = card_data
    builder = SearchRootBuilder(rules)
    games = []
    for seed in (40, 41, 42):
        game = _make_game(cards, registry, seed)
        for _ in range(60 + seed):
            game.apply_decision(game.current_player.make_decision(game))
        games.append(game)
    state, roots = builder.build(games)
    for row, game in enumerate(games):
        single_state, single_roots = builder.build([game])
        assert roots[row].digest == single_roots[0].digest
        for name in ARRAY_FIELDS:
            assert np.array_equal(getattr(state, name)[row], getattr(single_state, name)[0]), name


def test_digest_changes_with_public_information(card_data):
    cards, registry, rules = card_data
    builder = SearchRootBuilder(rules)
    game = _make_game(cards, registry, seed=43)
    before = builder.build([game])[1][0].digest
    game.current_player.trade += 1
    assert builder.build([game])[1][0].digest != before


def test_root_building_makes_no_deep_copies(card_data, monkeypatch):
    cards, registry, rules = card_data
    builder = SearchRootBuilder(rules)
    game = _make_game(cards, registry, seed=44)
    for _ in range(80):
        game.apply_decision(game.current_player.make_decision(game))

    def forbidden(*args, **kwargs):
        raise AssertionError("search roots must not deep copy games")

    monkeypatch.setattr(copy, "deepcopy", forbidden)
    builder.build([game])


def _rejects(builder, game, match):
    with pytest.raises(InformationSetError, match=match):
        project_information_set(game)
    with pytest.raises(InformationSetError, match=match):
        builder.build([game])


def test_roots_reject_games_without_public_ledgers(card_data):
    _, registry, rules = card_data
    game = Game([], card_index_map=registry.card_index_map)
    first = game.add_player("P0", RandomAgent("P0"))
    game.add_player("P1", RandomAgent("P1"))
    game.current_player = first
    with pytest.raises(InformationSetError):
        SearchRootBuilder(rules).build([game])


def test_roots_require_unambiguous_card_identity(card_data):
    cards, _, rules = card_data
    game = Game(cards)
    game.add_player("P0", RandomAgent("P0"))
    game.add_player("P1", RandomAgent("P1"))
    game.start_game(first_player_name="P0")
    _rejects(SearchRootBuilder(rules), game, "card_index_map")


def test_roots_reject_mutable_hidden_card_state(card_data):
    cards, registry, rules = card_data
    game = _make_game(cards, registry, seed=19)
    game.get_opponent(game.current_player).deck[0].effects[0].applied = True
    _rejects(SearchRootBuilder(rules), game, "outside public play")


def test_roots_reject_hidden_pending_references(card_data):
    cards, registry, rules = card_data
    game = _make_game(cards, registry, seed=18)
    opponent_card = game.get_opponent(game.current_player).hand[0]
    game.current_player.add_pending_actions(
        [Action(
            ActionType.DISCARD_CARDS,
            card=opponent_card,
            card_id=opponent_card.index,
            card_source=CardSource.OPPONENT,
        )],
        1,
        mandatory=True,
    )
    _rejects(SearchRootBuilder(rules), game, "hidden hand")


def test_roots_reject_ledgers_that_disagree_with_zone_sizes(card_data):
    cards, registry, rules = card_data
    game = _make_game(cards, registry, seed=45)
    opponent = game.get_opponent(game.current_player)
    opponent.deck.pop()
    with pytest.raises(InformationSetError, match="hidden hand and deck sizes"):
        SearchRootBuilder(rules).build([game])
