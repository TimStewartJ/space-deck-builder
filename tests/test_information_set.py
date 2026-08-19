"""Information-safe hidden-state determinization correctness gates."""
from __future__ import annotations

import copy
import random
from collections import Counter

import numpy as np
import pytest

from src.ai.random_agent import RandomAgent
from src.config import DataConfig
from src.encoding.action_context import build_action_context
from src.encoding.action_encoder import get_action_space_size
from src.encoding.state_encoder import encode_state
from src.engine.actions import (
    Action,
    ActionType,
    CardSource,
    get_available_actions,
)
from src.engine.game import Game
from src.search import (
    InformationSetError,
    assert_public_knowledge_consistent,
    determinize,
    project_information_set,
)
from src.utils.logger import set_disabled
from src.vector_engine.kernels import VectorEngine
from src.vector_engine.differential import run_differential_trace
from src.vector_engine.rules import compile_rules
from src.vector_engine.state import (
    assert_canonical_equal,
    canonical_scalar_snapshot,
    canonical_vector_snapshot,
    from_games,
)


@pytest.fixture(scope="module")
def card_data():
    set_disabled(True)
    data = DataConfig()
    cards = data.load_cards()
    registry = data.build_registry(cards)
    rules = compile_rules(cards, registry)
    return cards, registry, rules


def _make_game(cards, registry, seed: int = 0) -> Game:
    random.seed(seed)
    game = Game(
        cards,
        card_names=registry.card_names,
        card_index_map=registry.card_index_map,
    )
    game.add_player("P0", RandomAgent("P0"))
    game.add_player("P1", RandomAgent("P1"))
    game.start_game(first_player_name="P0")
    return game


def _choose_action(game: Game):
    actions = get_available_actions(game, game.current_player)
    pending = game.current_player.get_current_pending_set()
    if pending is not None:
        return actions[0]
    priorities = (
        ActionType.PLAY_CARD,
        ActionType.APPLY_EFFECT,
        ActionType.ATTACK_BASE,
        ActionType.ATTACK_PLAYER,
        ActionType.BUY_CARD,
        ActionType.END_TURN,
    )
    for action_type in priorities:
        candidates = [
            action for action in actions if action.type == action_type
        ]
        if candidates:
            if action_type == ActionType.BUY_CARD:
                return max(
                    candidates,
                    key=lambda action: (
                        action.card.cost if action.card is not None else -1,
                        action.card_id if action.card_id is not None else -1,
                    ),
                )
            return candidates[0]
    raise AssertionError("live game has no legal action")


def _advance(game: Game, steps: int) -> None:
    for _ in range(steps):
        if game.is_game_over:
            return
        game.apply_decision(_choose_action(game))


def _acquire_named_to_hand(game: Game, name: str):
    for zone in (game.trade_row, game.trade_deck):
        for index, card in enumerate(zone):
            if card.name != name:
                continue
            zone.pop(index)
            game._remove_public_trade_card(card)
            game.current_player.add_owned_card(card)
            game.current_player.hand.append(card)
            return card
    raise AssertionError(f"{name} is not available in the trade supply")


def _full_hidden_signature(game: Game) -> tuple:
    return (
        tuple(
            (
                tuple(card.index for card in player.hand),
                tuple(card.index for card in player.deck),
                tuple(card.index for card in player.ordered_top),
                tuple(card.index for card in player.discard_pile),
                tuple(card.index for card in player.played_cards),
                tuple(card.index for card in player.bases),
                player.health,
                player.trade,
                player.combat,
            )
            for player in game.players
        ),
        tuple(card.index for card in game.trade_row),
        tuple(card.index for card in game.trade_deck),
        game.current_turn,
        game.stats.total_turns,
        game.is_game_over,
    )


def _hidden_variant(game: Game, seed: int) -> Game:
    memo = {id(player.agent): player.agent for player in game.players}
    variant = copy.deepcopy(game, memo)
    rng = random.Random(seed)
    viewer_index = variant.players.index(variant.current_player)
    viewer = variant.players[viewer_index]
    top_count = len(viewer.ordered_top)
    viewer_unknown = (
        viewer.deck[:-top_count] if top_count else list(viewer.deck)
    )
    rng.shuffle(viewer_unknown)
    viewer.deck = [*viewer_unknown, *viewer.ordered_top]

    opponent = variant.players[1 - viewer_index]
    known_hand = list(opponent.public_known_hand)
    known_ids = {id(card) for card in known_hand}
    unknown_hand = [
        card for card in opponent.hand if id(card) not in known_ids
    ]
    top_count = len(opponent.ordered_top)
    unknown = [
        *unknown_hand,
        *(
            opponent.deck[:-top_count]
            if top_count
            else opponent.deck
        ),
    ]
    rng.shuffle(unknown)
    opponent.hand = [
        *known_hand,
        *unknown[: len(unknown_hand)],
    ]
    opponent.deck = [
        *unknown[len(unknown_hand):],
        *opponent.ordered_top,
    ]
    rng.shuffle(variant.trade_deck)
    return variant


def test_public_ledgers_remain_exact_through_complete_games(card_data):
    cards, registry, _ = card_data
    for seed in range(3):
        game = _make_game(cards, registry, seed)
        steps = 0
        while not game.is_game_over and steps < 5000:
            assert_public_knowledge_consistent(game)
            game.apply_decision(_choose_action(game))
            steps += 1
        assert game.is_game_over
        assert_public_knowledge_consistent(game)


def test_same_information_ignores_true_hidden_arrangement(card_data):
    cards, registry, _ = card_data
    game = _make_game(cards, registry, seed=11)
    _advance(game, 140)
    assert not game.is_game_over
    variant = _hidden_variant(game, seed=999)
    assert _full_hidden_signature(game) != _full_hidden_signature(variant)
    assert_public_knowledge_consistent(variant)
    original_observation = encode_state(
        game,
        is_current_player_training=True,
        cards=registry.card_names,
        card_index_map=registry.card_index_map,
        return_numpy=True,
    )
    variant_observation = encode_state(
        variant,
        is_current_player_training=True,
        cards=registry.card_names,
        card_index_map=registry.card_index_map,
        return_numpy=True,
    )
    assert np.array_equal(original_observation, variant_observation)

    first = project_information_set(game)
    second = project_information_set(variant)
    assert first.fingerprint == second.fingerprint
    assert _full_hidden_signature(
        determinize(first, seed=41)
    ) == _full_hidden_signature(
        determinize(second, seed=41)
    )


def test_seeded_determinization_is_reproducible_and_diverse(card_data):
    cards, registry, _ = card_data
    game = _make_game(cards, registry, seed=12)
    _advance(game, 160)
    information = project_information_set(game)

    global_rng_before = random.getstate()
    first = determinize(information, seed=7)
    second = determinize(information, seed=7)
    assert random.getstate() == global_rng_before
    assert _full_hidden_signature(first) == _full_hidden_signature(second)
    for _ in range(300):
        if first.is_game_over or second.is_game_over:
            break
        assert _full_hidden_signature(first) == _full_hidden_signature(second)
        first.apply_decision(_choose_action(first))
        second.apply_decision(_choose_action(second))
    assert _full_hidden_signature(first) == _full_hidden_signature(second)

    signatures = {
        _full_hidden_signature(determinize(information, seed=seed))
        for seed in range(8)
    }
    assert len(signatures) > 1


def test_public_state_and_card_conservation_survive_sampling(card_data):
    cards, registry, _ = card_data
    game = _make_game(cards, registry, seed=13)
    _advance(game, 120)
    information = project_information_set(game)

    for seed in range(5):
        branch = determinize(information, seed)
        assert_public_knowledge_consistent(branch)
        assert (
            project_information_set(branch).fingerprint
            == information.fingerprint
        )
        viewer = branch.players[information.viewer_index]
        original_viewer = game.players[information.viewer_index]
        assert [card.index for card in viewer.hand] == [
            card.index for card in original_viewer.hand
        ]
        assert [card.index for card in branch.trade_row] == [
            card.index for card in game.trade_row
        ]


def test_publicly_known_hand_and_top_constraints_are_preserved(card_data):
    cards, registry, _ = card_data
    game = _make_game(cards, registry, seed=14)
    viewer = game.current_player
    opponent = game.get_opponent(viewer)
    acquired = game.trade_row.pop()
    game._remove_public_trade_card(acquired)
    opponent.add_owned_card(acquired)
    opponent.put_on_top(acquired)

    top_information = project_information_set(game)
    for seed in range(4):
        branch = determinize(top_information, seed)
        sampled_opponent = branch.get_opponent(branch.current_player)
        assert sampled_opponent.deck[-1].index == acquired.index
        assert sampled_opponent.ordered_top[-1] is sampled_opponent.deck[-1]

    assert opponent.draw_card() is acquired
    assert opponent.public_known_hand == [acquired]
    hand_information = project_information_set(game)
    for seed in range(4):
        branch = determinize(hand_information, seed)
        sampled_opponent = branch.get_opponent(branch.current_player)
        sampled_counts = Counter(
            card.index for card in sampled_opponent.hand
        )
        assert sampled_counts[acquired.index] >= 1


def test_pending_copy_references_and_dynamic_card_state_survive(card_data):
    cards, registry, _ = card_data
    game = _make_game(cards, registry, seed=141)
    ram = _acquire_named_to_hand(game, "Ram")
    needle = _acquire_named_to_hand(game, "Stealth Needle")
    game.apply_decision(next(
        action
        for action in get_available_actions(game, game.current_player)
        if action.type == ActionType.PLAY_CARD and action.card is ram
    ))
    game.apply_decision(next(
        action
        for action in get_available_actions(game, game.current_player)
        if action.type == ActionType.PLAY_CARD and action.card is needle
    ))
    assert game.current_player.get_current_pending_set() is not None

    branch = determinize(project_information_set(game), seed=77)
    pending = branch.current_player.get_current_pending_set()
    assert pending is not None
    copy_action = next(
        action
        for action in pending.actions
        if action.card is not None and action.card.name == "Ram"
    )
    branch.apply_decision(copy_action)
    copied_needle = next(
        card
        for card in branch.current_player.played_cards
        if card.name == "Stealth Needle"
    )
    assert copied_needle.effective_rule_card_index == copy_action.card.index
    assert_public_knowledge_consistent(branch)


def test_pending_explorer_reference_survives(card_data):
    cards, registry, _ = card_data
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
    pending = game.current_player.get_current_pending_set()

    information = project_information_set(game)
    branch = determinize(information, seed=78)
    sampled_explorer = branch.explorer_pile[-1]
    sampled_action = branch.current_player.get_current_pending_set().actions[0]
    assert sampled_action.card is sampled_explorer
    assert not sampled_explorer.effects[1].applied
    assert information.fingerprint == project_information_set(
        branch
    ).fingerprint
    branch.apply_decision(sampled_action)
    assert sampled_explorer is branch.current_player.deck[-1]
    assert not sampled_explorer.effects[1].applied
    assert_public_knowledge_consistent(branch)
    reprojected = project_information_set(branch)
    resampled = determinize(reprojected, seed=79)
    assert not resampled.current_player.deck[-1].effects[1].applied


def test_self_scrapped_explorer_returns_pristine(card_data):
    cards, registry, _ = card_data
    game = _make_game(cards, registry, seed=143)
    explorer = game.explorer_pile.pop()
    game.current_player.add_owned_card(explorer)
    game.current_player.hand.append(explorer)
    game.apply_decision(next(
        action
        for action in get_available_actions(game, game.current_player)
        if action.type == ActionType.PLAY_CARD and action.card is explorer
    ))
    scrap = next(
        action
        for action in get_available_actions(game, game.current_player)
        if action.type == ActionType.APPLY_EFFECT
        and action.card is explorer
        and action.card_effect.is_scrap_effect
    )
    game.apply_decision(scrap)
    assert game.explorer_pile[-1] is explorer
    assert all(not effect.applied for effect in explorer.effects)
    assert_public_knowledge_consistent(game)


def test_determinizations_are_independent_playable_clones(card_data):
    cards, registry, _ = card_data
    game = _make_game(cards, registry, seed=15)
    _advance(game, 100)
    information = project_information_set(game)
    first = determinize(information, seed=21)
    second = determinize(information, seed=21)
    second_before = _full_hidden_signature(second)

    first.apply_decision(_choose_action(first))
    assert _full_hidden_signature(second) == second_before
    assert information.fingerprint == project_information_set(
        second
    ).fingerprint

    for branch in (
        determinize(information, seed=31),
        determinize(information, seed=32),
    ):
        steps = 0
        while not branch.is_game_over and steps < 5000:
            branch.apply_decision(_choose_action(branch))
            steps += 1
        assert branch.is_game_over
        assert_public_knowledge_consistent(branch)


def test_determinized_state_matches_vector_legality(card_data):
    cards, registry, rules = card_data
    game = _make_game(cards, registry, seed=16)
    _advance(game, 130)
    branch = determinize(project_information_set(game), seed=88)
    state = from_games([branch], rules)
    engine = VectorEngine(state, rules)
    vector_mask = engine.build_masks()[0]
    scalar = build_action_context(
        branch,
        branch.current_player,
        registry.card_index_map,
        get_action_space_size(registry.card_names),
    )

    assert np.array_equal(scalar.mask, vector_mask)
    assert_canonical_equal(
        canonical_scalar_snapshot(branch, rules),
        canonical_vector_snapshot(state, 0),
    )


def test_determinized_game_matches_vector_transitions_to_completion(card_data):
    cards, registry, rules = card_data
    game = _make_game(cards, registry, seed=17)
    _advance(game, 90)
    branch = determinize(project_information_set(game), seed=89)
    result = run_differential_trace(
        [branch],
        rules,
        registry,
        max_steps=2000,
    )
    assert result.games_completed == 1
    assert result.action_counts["play"] > 0
    assert result.action_counts["buy"] > 0
    assert result.action_counts["end"] > 0
    assert_public_knowledge_consistent(branch)


def test_projection_rejects_games_without_public_ledgers(card_data):
    _, registry, _ = card_data
    game = Game([], card_index_map=registry.card_index_map)
    first = game.add_player("P0", RandomAgent("P0"))
    game.add_player("P1", RandomAgent("P1"))
    game.current_player = first
    with pytest.raises(
        InformationSetError,
        match="public trade-supply ledger",
    ):
        project_information_set(game)


def test_projection_requires_unambiguous_card_identity(card_data):
    cards, _, _ = card_data
    game = Game(cards)
    game.add_player("P0", RandomAgent("P0"))
    game.add_player("P1", RandomAgent("P1"))
    game.start_game(first_player_name="P0")
    with pytest.raises(InformationSetError, match="card_index_map"):
        project_information_set(game)


def test_projection_rejects_mutable_hidden_card_state(card_data):
    cards, registry, _ = card_data
    game = _make_game(cards, registry, seed=19)
    hidden = game.get_opponent(game.current_player).deck[0]
    hidden.effects[0].applied = True
    with pytest.raises(InformationSetError, match="outside public play"):
        project_information_set(game)


def test_projection_rejects_owned_card_index_collision(card_data):
    cards, registry, _ = card_data
    game = _make_game(cards, registry, seed=20)
    opponent = game.get_opponent(game.current_player)
    card = opponent.deck[0]
    old_index = card.index
    wrong_name = next(
        name
        for name in registry.card_names
        if name != card.name
    )
    wrong_index = registry.card_index_map[wrong_name]
    opponent.public_owned_cards[old_index] -= 1
    if opponent.public_owned_cards[old_index] == 0:
        del opponent.public_owned_cards[old_index]
    opponent.public_owned_cards[wrong_index] += 1
    card.index = wrong_index
    with pytest.raises(InformationSetError, match="owned card index"):
        project_information_set(game)


def test_projection_rejects_hidden_pending_references(card_data):
    cards, registry, _ = card_data
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
    with pytest.raises(InformationSetError, match="hidden hand"):
        project_information_set(game)
