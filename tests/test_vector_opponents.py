"""Scalar-policy parity for fixed vector opponents."""
from __future__ import annotations

import numpy as np
import pytest

from src.ai.agent import Agent
from src.ai.heuristic_agent import HeuristicAgent
from src.ai.simple_agent import SimpleAgent
from src.cards.card import CardType
from src.config import DataConfig
from src.encoding.action_encoder import encode_action
from src.engine.actions import Action, ActionType, CardSource
from src.engine.game import Game
from src.vector_engine.kernels import END_TURN_INDEX, VectorEngine
from src.vector_engine.opponents import (
    FixedOpponent,
    compile_opponent_policy_data,
    sample_mixed_opponent_kinds,
    sample_opponent_kinds,
    select_fixed_actions,
)
from src.vector_engine.rules import compile_rules
from src.vector_engine.state import from_games
from src.ppo.vector_batch_runner import initialize_training_state


DATA = DataConfig()
CARDS = DATA.load_cards()
REGISTRY = DATA.build_registry(CARDS)
RULES = compile_rules(CARDS, REGISTRY)
POLICY_DATA = compile_opponent_policy_data(CARDS, REGISTRY, RULES)
TEMPLATES = {card.name: card for card in CARDS}


def _card(name):
    return TEMPLATES[name].clone()


def _game():
    game = Game([], card_index_map=REGISTRY.card_index_map)
    player = game.add_player("P0", Agent("P0"))
    opponent = game.add_player("P1", Agent("P1"))
    for participant in game.players:
        game.stats.add_player(participant.name)
    game.current_player = player
    game.current_turn = 0
    game.first_player_name = player.name
    game.is_running = True
    return game, player, opponent


def _vector_choice(game, kind):
    engine = VectorEngine(from_games([game], RULES), RULES)
    engine.build_masks()
    actions = np.asarray([END_TURN_INDEX], dtype=np.int32)
    variants = np.zeros(1, dtype=np.int16)
    select_fixed_actions(
        engine,
        POLICY_DATA,
        np.asarray([int(kind)], dtype=np.int8),
        np.asarray([0], dtype=np.int64),
        actions,
        variants,
    )
    return int(actions[0]), int(variants[0])


def _scalar_choice(game, agent):
    action = agent.make_decision(game)
    return encode_action(
        action,
        REGISTRY.card_names,
        REGISTRY.card_index_map,
    )


@pytest.mark.parametrize(
    "fixture",
    ["play", "buy", "attack", "effect", "pending"],
)
def test_simple_matches_scalar_action_group_priority(fixture):
    game, player, opponent = _game()
    if fixture == "play":
        player.hand = [_card("Trade Pod")]
        player.trade = 10
        game.trade_row = [_card("Battle Blob")]
    elif fixture == "buy":
        player.trade = 10
        game.trade_row = [_card("Trade Pod"), _card("Battle Blob")]
    elif fixture == "attack":
        player.combat = 8
        opponent.bases = [_card("Blob Wheel")]
    elif fixture == "effect":
        station = _card("Battle Station")
        player.played_cards = [station]
        player.bases = [station]
    else:
        scout = Game([], card_index_map=REGISTRY.card_index_map).create_starting_deck()[0]
        player.hand = [scout]
        player.add_pending_actions(
            [
                Action(
                    ActionType.DISCARD_CARDS,
                    card_id=scout.index,
                    card=scout,
                    card_source=CardSource.SELF,
                )
            ],
            1,
            mandatory=True,
        )

    expected = _scalar_choice(game, SimpleAgent("Simple"))
    actual, _ = _vector_choice(game, FixedOpponent.SIMPLE)
    assert actual == expected


def test_heuristic_matches_scalar_draw_first_play():
    game, player, _ = _game()
    player.hand = [_card("Fleet HQ"), _card("Battle Blob")]
    expected = _scalar_choice(game, HeuristicAgent())
    actual, _ = _vector_choice(game, FixedOpponent.HEURISTIC)
    assert actual == expected


def test_heuristic_matches_scalar_effect_priority():
    game, player, _ = _game()
    fighter = _card("Blob Fighter")
    fighter.effects[0].applied = True
    ally = _card("Trade Pod")
    for effect in ally.effects:
        effect.applied = True
    player.played_cards = [fighter, ally]

    expected = _scalar_choice(game, HeuristicAgent())
    actual, _ = _vector_choice(game, FixedOpponent.HEURISTIC)
    assert actual == expected


def test_heuristic_or_followup_uses_parent_priority():
    game, player, _ = _game()
    barter = _card("Barter World")
    barter.effects[0].child_effects[1].applied = True
    barter.effects[1].applied = True

    patrol = _card("Patrol Mech")
    for child in patrol.effects[0].child_effects:
        child.applied = True
    ally = _card("Battle Station")
    ally.effects[0].applied = True
    player.played_cards = [barter, patrol, ally]
    player.bases = [barter, ally]

    scalar_action = HeuristicAgent().make_decision(game)
    assert scalar_action.card is patrol
    expected = encode_action(
        scalar_action, REGISTRY.card_names, REGISTRY.card_index_map
    )
    actual, _ = _vector_choice(game, FixedOpponent.HEURISTIC)
    assert actual == expected


def test_heuristic_matches_scalar_lethal_attack():
    game, player, opponent = _game()
    player.combat = 10
    opponent.health = 8
    expected = _scalar_choice(game, HeuristicAgent())
    actual, _ = _vector_choice(game, FixedOpponent.HEURISTIC)
    assert actual == expected


def test_heuristic_matches_scalar_cheapest_base_attack():
    game, player, opponent = _game()
    player.combat = 10
    expensive = _card("Fleet HQ")
    cheap = _card("Blob Wheel")
    opponent.bases = [expensive, cheap]
    expected = _scalar_choice(game, HeuristicAgent())
    actual, _ = _vector_choice(game, FixedOpponent.HEURISTIC)
    assert actual == expected


def test_heuristic_matches_scalar_buy_scoring_with_synergy():
    game, player, _ = _game()
    player.trade = 8
    player.discard_pile = [_card("Trade Pod"), _card("Blob Fighter")]
    game.trade_row = [_card("Fleet HQ"), _card("Battle Blob")]
    expected = _scalar_choice(game, HeuristicAgent())
    actual, _ = _vector_choice(game, FixedOpponent.HEURISTIC)
    assert actual == expected


@pytest.mark.parametrize(
    ("deck_size", "expected_name"),
    [(10, "Viper"), (15, "Scout")],
)
def test_heuristic_matches_scalar_phase_aware_scrap(deck_size, expected_name):
    game, player, _ = _game()
    starting = Game([], card_index_map=REGISTRY.card_index_map).create_starting_deck()
    scout = next(card for card in starting if card.name == "Scout")
    viper = next(card for card in starting if card.name == "Viper")
    player.hand = [scout, viper]
    if deck_size > 2:
        player.deck = [scout.clone() for _ in range(deck_size - 2)]
    actions = [
        Action(
            ActionType.SCRAP_CARD,
            card_id=scout.index,
            card=scout,
            card_source=CardSource.HAND,
        ),
        Action(
            ActionType.SCRAP_CARD,
            card_id=viper.index,
            card=viper,
            card_source=CardSource.HAND,
        ),
    ]
    player.add_pending_actions(actions, 1, mandatory=False)

    scalar_action = HeuristicAgent().make_decision(game)
    assert scalar_action.card.name == expected_name
    expected = encode_action(
        scalar_action, REGISTRY.card_names, REGISTRY.card_index_map
    )
    actual, _ = _vector_choice(game, FixedOpponent.HEURISTIC)
    assert actual == expected


def test_heuristic_matches_scalar_discard_value():
    game, player, _ = _game()
    starting = Game([], card_index_map=REGISTRY.card_index_map).create_starting_deck()
    scout = next(card for card in starting if card.name == "Scout")
    viper = next(card for card in starting if card.name == "Viper")
    player.hand = [viper, scout]
    player.add_pending_actions(
        [
            Action(
                ActionType.DISCARD_CARDS,
                card_id=viper.index,
                card=viper,
                card_source=CardSource.SELF,
            ),
            Action(
                ActionType.DISCARD_CARDS,
                card_id=scout.index,
                card=scout,
                card_source=CardSource.SELF,
            ),
        ],
        1,
        mandatory=True,
    )
    expected = _scalar_choice(game, HeuristicAgent())
    actual, _ = _vector_choice(game, FixedOpponent.HEURISTIC)
    assert actual == expected


def test_weighted_opponent_sampling_is_reproducible_and_normalized():
    entries = [("random", 0.6), ("heuristic", 0.3), ("simple", 0.1)]
    first = sample_opponent_kinds(entries, 100_000, seed=123)
    second = sample_opponent_kinds(entries, 100_000, seed=123)
    assert np.array_equal(first, second)
    frequencies = np.bincount(first, minlength=3) / len(first)
    assert frequencies[int(FixedOpponent.RANDOM)] == pytest.approx(0.6, abs=0.01)
    assert frequencies[int(FixedOpponent.HEURISTIC)] == pytest.approx(
        0.3, abs=0.01
    )
    assert frequencies[int(FixedOpponent.SIMPLE)] == pytest.approx(0.1, abs=0.01)


def test_snapshot_assignment_obeys_ratio_and_pfsp_weights():
    entries = [("random", 0.5), ("heuristic", 0.5)]
    assignments = sample_mixed_opponent_kinds(
        entries,
        snapshot_count=2,
        self_play_ratio=0.4,
        pfsp_weights=[0.8, 0.2],
        games=100_000,
        seed=77,
    )
    snapshot = assignments >= 3
    assert snapshot.mean() == pytest.approx(0.4, abs=0.01)
    selected = assignments[snapshot] - 3
    assert (selected == 0).mean() == pytest.approx(0.8, abs=0.01)


def test_snapshot_assignment_falls_back_to_fixed_without_snapshots():
    entries = [("random", 0.25), ("simple", 0.75)]
    mixed = sample_mixed_opponent_kinds(
        entries,
        snapshot_count=0,
        self_play_ratio=1.0,
        pfsp_weights=None,
        games=500,
        seed=12,
    )
    fixed = sample_opponent_kinds(entries, games=500, seed=12)
    np.testing.assert_array_equal(mixed, fixed)


@pytest.mark.parametrize(
    ("player_zero", "player_one"),
    [
        (FixedOpponent.HEURISTIC, FixedOpponent.RANDOM),
        (FixedOpponent.SIMPLE, FixedOpponent.RANDOM),
        (FixedOpponent.HEURISTIC, FixedOpponent.SIMPLE),
    ],
)
def test_fixed_vector_opponents_complete_full_games(player_zero, player_one):
    state, rules = initialize_training_state(
        12, CARDS, REGISTRY, seed=41
    )
    engine = VectorEngine(state, rules)
    actions = np.full(12, END_TURN_INDEX, dtype=np.int32)
    variants = np.zeros(12, dtype=np.int16)
    rows = np.arange(12, dtype=np.int64)

    for _ in range(20_000):
        if state.done.all():
            break
        masks = engine.build_masks()
        active_rows = rows[~state.done]
        kinds = np.where(
            state.current_player == 0,
            int(player_zero),
            int(player_one),
        ).astype(np.int8)
        select_fixed_actions(
            engine,
            POLICY_DATA,
            kinds,
            active_rows,
            actions,
            variants,
        )
        assert masks[active_rows, actions[active_rows]].all()
        engine.step(actions, action_variants=variants)
    else:
        pytest.fail("fixed-opponent vector games exceeded the safety limit")

    assert state.done.all()
    assert np.isin(state.winner, [0, 1]).all()
