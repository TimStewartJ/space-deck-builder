"""Scalar-policy parity for fixed vector opponents."""
from __future__ import annotations

import numpy as np
import pytest

from src.ai.agent import Agent
from src.ai.heuristic_agent import HeuristicAgent
from src.ai.simple_agent import SimpleAgent
from src.cards.card import CardType
from src.config import DataConfig
from src.encoding.action_encoder import encode_action, get_action_offsets
from src.engine.actions import Action, ActionType, CardSource, get_available_actions
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
from src.vector_engine.state import (
    assert_canonical_equal,
    canonical_scalar_snapshot,
    canonical_vector_snapshot,
    from_games,
)
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


def _or_game(card_name, *, row=(), combat=0, opponent_health=50, allies=()):
    """A main-phase decision where only *card_name*'s OR ability is unused."""
    game, player, opponent = _game()
    card = _card(card_name)
    for effect in card.effects:
        if not effect.is_or_effect:
            effect.applied = True
    player.played_cards = [card, *(_card(name) for name in allies)]
    for ally in player.played_cards[1:]:
        for effect in ally.effects:
            effect.applied = True
    for played in player.played_cards:
        player.record_faction_play(played.effective_faction)
    if card.is_base():
        player.bases = [card]
    player.combat = combat
    opponent.health = opponent_health
    game.trade_row = [_card(name) for name in row]
    return game, card


@pytest.mark.parametrize(
    ("card_name", "situation", "expected_group"),
    [
        ("Trading Post", dict(row=("Federation Shuttle",)), "effect_alt"),
        ("Trading Post", dict(row=("Cutter",)), "effect_nonscrap"),
        ("Barter World", dict(row=("Cutter",)), "effect_alt"),
        ("Barter World", dict(), "effect_nonscrap"),
        ("Defense Center", dict(), "effect_alt"),
        ("Blob World", dict(allies=("Trade Pod",)), "effect_alt"),
        ("Blob World", dict(), "effect_nonscrap"),
        (
            "Blob World",
            dict(allies=("Trade Pod",), combat=2, opponent_health=7),
            "effect_nonscrap",
        ),
        ("Recycling Station", dict(row=("Federation Shuttle",)), "effect_nonscrap"),
        ("Recycling Station", dict(), "effect_nonscrap"),
    ],
)
def test_heuristic_or_option_rule_matches_scalar(card_name, situation, expected_group):
    game, card = _or_game(card_name, **situation)
    offsets = get_action_offsets(REGISTRY.num_cards)
    expected = getattr(offsets, expected_group) + card.index
    assert _scalar_choice(game, HeuristicAgent()) == expected
    assert _vector_choice(game, FixedOpponent.HEURISTIC)[0] == expected


def _blob_world_after_a_scrap():
    """Blob World in play; Ram and Trade Pod played this turn, Ram scrapped.

    Every other ability is used, so only Blob World's OR ability is left.
    """
    game, player, _ = _game()
    world = _card("Blob World")
    player.bases = [world]
    ram, pod = _card("Ram"), _card("Trade Pod")
    for card in (ram, pod):
        player.hand.append(card)
        game.execute_action(Action(ActionType.PLAY_CARD, card=card, card_id=card.index))
    scrap = next(
        action for action in get_available_actions(game, player)
        if action.type == ActionType.APPLY_EFFECT
        and action.card is ram
        and action.card_effect.is_scrap_effect
    )
    game.execute_action(scrap)
    assert ram not in player.played_cards
    while True:
        other = next(
            (
                action for action in get_available_actions(game, player)
                if action.type == ActionType.APPLY_EFFECT and action.card is not world
            ),
            None,
        )
        if other is None:
            break
        game.execute_action(other)
    return game, world


def test_heuristic_counts_a_blob_card_scrapped_this_turn():
    """Two Blob cards were played, so Blob World's draw option is taken."""
    game, world = _blob_world_after_a_scrap()
    expected = get_action_offsets(REGISTRY.num_cards).effect_alt + world.index
    assert _scalar_choice(game, HeuristicAgent()) == expected
    assert _vector_choice(game, FixedOpponent.HEURISTIC)[0] == expected


@pytest.mark.parametrize(
    "style", ["balanced", "aggro", "economy", "scrapper", "fortress", "blobrush"]
)
def test_held_out_bots_count_a_blob_card_scrapped_this_turn(style):
    from src.ai.held_out_agents import HeldOutAgent

    game, world = _blob_world_after_a_scrap()
    player, opponent = game.players
    ability = world.effects[0]
    pick = HeldOutAgent("P0", style)._choose_or(game, player, opponent, ability)
    assert pick is ability.child_effects[1]


def _established_central_office():
    """Central Office in play since an earlier turn, a Cutter for sale."""
    game, player, _ = _game()
    office = _card("Central Office")
    player.bases = [office]
    game.trade_row = [_card("Cutter")]
    return game, player, office


@pytest.mark.parametrize("kind", [FixedOpponent.HEURISTIC, FixedOpponent.SIMPLE])
def test_fixed_opponents_use_central_office_in_two_activations(kind):
    """Its trade and its optional top-deck are two actions in both engines."""
    game, player, office = _established_central_office()
    agent = HeuristicAgent() if kind == FixedOpponent.HEURISTIC else SimpleAgent("Simple")
    engine = VectorEngine(from_games([game], RULES), RULES)
    effect = get_action_offsets(REGISTRY.num_cards).effect_nonscrap + office.index
    chosen = []
    while True:
        engine.build_masks()
        actions = np.asarray([END_TURN_INDEX], dtype=np.int32)
        variants = np.zeros(1, dtype=np.int16)
        select_fixed_actions(
            engine, POLICY_DATA, np.asarray([int(kind)], dtype=np.int8),
            np.asarray([0], dtype=np.int64), actions, variants,
        )
        scalar_action = agent.make_decision(game)
        index = encode_action(scalar_action, REGISTRY.card_names, REGISTRY.card_index_map)
        assert int(actions[0]) == index
        if scalar_action.type == ActionType.END_TURN:
            break
        chosen.append(index)
        game.execute_action(scalar_action)
        engine.step(actions, action_variants=variants)
        assert_canonical_equal(
            canonical_scalar_snapshot(game, RULES),
            canonical_vector_snapshot(engine.state, 0),
        )
    assert chosen.count(effect) == 2


@pytest.mark.parametrize(
    "style", ["balanced", "aggro", "economy", "scrapper", "fortress", "blobrush"]
)
def test_held_out_bots_take_central_office_top_deck_on_later_turns(style):
    """As on the turn it is played, the bots take the trade and the top-deck."""
    from src.ai.held_out_agents import HeldOutAgent

    game, player, office = _established_central_office()
    agent = HeldOutAgent("P0", style)
    first = agent.make_decision(game)
    assert first.card is office
    game.execute_action(first)
    assert (player.trade, player.next_ship_to_top) == (2, 0)
    second = agent.make_decision(game)
    assert second.card is office
    game.execute_action(second)
    assert (player.trade, player.next_ship_to_top) == (2, 1)


def _patrol_mech_choice(*, hand=(), row=(), combat=0, opponent_health=50):
    """Patrol Mech just played: its OR ability awaits a choice."""
    game, player, opponent = _game()
    patrol = _card("Patrol Mech")
    scout = Game([], card_index_map=REGISTRY.card_index_map).create_starting_deck()[0]
    player.hand = [patrol, *(scout.clone() for name in hand if name == "Scout")]
    player.combat = combat
    opponent.health = opponent_health
    game.trade_row = [_card(name) for name in row]
    game.execute_action(Action(ActionType.PLAY_CARD, card=patrol, card_id=patrol.index))
    assert player.get_current_pending_set() is not None
    return game, patrol


@pytest.mark.parametrize(
    ("situation", "expected_group"),
    [
        (dict(combat=1, opponent_health=6), "effect_alt"),
        (dict(row=("Blob Wheel",)), "effect_nonscrap"),
        (dict(row=("Battle Blob",)), "effect_alt"),
        # Three Scouts still in hand already afford the Blob Wheel.
        (dict(row=("Blob Wheel",), hand=("Scout",) * 3), "effect_alt"),
    ],
)
def test_heuristic_chooses_patrol_mech_option_when_played(situation, expected_group):
    game, patrol = _patrol_mech_choice(**situation)
    offsets = get_action_offsets(REGISTRY.num_cards)
    expected = getattr(offsets, expected_group) + patrol.index
    assert _scalar_choice(game, HeuristicAgent()) == expected
    assert _vector_choice(game, FixedOpponent.HEURISTIC)[0] == expected


def test_simple_takes_patrol_mech_first_option_when_played():
    game, patrol = _patrol_mech_choice(hand=("Scout",))
    expected = get_action_offsets(REGISTRY.num_cards).effect_nonscrap + patrol.index
    assert _scalar_choice(game, SimpleAgent("Simple")) == expected
    assert _vector_choice(game, FixedOpponent.SIMPLE)[0] == expected


def test_heuristic_counts_the_hand_it_will_play_toward_a_purchase():
    agent = HeuristicAgent()
    game, card = _or_game("Trading Post", row=("Federation Shuttle",))
    player = game.current_player
    options = [
        action
        for action in get_available_actions(game, player)
        if action.card is card and action.card_effect.or_parent is not None
    ]
    assert agent._choose_or_option(game, player, options).card_effect.option_index == 1
    # A Scout still to be played already affords the Federation Shuttle.
    scout = Game([], card_index_map=REGISTRY.card_index_map).create_starting_deck()[0]
    player.hand = [scout]
    assert agent._choose_or_option(game, player, options).card_effect.option_index == 0


def test_heuristic_takes_the_chosen_option_of_each_copy():
    game, card = _or_game(
        "Trading Post", row=("Federation Shuttle", "Cutter")
    )
    second = _card("Trading Post")
    game.current_player.bases.append(second)
    engine = VectorEngine(from_games([game], RULES), RULES)
    alt = get_action_offsets(REGISTRY.num_cards).effect_alt + card.index
    for used in (card, second):
        engine.build_masks()
        actions = np.asarray([END_TURN_INDEX], dtype=np.int32)
        variants = np.zeros(1, dtype=np.int16)
        select_fixed_actions(
            engine, POLICY_DATA,
            np.asarray([int(FixedOpponent.HEURISTIC)], dtype=np.int8),
            np.asarray([0], dtype=np.int64), actions, variants,
        )
        scalar_action = HeuristicAgent().make_decision(game)
        assert scalar_action.card is used
        assert (int(actions[0]), int(variants[0])) == (alt, 0)
        game.execute_action(scalar_action)
        engine.step(actions, action_variants=variants)
        assert_canonical_equal(
            canonical_scalar_snapshot(game, RULES),
            canonical_vector_snapshot(engine.state, 0),
        )


def test_simple_takes_the_first_option_of_an_or_ability():
    game, card = _or_game("Defense Center")
    expected = get_action_offsets(REGISTRY.num_cards).effect_nonscrap + card.index
    assert _scalar_choice(game, SimpleAgent("Simple")) == expected
    assert _vector_choice(game, FixedOpponent.SIMPLE)[0] == expected


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


def test_snapshot_assignment_caps_models_per_rollout():
    assignments = sample_mixed_opponent_kinds(
        [("random", 1.0)],
        snapshot_count=10,
        self_play_ratio=0.5,
        pfsp_weights=[1.0] * 10,
        games=20_000,
        seed=99,
        max_active_snapshots=3,
    )
    snapshot_assignments = assignments[assignments >= 3]
    assert len(np.unique(snapshot_assignments)) == 3
    assert len(snapshot_assignments) / len(assignments) == pytest.approx(
        0.5, abs=0.01
    )


def test_snapshot_cap_preserves_pool_coverage_across_rollouts():
    selected = set()
    for seed in range(40):
        assignments = sample_mixed_opponent_kinds(
            [("random", 1.0)],
            snapshot_count=10,
            self_play_ratio=1.0,
            pfsp_weights=[1.0] * 10,
            games=200,
            seed=seed,
            max_active_snapshots=3,
        )
        selected.update((np.unique(assignments) - 3).tolist())
    assert selected == set(range(10))


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
