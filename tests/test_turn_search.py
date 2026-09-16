"""Correctness gates for turn-bounded information-set search."""
from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from src.ai.random_agent import RandomAgent
from src.config import DataConfig, ModelConfig, load_checkpoint
from src.encoding.action_context import build_action_context
from src.encoding.action_encoder import (
    END_TURN_INDEX,
    get_action_space_layout,
    get_action_space_size,
)
from src.encoding.state_encoder import get_state_size
from src.encoding.state_utils import unpack_state
from src.engine.actions import get_available_actions
from src.engine.game import Game
from src.ppo.ppo_actor_critic import PPOActorCritic
from src.search.agent import SearchAgent
from src.search.information_set import (
    InformationSetError,
    assert_public_knowledge_consistent,
    project_information_set,
)
from src.search.turn_search import TurnSearch, TurnSearchConfig
from src.search.vector_determinization import (
    derive_search_seed,
    known_hand_counts,
    sample_world_hands,
    tile_state,
)
from src.utils.logger import set_disabled
from src.vector_engine.rules import compile_rules
from src.vector_engine.state import ZONE_DECK, ZONE_HAND, from_games

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO / "results" / "optimized_selfplay_baseline" / "checkpoints" / "selfplay_seed0.pth"


@pytest.fixture(scope="module")
def card_data():
    set_disabled(True)
    data = DataConfig()
    cards = data.load_cards()
    registry = data.build_registry(cards)
    rules = compile_rules(cards, registry)
    return cards, registry, rules


@pytest.fixture(scope="module")
def policy(card_data):
    _, registry, _ = card_data
    names = registry.card_names
    checkpoint = load_checkpoint(str(CHECKPOINT), map_location="cpu")
    model = PPOActorCritic(
        get_state_size(names), get_action_space_size(names), len(names),
        model_config=ModelConfig.from_dict(checkpoint["config"]["model"]),
        card_registry=registry,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


class ScriptedPolicy(torch.nn.Module):
    """Fixed action preferences plus an optional value function of observations."""

    def __init__(self, action_dim: int, preferences: dict[int, float], value_fn=None):
        super().__init__()
        row = torch.zeros(action_dim)
        for action, logit in preferences.items():
            row[action] = logit
        self.register_buffer("row", row)
        self.value_fn = value_fn

    def forward(self, states):
        batch = states.shape[0]
        values = (
            self.value_fn(states) if self.value_fn is not None
            else torch.zeros(batch)
        )
        return self.row.expand(batch, -1).clone(), values


def _make_game(cards, registry, seed: int) -> Game:
    random.seed(seed)
    game = Game(cards, card_names=registry.card_names, card_index_map=registry.card_index_map)
    game.add_player("P0", RandomAgent("P0"))
    game.add_player("P1", RandomAgent("P1"))
    game.start_game(first_player_name="P0")
    return game


def _advance(game: Game, steps: int) -> None:
    for _ in range(steps):
        if game.is_game_over:
            return
        game.apply_decision(game.current_player.make_decision(game))


def _advance_to_choice(game: Game, registry, minimum_steps: int) -> None:
    _advance(game, minimum_steps)
    action_dim = get_action_space_size(registry.card_names)
    for _ in range(400):
        if game.is_game_over:
            break
        context = build_action_context(game, game.current_player, registry.card_index_map, action_dim)
        mask = context.mask.copy()
        if context.has_meaningful:
            mask[END_TURN_INDEX] = False
        if mask.sum() >= 3:
            return
        game.apply_decision(game.current_player.make_decision(game))
    pytest.skip("no multi-action decision reached")


def _hidden_variant(game: Game, seed: int) -> Game:
    """Same public information, different true hidden arrangement."""
    memo = {id(player.agent): player.agent for player in game.players}
    variant = copy.deepcopy(game, memo)
    rng = random.Random(seed)
    viewer_index = variant.players.index(variant.current_player)
    viewer = variant.players[viewer_index]
    top = len(viewer.ordered_top)
    unknown = viewer.deck[:-top] if top else list(viewer.deck)
    rng.shuffle(unknown)
    viewer.deck = [*unknown, *viewer.ordered_top]
    opponent = variant.players[1 - viewer_index]
    known = {id(card) for card in opponent.public_known_hand}
    unknown_hand = [card for card in opponent.hand if id(card) not in known]
    top = len(opponent.ordered_top)
    pool = [*unknown_hand, *(opponent.deck[:-top] if top else opponent.deck)]
    rng.shuffle(pool)
    opponent.hand = [*opponent.public_known_hand, *pool[: len(unknown_hand)]]
    opponent.deck = [*pool[len(unknown_hand):], *opponent.ordered_top]
    rng.shuffle(variant.trade_deck)
    return variant


def _result_signature(result):
    return (
        result.action, result.greedy_action, result.forced, result.rollouts,
        result.fingerprint,
        tuple((c.action, c.rollouts, round(c.value, 6), round(c.score, 6)) for c in result.candidates),
    )


def _take_from_supply(game: Game, name: str):
    for zone in (game.trade_row, game.trade_deck):
        for index, card in enumerate(zone):
            if card.name == name:
                zone.pop(index)
                game._remove_public_trade_card(card)
                return card
    raise AssertionError(f"{name} is not in the trade supply")


def _give_base(game: Game, player, name: str) -> None:
    card = _take_from_supply(game, name)
    player.add_owned_card(card)
    player.bases.append(card)
    game.fill_trade_row()


def _empty_hand(player) -> None:
    for card in list(player.hand):
        player.hand.remove(card)
        if card in player.public_known_hand:
            player.public_known_hand.remove(card)
        player.discard_pile.append(card)


def _tactical_game(cards, registry, seed: int = 5) -> Game:
    game = _make_game(cards, registry, seed)
    searcher = game.current_player
    _empty_hand(searcher)
    searcher.trade = 0
    searcher.combat = 0
    return game


def _layout(registry):
    layout = get_action_space_layout(len(registry.card_names))
    groups = {name: offset for name, offset, _ in layout["card_groups"]}
    return layout, groups


def test_world_hand_sampling_preserves_public_constraints(card_data):
    cards, registry, rules = card_data
    game = _make_game(cards, registry, seed=21)
    _advance(game, 150)
    information = project_information_set(game)
    viewer = information.viewer_index
    opponent = 1 - viewer
    root = from_games([information.canonical_game], rules, training_players=[viewer])
    known = known_hand_counts(information, opponent, rules.num_cards)
    state = tile_state(root, 6)
    worlds = np.array([0, 1, 2, 0, 1, 2])
    sample_world_hands(state, viewer=viewer, known_hand=known, worlds=worlds, base_seed=99)

    unseen = root.zones[0, opponent, ZONE_HAND] + root.zones[0, opponent, ZONE_DECK]
    for row in range(6):
        hand = state.zones[row, opponent, ZONE_HAND]
        deck = state.zones[row, opponent, ZONE_DECK]
        assert np.array_equal(hand + deck, unseen)
        assert hand.sum() == root.zones[0, opponent, ZONE_HAND].sum()
        assert (hand >= known).all()
        assert np.array_equal(state.zones[row, viewer], root.zones[0, viewer])
        assert np.array_equal(state.trade_deck[row], root.trade_deck[0])
    for a, b in ((0, 3), (1, 4), (2, 5)):
        assert np.array_equal(state.zones[a], state.zones[b])
        assert state.rng_state[a] == state.rng_state[b]
    assert len({int(value) for value in state.rng_state}) == 3
    assert all(int(value) > 0 for value in state.rng_state)


def test_world_hand_sampling_depends_only_on_the_unseen_multiset(card_data):
    cards, registry, rules = card_data
    game = _make_game(cards, registry, seed=22)
    _advance(game, 120)
    information = project_information_set(game)
    viewer = information.viewer_index
    opponent = 1 - viewer
    root = from_games([information.canonical_game], rules, training_players=[viewer])
    known = known_hand_counts(information, opponent, rules.num_cards)
    moved = tile_state(root, 1)
    unknown_in_hand = np.flatnonzero(moved.zones[0, opponent, ZONE_HAND] > known)
    in_deck = np.flatnonzero(moved.zones[0, opponent, ZONE_DECK] > 0)
    if len(unknown_in_hand) == 0 or len(in_deck) == 0:
        pytest.skip("position has no swappable hidden cards")
    hand_card, deck_card = int(unknown_in_hand[0]), int(in_deck[-1])
    moved.zones[0, opponent, ZONE_HAND, hand_card] -= 1
    moved.zones[0, opponent, ZONE_DECK, hand_card] += 1
    moved.zones[0, opponent, ZONE_DECK, deck_card] -= 1
    moved.zones[0, opponent, ZONE_HAND, deck_card] += 1

    first, second = tile_state(root, 4), tile_state(moved, 4)
    worlds = np.arange(4)
    for state in (first, second):
        sample_world_hands(state, viewer=viewer, known_hand=known, worlds=worlds, base_seed=7)
    assert np.array_equal(first.zones, second.zones)
    assert np.array_equal(first.rng_state, second.rng_state)


def test_world_hand_sampling_rejects_impossible_known_hands(card_data):
    cards, registry, rules = card_data
    game = _make_game(cards, registry, seed=23)
    information = project_information_set(game)
    viewer = information.viewer_index
    root = from_games([information.canonical_game], rules, training_players=[viewer])
    impossible = np.full(rules.num_cards, 50, dtype=np.int64)
    with pytest.raises(InformationSetError):
        sample_world_hands(
            tile_state(root, 2), viewer=viewer, known_hand=impossible,
            worlds=np.arange(2), base_seed=1,
        )


def test_search_seed_combines_seed_and_public_fingerprint():
    assert derive_search_seed(1, "abc") == derive_search_seed(1, "abc")
    assert derive_search_seed(1, "abc") != derive_search_seed(2, "abc")
    assert derive_search_seed(1, "abc") != derive_search_seed(1, "abd")


def test_search_is_repeatable_and_respects_budget(card_data, policy):
    cards, registry, rules = card_data
    game = _make_game(cards, registry, seed=31)
    _advance_to_choice(game, registry, 90)
    config = TurnSearchConfig(budget=24, max_candidates=5, seed=4)
    search = TurnSearch(policy, rules, registry.card_index_map, device="cpu", config=config)

    first, second = search.choose(game), search.choose(game)

    assert _result_signature(first) == _result_signature(second)
    assert not first.forced
    assert 0 < first.rollouts <= config.budget
    assert len(first.candidates) <= config.max_candidates
    assert sum(c.rollouts for c in first.candidates) == first.rollouts
    assert all(-1.0 <= c.value <= 1.0 for c in first.candidates)
    context = build_action_context(game, game.current_player, registry.card_index_map, search.action_dim)
    assert first.action in context.resolvers


def test_search_ignores_true_hidden_arrangement(card_data, policy):
    cards, registry, rules = card_data
    game = _make_game(cards, registry, seed=32)
    _advance_to_choice(game, registry, 140)
    variant = _hidden_variant(game, seed=1234)
    assert_public_knowledge_consistent(variant)
    opponent = 1 - game.players.index(game.current_player)
    assert (
        [c.index for c in game.players[opponent].hand]
        != [c.index for c in variant.players[opponent].hand]
        or [c.index for c in game.trade_deck] != [c.index for c in variant.trade_deck]
    )
    search = TurnSearch(
        policy, rules, registry.card_index_map, device="cpu",
        config=TurnSearchConfig(budget=16, max_candidates=4, seed=9),
    )

    assert _result_signature(search.choose(game)) == _result_signature(search.choose(variant))


def test_search_prefers_lethal_over_the_policy_prior(card_data):
    cards, registry, rules = card_data
    game = _tactical_game(cards, registry)
    searcher = game.current_player
    opponent = game.get_opponent(searcher)
    _give_base(game, opponent, "Blob Wheel")
    opponent.health = 5
    searcher.combat = 5
    assert_public_knowledge_consistent(game)
    layout, groups = _layout(registry)
    attack_base = groups["attack_base"] + registry.card_index_map["Blob Wheel"]
    attack_player = layout["global_indices"]["attack_player"]
    model = ScriptedPolicy(get_action_space_size(registry.card_names), {attack_base: 5.0})
    search = TurnSearch(model, rules, registry.card_index_map, device="cpu",
                        config=TurnSearchConfig(budget=8, max_candidates=4, seed=1))

    result = search.choose(game)

    assert result.greedy_action == attack_base
    assert result.action == attack_player
    values = {c.action: c.value for c in result.candidates}
    assert values[attack_player] == 1.0
    assert values[attack_base] == 0.0


def test_search_destroys_blocking_outpost_to_reach_lethal(card_data):
    cards, registry, rules = card_data
    game = _tactical_game(cards, registry)
    searcher = game.current_player
    opponent = game.get_opponent(searcher)
    _give_base(game, opponent, "Trading Post")
    opponent.health = 3
    searcher.combat = 7
    assert_public_knowledge_consistent(game)
    _, groups = _layout(registry)
    attack_outpost = groups["attack_base"] + registry.card_index_map["Trading Post"]
    model = ScriptedPolicy(get_action_space_size(registry.card_names), {END_TURN_INDEX: 5.0})
    search = TurnSearch(model, rules, registry.card_index_map, device="cpu",
                        config=TurnSearchConfig(budget=8, max_candidates=4, seed=2))

    result = search.choose(game)

    assert result.greedy_action == END_TURN_INDEX
    assert result.action == attack_outpost
    assert {c.action: c.value for c in result.candidates}[attack_outpost] == 1.0


def _place_in_trade_row(game: Game, names: tuple[str, ...]) -> None:
    """Swap supply cards so every named card is visible in the trade row."""
    for name in names:
        if any(card.name == name for card in game.trade_row):
            continue
        slot = next(
            index for index, card in enumerate(game.trade_row)
            if card.name not in names
        )
        position = next(
            index for index, card in enumerate(game.trade_deck)
            if card.name == name
        )
        game.trade_row[slot], game.trade_deck[position] = (
            game.trade_deck[position], game.trade_row[slot]
        )


def test_search_buys_the_card_the_critic_values(card_data):
    cards, registry, rules = card_data
    game = _tactical_game(cards, registry)
    searcher = game.current_player
    _place_in_trade_row(game, ("Ram", "Supply Bot"))
    searcher.trade = 3
    assert_public_knowledge_consistent(game)
    names = registry.card_names
    _, groups = _layout(registry)
    buy_ram = groups["buy_card"] + registry.card_index_map["Ram"]
    buy_supply_bot = groups["buy_card"] + registry.card_index_map["Supply Bot"]
    pieces, _ = unpack_state(torch.arange(get_state_size(names), dtype=torch.float32), len(names), 0)
    ram_in_opponent_discard = int(pieces["opp_disc"][0, registry.card_index_map["Ram"]].item())

    def critic(states):
        return -8.0 * states[:, ram_in_opponent_discard]

    model = ScriptedPolicy(get_action_space_size(names), {buy_supply_bot: 1.0}, critic)
    search = TurnSearch(model, rules, registry.card_index_map, device="cpu",
                        config=TurnSearchConfig(budget=16, max_candidates=4, seed=3))

    result = search.choose(game)

    assert result.greedy_action == buy_supply_bot
    assert result.action == buy_ram
    values = {c.action: c.value for c in result.candidates}
    assert values[buy_ram] == pytest.approx(1.0)
    assert values[buy_supply_bot] == pytest.approx(0.0)


def test_rows_passing_on_the_last_allowed_step_use_the_opponent_value(card_data):
    cards, registry, rules = card_data
    game = _tactical_game(cards, registry)
    game.current_player.trade = 3
    assert_public_knowledge_consistent(game)
    names = registry.card_names
    model = ScriptedPolicy(
        get_action_space_size(names), {END_TURN_INDEX: 5.0},
        lambda states: torch.full((states.shape[0],), 0.8),
    )
    search = TurnSearch(model, rules, registry.card_index_map, device="cpu",
                        config=TurnSearchConfig(budget=16, max_candidates=4, seed=1, max_turn_steps=1))

    result = search.choose(game)

    assert len(result.candidates) > 1
    assert all(c.value == pytest.approx(-0.8) for c in result.candidates)
    assert result.action == END_TURN_INDEX


def test_rows_still_in_turn_at_the_step_cap_use_the_searcher_value(card_data):
    cards, registry, rules = card_data
    game = _tactical_game(cards, registry)
    _place_in_trade_row(game, ("Trade Bot", "Imperial Fighter"))
    game.current_player.trade = 2
    assert_public_knowledge_consistent(game)
    names = registry.card_names
    _, groups = _layout(registry)
    buy_trade_bot = groups["buy_card"] + registry.card_index_map["Trade Bot"]
    buy_fighter = groups["buy_card"] + registry.card_index_map["Imperial Fighter"]
    model = ScriptedPolicy(
        get_action_space_size(names), {buy_trade_bot: 5.0, buy_fighter: 4.0},
        lambda states: torch.full((states.shape[0],), 0.8),
    )
    search = TurnSearch(model, rules, registry.card_index_map, device="cpu",
                        config=TurnSearchConfig(budget=8, max_candidates=2, seed=1, max_turn_steps=1))

    result = search.choose(game)

    values = {c.action: c.value for c in result.candidates}
    # Each purchase leaves one trade for the other card, bought on the only
    # allowed continuation step; the turn is still the searcher's at the cap.
    assert values[buy_trade_bot] == pytest.approx(0.8)
    assert values[buy_fighter] == pytest.approx(0.8)


def test_search_agent_completes_games_with_legal_actions(card_data, policy):
    cards, registry, rules = card_data
    search = TurnSearch(policy, rules, registry.card_index_map, device="cpu",
                        config=TurnSearchConfig(budget=4, max_candidates=3, seed=5))
    for seed in (41, 42):
        random.seed(seed)
        game = Game(cards, card_names=registry.card_names, card_index_map=registry.card_index_map)
        agent = SearchAgent("Search", search)
        game.add_player("Search", agent)
        game.add_player("Random", RandomAgent("Random"))
        game.start_game(first_player_name="Search" if seed % 2 else "Random")
        for _ in range(5000):
            if game.is_game_over:
                break
            game.apply_decision(game.current_player.make_decision(game))
        assert game.is_game_over
        assert agent.statistics.decisions > 0
        assert agent.statistics.rollouts <= 4 * agent.statistics.decisions


def test_forced_decisions_skip_rollouts(card_data):
    cards, registry, rules = card_data
    game = _tactical_game(cards, registry)
    for _ in range(40):
        context = build_action_context(
            game, game.current_player, registry.card_index_map,
            get_action_space_size(registry.card_names),
        )
        mask = context.mask.copy()
        if context.has_meaningful:
            mask[END_TURN_INDEX] = False
        if mask.sum() == 1:
            break
        game.apply_decision(get_available_actions(game, game.current_player)[-1])
    else:
        pytest.skip("no forced decision reached")
    model = ScriptedPolicy(get_action_space_size(registry.card_names), {})
    result = TurnSearch(model, rules, registry.card_index_map, device="cpu").choose(game)
    assert result.forced
    assert result.rollouts == 0
    assert result.candidates == ()


def _choice_games(cards, registry, specs):
    """Games advanced to multi-action decisions of the requested acting players."""
    action_dim = get_action_space_size(registry.card_names)
    games = []
    for seed, steps in specs:
        game = _make_game(cards, registry, seed=seed)
        _advance(game, steps)
        viewer = len(games) % 2
        for _ in range(2000):
            if game.is_game_over:
                break
            context = build_action_context(
                game, game.current_player, registry.card_index_map, action_dim
            )
            mask = context.mask.copy()
            if context.has_meaningful:
                mask[END_TURN_INDEX] = False
            if game.players.index(game.current_player) == viewer and mask.sum() >= 3:
                break
            game.apply_decision(game.current_player.make_decision(game))
        else:
            pytest.skip("no multi-action decision reached for the requested viewer")
        if game.is_game_over:
            pytest.skip("game ended before a multi-action decision")
        games.append(game)
    assert len({game.players.index(game.current_player) for game in games}) == min(2, len(games))
    return games


def _exact_value_policy(action_dim: int):
    """Scripted policy whose value is an elementwise function of the observation."""
    return ScriptedPolicy(
        action_dim, {END_TURN_INDEX: 0.5},
        lambda states: states[:, 3] - 2.0 * states[:, 60] + states[:, 61],
    )


def _assert_results_equal(batched, individual, *, exact: bool):
    assert len(batched) == len(individual)
    for first, second in zip(batched, individual):
        assert first.action == second.action
        assert first.greedy_action == second.greedy_action
        assert first.forced == second.forced
        assert first.rollouts == second.rollouts
        assert first.fingerprint == second.fingerprint
        assert [c.action for c in first.candidates] == [c.action for c in second.candidates]
        assert [c.rollouts for c in first.candidates] == [c.rollouts for c in second.candidates]
        for a, b in zip(first.candidates, second.candidates):
            if exact:
                assert a.value == b.value and a.score == b.score
            else:
                assert a.value == pytest.approx(b.value, abs=1e-9)
                assert a.score == pytest.approx(b.score, abs=1e-9)


def test_choose_many_equals_individual_choose(card_data, policy):
    cards, registry, rules = card_data
    games = _choice_games(cards, registry, ((51, 90), (52, 141), (53, 60), (54, 173), (55, 30)))
    action_dim = get_action_space_size(registry.card_names)
    config = TurnSearchConfig(budget=16, max_candidates=5, seed=7)
    for model, exact in (
        (_exact_value_policy(action_dim), True),
        (copy.deepcopy(policy).double(), False),
    ):
        search = TurnSearch(model, rules, registry.card_index_map, device="cpu", config=config)
        batched = search.choose_many(games)
        individual = [search.choose(game) for game in games]
        assert any(not result.forced for result in batched)
        _assert_results_equal(batched, individual, exact=exact)


def test_choose_many_applies_per_decision_budgets(card_data, policy):
    cards, registry, rules = card_data
    games = _choice_games(cards, registry, ((61, 90), (62, 141), (63, 75)))
    budgets = (3, 24, 9)
    model = copy.deepcopy(policy).double()
    batched = TurnSearch(
        model, rules, registry.card_index_map, device="cpu",
        config=TurnSearchConfig(budget=64, max_candidates=6, seed=2),
    ).choose_many(games, budgets=budgets)
    individual = [
        TurnSearch(
            model, rules, registry.card_index_map, device="cpu",
            config=TurnSearchConfig(budget=budget, max_candidates=6, seed=2),
        ).choose(game)
        for game, budget in zip(games, budgets)
    ]
    _assert_results_equal(batched, individual, exact=False)
    assert all(result.rollouts <= budget for result, budget in zip(batched, budgets))


def test_choose_many_validates_inputs(card_data):
    cards, registry, rules = card_data
    games = _choice_games(cards, registry, ((71, 90), (72, 141)))
    model = _exact_value_policy(get_action_space_size(registry.card_names))
    search = TurnSearch(model, rules, registry.card_index_map, device="cpu")
    assert search.choose_many([]) == []
    with pytest.raises(ValueError, match="positive budget"):
        search.choose_many(games, budgets=[4, 0])
    with pytest.raises(ValueError, match="action context"):
        search.choose_many(games, contexts=[])
    swapped = [
        build_action_context(game, game.current_player, registry.card_index_map, search.action_dim)
        for game in reversed(games)
    ]
    if not np.array_equal(swapped[0].mask, swapped[1].mask):
        with pytest.raises(RuntimeError, match="legality differs"):
            search.choose_many(games, contexts=swapped)


def test_choose_many_makes_no_deep_copies_and_counts_work(card_data, monkeypatch):
    cards, registry, rules = card_data
    games = _choice_games(cards, registry, ((81, 90), (82, 141)))
    search = TurnSearch(
        _exact_value_policy(get_action_space_size(registry.card_names)),
        rules, registry.card_index_map, device="cpu",
        config=TurnSearchConfig(budget=8, max_candidates=4, seed=1),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("search must not deep copy games")

    monkeypatch.setattr(copy, "deepcopy", forbidden)
    results = search.choose_many(games)
    timings = search.timings
    assert timings.calls == 1 and timings.decisions == 2
    assert timings.searched == sum(not result.forced for result in results)
    assert timings.rollout_steps > 0 and timings.rounds > 0


def test_multi_block_dealing_equals_individual_dealing(card_data):
    from src.search.vector_determinization import deal_world_hands
    from src.search.vector_root import SearchRootBuilder
    from src.vector_engine.state import take_rows

    cards, registry, rules = card_data
    games = []
    for seed, steps in ((91, 70), (92, 131), (93, 190)):
        game = _make_game(cards, registry, seed=seed)
        _advance(game, steps)
        games.append(game)
    roots_state, roots = SearchRootBuilder(rules).build(games)
    sizes = [6, 4, 9]
    worlds = np.concatenate([np.tile(np.arange(3, 3 + n // 3 + 1), 3)[:n] for n in sizes])
    population = take_rows(roots_state, np.repeat(np.arange(3), sizes))
    stops = np.cumsum(sizes)
    deal_world_hands(
        population, starts=stops - sizes, stops=stops,
        viewers=[root.viewer for root in roots],
        known_hands=np.stack([root.known_hand for root in roots]),
        base_seeds=[11, 12, 13], worlds=worlds,
    )
    for block, (start, stop) in enumerate(zip(stops - sizes, stops)):
        alone = take_rows(roots_state, np.full(sizes[block], block))
        sample_world_hands(
            alone, viewer=roots[block].viewer, known_hand=roots[block].known_hand,
            worlds=worlds[start:stop], base_seed=11 + block,
        )
        assert np.array_equal(population.zones[start:stop], alone.zones)
        assert np.array_equal(population.rng_state[start:stop], alone.rng_state)


def test_dealt_hands_are_uniform_over_unseen_cards(card_data):
    cards, registry, rules = card_data
    game = _make_game(cards, registry, seed=94)
    _advance(game, 170)
    information = project_information_set(game)
    viewer = information.viewer_index
    opponent = 1 - viewer
    root = from_games([information.canonical_game], rules, training_players=[viewer])
    known = known_hand_counts(information, opponent, rules.num_cards)
    unseen = (root.zones[0, opponent, ZONE_HAND] + root.zones[0, opponent, ZONE_DECK]).astype(np.int64)
    top = np.bincount(
        root.ordered_top[0, opponent, : root.ordered_top_count[0, opponent]].astype(np.int64),
        minlength=rules.num_cards,
    )
    pool = unseen - known - top
    k = int(root.zones[0, opponent, ZONE_HAND].sum() - known.sum())
    m = int(pool.sum())
    if k == 0 or k == m or np.count_nonzero(pool) < 2:
        pytest.skip("position has no hidden hand choice")
    worlds = 4000
    state = tile_state(root, worlds)
    sample_world_hands(state, viewer=viewer, known_hand=known, worlds=np.arange(worlds), base_seed=5)
    dealt = state.zones[:, opponent, ZONE_HAND].astype(np.int64) - known
    assert (dealt.sum(axis=1) == k).all()
    for card in np.flatnonzero(pool):
        p = pool[card] / m
        expected = k * p
        variance = k * p * (1 - p) * (m - k) / (m - 1)
        standard_error = np.sqrt(variance / worlds)
        assert abs(dealt[:, card].mean() - expected) < 5 * standard_error + 1e-9, card


def test_elapsed_time_is_attributed_by_rollouts_within_a_batch(card_data):
    cards, registry, rules = card_data
    games = _choice_games(cards, registry, ((101, 90), (102, 141)))
    search = TurnSearch(
        _exact_value_policy(get_action_space_size(registry.card_names)),
        rules, registry.card_index_map, device="cpu",
        config=TurnSearchConfig(budget=64, max_candidates=4, seed=1),
    )
    results = search.choose_many(games, budgets=[2, 48])
    if any(result.forced for result in results):
        pytest.skip("fixture reached a forced decision")
    small, large = results
    assert small.rollouts < large.rollouts
    assert large.elapsed_s > small.elapsed_s
    assert small.elapsed_s + large.elapsed_s == pytest.approx(search.timings.total_s, rel=1e-6)
