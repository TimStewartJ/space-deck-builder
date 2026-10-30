"""End-to-end gates for the batched game runner and search uplift arena."""
from __future__ import annotations

import copy
import random

import pytest
import torch

from src.ai.heuristic_agent import HeuristicAgent
from src.ai.policy_agent import PolicyAgent
from src.config import DataConfig, ModelConfig, load_checkpoint
from src.encoding.action_encoder import get_action_space_size
from src.encoding.state_encoder import get_state_size
from src.engine.game import Game
from src.ppo.ppo_actor_critic import PPOActorCritic
from src.search.agent import SearchAgent
from src.search.batched_games import (
    AgentSeat,
    BatchedGameRunner,
    GameSpec,
    PolicySeat,
    SearchSeat,
)
from src.search.turn_search import TurnSearch, TurnSearchConfig
from src.search.uplift import BASELINE_NAME, SEARCH_NAME, UpliftConfig, _Arena, game_seed
from src.utils.logger import set_disabled
from src.vector_engine.rules import compile_rules


@pytest.fixture(scope="module")
def setup(policy_checkpoint):
    set_disabled(True)
    data = DataConfig()
    cards = data.load_cards()
    registry = data.build_registry(cards)
    rules = compile_rules(cards, registry)
    names = registry.card_names
    checkpoint = load_checkpoint(policy_checkpoint, map_location="cpu")
    model = PPOActorCritic(
        get_state_size(names), get_action_space_size(names), len(names),
        model_config=ModelConfig.from_dict(checkpoint["config"]["model"]),
        card_registry=registry,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return cards, registry, rules, model.double().eval()


def _sequential_row(setup, config: UpliftConfig, budget: int, index: int) -> dict:
    """Play one uplift game the pre-batching way, with scalar agents."""
    cards, registry, rules, model = setup
    game = Game(cards, card_names=registry.card_names, card_index_map=registry.card_index_map)
    game.rng = random.Random(game_seed(config.seed, index))
    if budget > 0:
        search = TurnSearch(
            model, rules, registry.card_index_map, device="cpu",
            config=TurnSearchConfig(budget=budget, max_candidates=config.max_candidates, seed=config.seed),
        )
        searcher = SearchAgent(SEARCH_NAME, search)
    else:
        searcher = PolicyAgent(SEARCH_NAME, model, registry.card_names, registry.card_index_map, temperature=0.0)
    baseline = (
        HeuristicAgent(BASELINE_NAME)
        if config.baseline == "heuristic"
        else PolicyAgent(BASELINE_NAME, model, registry.card_names, registry.card_index_map, temperature=0.0)
    )
    game.add_player(SEARCH_NAME, searcher)
    game.add_player(BASELINE_NAME, baseline)
    game.start_game(first_player_name=SEARCH_NAME if index % 2 == 0 else BASELINE_NAME)
    steps = 0
    while not game.is_game_over:
        game.apply_decision(game.current_player.make_decision(game))
        steps += 1
    statistics = searcher.statistics.to_dict() if budget > 0 else None
    return {
        "search_won": game.get_winner() == SEARCH_NAME,
        "turns": int(game.stats.total_turns),
        "decisions": steps,
        "search": statistics,
    }


def _batched_rows(setup, config: UpliftConfig, tasks) -> dict:
    arena = _Arena(config, model=setup[3])
    rows = []
    arena.play_many(tasks, rows.append)
    assert sorted((row["budget"], row["index"]) for row in rows) == sorted(tasks)
    return {(row["budget"], row["index"]): row for row in rows}


def _assert_rows_match(setup, config: UpliftConfig, tasks) -> None:
    batched = _batched_rows(setup, config, tasks)
    for budget, index in tasks:
        expected = _sequential_row(setup, config, budget, index)
        row = batched[(budget, index)]
        assert row["search_first"] == (index % 2 == 0)
        assert row["search_won"] == expected["search_won"], (budget, index)
        assert row["turns"] == expected["turns"], (budget, index)
        assert row["decisions"] == expected["decisions"], (budget, index)
        if expected["search"] is not None:
            for key in ("decisions", "forced", "deviations", "rollouts"):
                assert row["search"][key] == expected["search"][key], (budget, index, key)
        else:
            assert row["search"]["decisions"] == 0


def test_batched_arena_reproduces_sequential_games_against_greedy(setup, policy_checkpoint):
    config = UpliftConfig(
        model_path=policy_checkpoint, budgets=(0, 2), games=4, baseline="greedy",
        max_candidates=3, seed=11, device="cpu", batch_size=3,
    )
    _assert_rows_match(setup, config, [(budget, index) for index in range(4) for budget in (0, 2)])


def test_batched_arena_reproduces_sequential_games_against_heuristic(setup, policy_checkpoint):
    config = UpliftConfig(
        model_path=policy_checkpoint, budgets=(3,), games=2, baseline="heuristic",
        max_candidates=3, seed=12, device="cpu", batch_size=2,
    )
    _assert_rows_match(setup, config, [(3, 0), (3, 1)])


def test_runner_outcomes_do_not_depend_on_batch_size(setup):
    cards, registry, rules, model = setup
    search = TurnSearch(model, rules, registry.card_index_map, device="cpu",
                        config=TurnSearchConfig(budget=2, max_candidates=3, seed=4))
    specs = [
        GameSpec(
            key=index,
            seed=game_seed(9, index),
            players=(
                ("A", SearchSeat(2) if index == 0 else PolicySeat(0.0)),
                ("B", PolicySeat(1.0) if index % 2 else AgentSeat(HeuristicAgent)),
            ),
            first_player="A" if index % 2 == 0 else "B",
        )
        for index in range(4)
    ]
    outcomes = {}
    for batch_size in (1, 4):
        runner = BatchedGameRunner(
            cards=cards, registry=registry, model=model, search=search,
            device="cpu", batch_size=batch_size,
        )
        finished = []
        runner.run(specs, finished.append)
        outcomes[batch_size] = {
            outcome.key: (
                outcome.winner, outcome.turns, outcome.decisions,
                {name: stats.to_dict()["rollouts"] for name, stats in outcome.search.items()},
            )
            for outcome in finished
        }
    assert outcomes[1] == outcomes[4]
    assert set(outcomes[1]) == {0, 1, 2, 3}


def test_runner_rejects_invalid_specs(setup):
    cards, registry, _, model = setup
    runner = BatchedGameRunner(
        cards=cards, registry=registry, model=model, search=None, device="cpu", batch_size=2,
    )
    with pytest.raises(ValueError, match="two distinct named seats"):
        runner.run(
            [GameSpec(key=0, seed=1, players=(("A", PolicySeat()), ("A", PolicySeat())), first_player="A")],
            lambda outcome: None,
        )
    with pytest.raises(ValueError, match="require a TurnSearch"):
        runner.run(
            [GameSpec(key=0, seed=1, players=(("A", SearchSeat(4)), ("B", PolicySeat())), first_player="A")],
            lambda outcome: None,
        )
    with pytest.raises(ValueError, match="batch_size"):
        BatchedGameRunner(cards=cards, registry=registry, model=model, search=None, device="cpu", batch_size=0)
