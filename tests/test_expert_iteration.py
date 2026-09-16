"""Gates for search self-play targets, shards, and distillation."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from src.__main__ import parse_cli_args
from src.ai.random_agent import RandomAgent
from src.config import DataConfig, ModelConfig, load_checkpoint
from src.encoding.action_context import build_action_context
from src.encoding.action_encoder import END_TURN_INDEX, get_action_space_size
from src.encoding.state_encoder import encode_state, get_state_size
from src.engine.game import Game
from src.ppo.ppo_actor_critic import PPOActorCritic
from src.search.distill import DistillConfig, distill, policy_value_losses
from src.search.selfplay_data import (
    SelfPlayConfig,
    completed_games,
    generate_selfplay,
    load_selfplay_data,
)
from src.search.turn_search import TurnSearch, TurnSearchConfig
from src.utils.logger import set_disabled
from src.vector_engine.rules import compile_rules

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO / "results" / "optimized_selfplay_baseline" / "checkpoints" / "selfplay_seed0.pth"


@pytest.fixture(scope="module")
def setup():
    set_disabled(True)
    data = DataConfig()
    cards = data.load_cards()
    registry = data.build_registry(cards)
    rules = compile_rules(cards, registry)
    names = registry.card_names
    checkpoint = load_checkpoint(str(CHECKPOINT), map_location="cpu")
    model = PPOActorCritic(
        get_state_size(names), get_action_space_size(names), len(names),
        model_config=ModelConfig.from_dict(checkpoint["config"]["model"]),
        card_registry=registry,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return cards, registry, rules, model.eval()


def _search(setup, **overrides):
    _, registry, rules, model = setup
    config = TurnSearchConfig(**{"budget": 12, "max_candidates": 4, "seed": 3, **overrides})
    return TurnSearch(model, rules, registry.card_index_map, device="cpu", config=config)


def _positions(setup, count: int, seed: int):
    cards, registry, _, _ = setup
    games = []
    index = 0
    while len(games) < count:
        random.seed(seed + index)
        game = Game(cards, card_names=registry.card_names, card_index_map=registry.card_index_map)
        game.add_player("P0", RandomAgent("P0"))
        game.add_player("P1", RandomAgent("P1"))
        game.start_game(first_player_name="P0" if index % 2 == 0 else "P1")
        for _ in range(random.Random(index).randint(5, 160)):
            if game.is_game_over:
                break
            game.apply_decision(game.current_player.make_decision(game))
        index += 1
        if not game.is_game_over:
            games.append(game)
    return games


def test_improved_policy_follows_completed_q(setup):
    search = _search(setup)
    logits = np.full(8, -np.inf)
    logits[[1, 3, 4, 6]] = [2.0, 1.0, 0.5, -1.0]
    mask = np.isfinite(logits)
    totals = {1: 0.6, 3: 2.0}
    counts = {1: 2, 3: 4}
    root_value = -0.2

    policy = search.improved_policy(logits, mask, totals, counts, root_value)

    assert policy.sum() == pytest.approx(1.0, abs=1e-6)
    assert (policy[~mask] == 0).all()
    prior = np.exp(logits[mask] - logits[mask].max())
    prior /= prior.sum()
    prior = dict(zip(np.flatnonzero(mask), prior))
    q = {1: 0.3, 3: 0.5}
    visits = 6
    mixed = (root_value + visits * (prior[1] * q[1] + prior[3] * q[3]) / (prior[1] + prior[3])) / (1 + visits)
    scale = 50 + 4
    expected = {action: logits[action] + scale * ((q.get(action, mixed)) + 1) / 2 for action in (1, 3, 4, 6)}
    top = max(expected.values())
    weights = {action: np.exp(value - top) for action, value in expected.items()}
    norm = sum(weights.values())
    for action, weight in weights.items():
        assert policy[action] == pytest.approx(weight / norm, rel=1e-5, abs=1e-12)
    assert policy[4] / policy[6] == pytest.approx(np.exp(1.5), rel=1e-5)
    assert int(np.argmax(policy)) == 3
    with pytest.raises(ValueError):
        search.improved_policy(logits, mask, {}, {}, 0.0)


def test_targets_match_live_observations_and_do_not_change_decisions(setup):
    _, registry, _, _ = setup
    games = _positions(setup, 6, seed=900)
    search = _search(setup)
    plain = search.choose_many(games)
    targeted = search.choose_many(games, return_targets=True)
    action_dim = get_action_space_size(registry.card_names)
    assert any(result.forced for result in targeted) or not any(result.forced for result in plain)
    for game, a, b in zip(games, plain, targeted):
        assert a.targets is None
        assert (a.action, a.greedy_action, a.forced, a.rollouts) == (b.action, b.greedy_action, b.forced, b.rollouts)
        assert [c.action for c in a.candidates] == [c.action for c in b.candidates]
        context = build_action_context(game, game.current_player, registry.card_index_map, action_dim)
        observation = encode_state(
            game, is_current_player_training=True, cards=registry.card_names,
            card_index_map=registry.card_index_map, can_buy=context.can_buy,
            has_actions=context.has_meaningful, return_numpy=True,
        )
        assert np.array_equal(b.targets.observation, observation)
        effective = context.mask.copy()
        if context.has_meaningful:
            effective[END_TURN_INDEX] = False
        assert np.array_equal(b.targets.mask, effective)
        assert b.targets.policy.sum() == pytest.approx(1.0, abs=1e-5)
        assert (b.targets.policy[~effective] == 0).all()
        if b.forced:
            assert b.targets.policy[b.action] == 1.0
        assert np.isfinite(b.targets.root_value)


def test_policy_value_losses_mask_illegal_actions():
    logits = torch.tensor([[0.0, 5.0, 1.0], [3.0, 0.0, 0.0]], requires_grad=True)
    values = torch.tensor([0.5, -0.5], requires_grad=True)
    masks = torch.tensor([[True, False, True], [True, False, False]])
    policy = torch.tensor([[0.25, 0.0, 0.75], [1.0, 0.0, 0.0]])
    outcome = torch.tensor([1.0, -1.0])

    cross_entropy, squared = policy_value_losses(logits, values, masks, policy, outcome)

    expected = -(0.25 * np.log(1 / (1 + np.e)) + 0.75 * np.log(np.e / (1 + np.e)))
    assert float(cross_entropy[0].detach()) == pytest.approx(expected, rel=1e-5)
    assert float(cross_entropy[1].detach()) == pytest.approx(0.0, abs=1e-7)
    assert torch.allclose(squared.detach(), torch.tensor([0.25, 0.25]))
    (cross_entropy.sum() + squared.sum()).backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(values.grad).all()


@pytest.fixture(scope="module")
def selfplay_dir(tmp_path_factory, setup):
    directory = tmp_path_factory.mktemp("selfplay")
    config = SelfPlayConfig(
        model_path=str(CHECKPOINT), games=4, budget=2, max_candidates=2, seed=5,
        device="cpu", workers=1, numba_threads=2, batch_size=4, shard_positions=200,
    )
    summary = generate_selfplay(config, directory, log_fn=lambda *_: None, model=setup[3])
    return directory, config, summary


def test_selfplay_shards_record_complete_games_with_outcomes(selfplay_dir):
    directory, config, summary = selfplay_dir
    data = load_selfplay_data(directory)
    assert summary["positions"] == data.positions
    assert completed_games(directory) == set(range(config.games))
    assert len(list(directory.glob("shard_*.npz"))) > 1
    assert set(np.unique(data.outcome)) <= {-1.0, 1.0}
    for game in range(config.games):
        rows = np.flatnonzero(data.game == game)
        steps = data.step[rows]
        assert (np.diff(steps) > 0).all()
        assert steps[0] == 0
        outcomes = {int(player): float(data.outcome[rows][data.player[rows] == player][0]) for player in np.unique(data.player[rows])}
        assert sorted(outcomes.values()) == [-1.0, 1.0]
    assert np.allclose(data.policy.sum(axis=1), 1.0, atol=1e-4)
    assert (data.policy[data.forced].max(axis=1) == 1.0).all()


def test_selfplay_generation_resumes_and_reproduces_games(selfplay_dir, setup):
    directory, config, _ = selfplay_dir
    before = load_selfplay_data(directory)
    logs = []
    generate_selfplay(config, directory, log_fn=logs.append, model=setup[3])
    assert any("remaining=0" in line for line in logs)

    victim = sorted(directory.glob("shard_*.npz"))[0]
    with np.load(victim) as shard:
        removed = set(int(game) for game in shard["games"])
    victim.unlink()
    logs.clear()
    generate_selfplay(config, directory, log_fn=logs.append, model=setup[3])
    assert any(f"remaining={len(removed)}" in line for line in logs)
    after = load_selfplay_data(directory)
    for game in removed:
        a = np.flatnonzero(before.game == game)
        b = np.flatnonzero(after.game == game)
        assert np.array_equal(before.action[a], after.action[b])
        assert np.array_equal(before.outcome[a], after.outcome[b])
        assert np.allclose(before.policy[a], after.policy[b], atol=1e-6)

    with pytest.raises(ValueError, match="different configuration"):
        generate_selfplay(
            SelfPlayConfig(**{**config.__dict__, "seed": 6}), directory,
            log_fn=lambda *_: None, model=setup[3],
        )
    with pytest.raises(ValueError, match="outside"):
        generate_selfplay(
            SelfPlayConfig(**{**config.__dict__, "games": 2}), directory,
            log_fn=lambda *_: None, model=setup[3],
        )

    logs.clear()
    extended = SelfPlayConfig(**{**config.__dict__, "games": config.games + 2})
    generate_selfplay(extended, directory, log_fn=logs.append, model=setup[3])
    assert any("remaining=2" in line for line in logs)
    assert completed_games(directory) == set(range(config.games + 2))


def test_distill_writes_loadable_checkpoint_and_history(selfplay_dir, tmp_path):
    directory, config, _ = selfplay_dir
    output = tmp_path / "student.pth"
    config = DistillConfig(
        model_path=str(CHECKPOINT), data_dir=str(directory), output_path=str(output),
        lr=1e-3, batch_size=64, max_epochs=2, patience=2, validation_modulus=2, device="cpu",
    )
    summary = distill(config, log_fn=lambda *_: None)

    history = summary["history"]
    assert history[0]["validation"]["kl_initial"] == pytest.approx(0.0, abs=1e-6)
    assert 0 <= summary["best_epoch"] <= 2
    assert len(history) >= 2
    trained = history[1]["validation"]
    assert trained["kl_initial"] > 0
    assert 0 <= trained["search_agreement"] <= 1
    checkpoint = load_checkpoint(str(output), map_location="cpu")
    assert checkpoint["distill"]["best_epoch"] == summary["best_epoch"]
    set_disabled(True)
    data = DataConfig()
    registry = data.build_registry(data.load_cards())
    names = registry.card_names
    model = PPOActorCritic(
        get_state_size(names), get_action_space_size(names), len(names),
        model_config=ModelConfig.from_dict(checkpoint["config"]["model"]), card_registry=registry,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    saved = json.loads((tmp_path / "student.pth.history.json").read_text(encoding="utf-8"))
    assert saved["best_epoch"] == summary["best_epoch"]


def test_new_cli_commands_parse():
    args = parse_cli_args(["search-selfplay", "--model", "m.pth", "--games", "8", "--output-dir", "out"])
    assert args.budget == 64 and args.workers == 2 and args.batch_size == 1024
    args = parse_cli_args(["distill", "--model", "m.pth", "--data", "d", "--output", "o.pth"])
    assert args.lr == 1e-4 and args.epochs == 8 and args.validation_modulus == 20
