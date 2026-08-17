"""End-to-end PPO rollout collection through the full vector engine."""
from __future__ import annotations

import numpy as np
import numba
import pytest
import torch

from src.config import DataConfig, ModelConfig, PPOConfig, RunConfig
from src.encoding.action_encoder import get_action_space_size
from src.encoding.state_encoder import get_state_size
from src.ppo.ppo_actor_critic import PPOActorCritic
from src.ppo.ppo_trainer import _make_runner
from src.ppo.opponent_pool import OpponentPool
from src.ppo.vector_batch_runner import (
    VectorBatchRunner,
    _compute_linked_gae,
    _select_random_legal,
    initialize_training_state,
)
from src.ppo.rollout_buffer import RolloutBuffer
from src.vector_engine.kernels import ATTACK_PLAYER_INDEX, PLAY_OFFSET
from src.vector_engine.state import ZONE_DECK, ZONE_HAND


@pytest.fixture(scope="module")
def setup():
    data = DataConfig()
    cards = data.load_cards()
    registry = data.build_registry(cards)
    return data, cards, registry


def make_model(registry):
    return PPOActorCritic(
        state_dim=get_state_size(registry.card_names),
        action_dim=get_action_space_size(registry.card_names),
        num_cards=registry.num_cards,
        model_config=ModelConfig(),
    )


def test_vector_initialization_preserves_physical_starter_decks(setup):
    _, cards, registry = setup
    state, rules = initialize_training_state(
        32, cards, registry, seed=7
    )
    scout = registry.card_index_map["Scout"]
    viper = registry.card_index_map["Viper"]
    for game in range(32):
        for player in range(2):
            assert (
                state.zones[game, player, ZONE_HAND, scout]
                + state.zones[game, player, ZONE_DECK, scout]
                == 8
            )
            assert (
                state.zones[game, player, ZONE_HAND, viper]
                + state.zones[game, player, ZONE_DECK, viper]
                == 2
            )
    assert rules.supported.all()
    assert (state.trade_row >= 0).all()
    assert state.explorer_count.tolist() == [10] * 32


def test_vector_runner_accepts_all_fixed_opponent_types(setup):
    _, cards, registry = setup
    runner = VectorBatchRunner(
        model=make_model(registry),
        cards=cards,
        registry=registry,
        device=torch.device("cpu"),
        ppo_config=PPOConfig(),
        opponent_spec="random:0.2,heuristic:0.5,simple:0.3",
    )
    assert runner.opponent_entries == [
        ("heuristic", 0.5),
        ("random", 0.2),
        ("simple", 0.3),
    ]


def test_vector_runner_caps_threads_to_numba_configuration(setup, monkeypatch):
    _, cards, registry = setup
    monkeypatch.setattr(numba.config, "NUMBA_NUM_THREADS", 2)
    runner = VectorBatchRunner(
        model=make_model(registry),
        cards=cards,
        registry=registry,
        device=torch.device("cpu"),
        ppo_config=PPOConfig(),
        opponent_spec="random",
        num_threads=4,
    )
    assert runner.num_threads == 2


def test_vector_runner_rejects_ignored_concurrency_setting(setup):
    _, cards, registry = setup
    runner = VectorBatchRunner(
        model=make_model(registry),
        cards=cards,
        registry=registry,
        device=torch.device("cpu"),
        ppo_config=PPOConfig(),
        opponent_spec="random",
        num_concurrent=4,
    )
    with pytest.raises(ValueError, match="must be omitted or equal"):
        runner.run_episodes(8)


def test_vector_random_opponent_matches_scalar_priority_policy():
    weights = np.zeros((3, get_action_space_size([""] * 49)), dtype=np.int16)
    weights[:, 1] = 1
    weights[0, PLAY_OFFSET + 2] = 1
    weights[0, PLAY_OFFSET + 3] = 1
    weights[1, ATTACK_PLAYER_INDEX] = 1
    weights[2, 2] = 1
    actions = np.zeros(3, dtype=np.int32)
    variants = np.zeros(3, dtype=np.int16)
    rng = np.asarray([1, 2, 3], dtype=np.int64)
    _select_random_legal(
        weights, rng, np.arange(3, dtype=np.int64), actions, variants
    )
    assert actions[0] in (PLAY_OFFSET + 2, PLAY_OFFSET + 3)
    assert actions[1] == ATTACK_PLAYER_INDEX
    assert actions[2] in (1, 2)


def test_vector_random_opponent_weights_duplicate_physical_cards():
    games = 10_000
    weights = np.zeros(
        (games, get_action_space_size([""] * 49)), dtype=np.int16
    )
    weights[:, PLAY_OFFSET + 2] = 4
    weights[:, PLAY_OFFSET + 3] = 1
    actions = np.zeros(games, dtype=np.int32)
    variants = np.zeros(games, dtype=np.int16)
    rng = np.arange(1, games + 1, dtype=np.int64)
    _select_random_legal(
        weights,
        rng,
        np.arange(games, dtype=np.int64),
        actions,
        variants,
    )
    four_copy_share = float((actions == PLAY_OFFSET + 2).mean())
    assert 0.78 < four_copy_share < 0.82


def test_linked_gae_matches_existing_rollout_buffer():
    previous = np.asarray([-1, -1, 0, 1, 2], dtype=np.int64)
    last = np.asarray([4, 3], dtype=np.int64)
    rewards = np.asarray([0, 0, 0, -1, 1], dtype=np.float32)
    dones = np.asarray([False, False, False, True, True])
    values = np.asarray([0.2, -0.1, 0.4, 0.3, 0.5], dtype=np.float32)
    returns = np.empty(5, dtype=np.float32)
    advantages = np.empty(5, dtype=np.float32)
    gamma, lam = 0.99, 0.95
    _compute_linked_gae(
        last,
        previous,
        rewards,
        dones,
        values,
        gamma,
        lam,
        returns,
        advantages,
    )

    for indices in ([0, 2, 4], [1, 3]):
        buffer = RolloutBuffer()
        for index in indices:
            buffer.add(
                np.zeros(1, dtype=np.float32),
                1,
                0.0,
                values[index],
                reward=rewards[index],
                done=bool(dones[index]),
            )
        _, _, _, expected_returns, expected_adv, _ = buffer.finish(
            gamma, lam, torch.device("cpu"), normalize=False
        )
        assert np.allclose(returns[indices], expected_returns.numpy())
        assert np.allclose(advantages[indices], expected_adv.numpy())


def test_vector_runner_collects_complete_ppo_rollouts(setup):
    _, cards, registry = setup
    runner = VectorBatchRunner(
        model=make_model(registry),
        cards=cards,
        registry=registry,
        device=torch.device("cpu"),
        ppo_config=PPOConfig(),
        opponent_spec="random",
        seed=11,
    )
    states, actions, old_lp, returns, advantages, masks = runner.run_episodes(8)
    count = len(states)
    assert count > 0
    assert states.shape == (count, get_state_size(registry.card_names))
    assert masks.shape == (count, get_action_space_size(registry.card_names))
    assert actions.shape == old_lp.shape == returns.shape == advantages.shape
    assert torch.isfinite(states).all()
    assert torch.isfinite(old_lp).all()
    assert torch.isfinite(returns).all()
    assert torch.isfinite(advantages).all()
    assert masks[torch.arange(count), actions].all()
    assert abs(float(advantages.mean())) < 1e-5
    wins, games = runner.opponent_results["Random"]
    assert 0 <= wins <= games
    assert games == 8


def test_vector_runner_collects_weighted_mixed_opponents(setup):
    _, cards, registry = setup
    runner = VectorBatchRunner(
        model=make_model(registry),
        cards=cards,
        registry=registry,
        device=torch.device("cpu"),
        ppo_config=PPOConfig(),
        opponent_spec="random,heuristic,simple",
        seed=17,
    )
    states, actions, old_lp, returns, advantages, masks = runner.run_episodes(24)
    assert len(states) > 0
    assert actions.shape == old_lp.shape == returns.shape == advantages.shape
    assert masks.shape[0] == len(states)
    assert set(runner.opponent_results) == {"Random", "Heuristic", "Simple"}
    assert sum(result[1] for result in runner.opponent_results.values()) == 24
    assert all(result[1] > 0 for result in runner.opponent_results.values())


def test_vector_runner_collects_snapshot_self_play(setup):
    _, cards, registry = setup
    snapshot_model = make_model(registry)
    runner = VectorBatchRunner(
        model=make_model(registry),
        cards=cards,
        registry=registry,
        device=torch.device("cpu"),
        ppo_config=PPOConfig(),
        opponent_spec="random,heuristic,simple",
        seed=23,
        snapshot_state_dicts=[
            (
                "PPO_1",
                snapshot_model.state_dict(),
                ModelConfig(),
            )
        ],
        self_play_ratio=1.0,
        pfsp_weights=[1.0],
    )
    rollout = runner.run_episodes(8)
    assert len(rollout) == 6
    assert len(rollout[0]) > 0
    assert runner.opponent_results["PPO_1"][1] == 8
    assert sum(
        runner.opponent_results[name][1]
        for name in ("Random", "Heuristic", "Simple")
    ) == 0
    assert (runner._last_opponent_kinds == 3).all()


def test_stacked_snapshot_evaluators_are_lazy_and_cached(setup):
    _, cards, registry = setup
    snapshot_model = make_model(registry)
    runner = VectorBatchRunner(
        model=make_model(registry),
        cards=cards,
        registry=registry,
        device=torch.device("cpu"),
        ppo_config=PPOConfig(),
        opponent_spec="random",
        snapshot_state_dicts=[
            ("PPO_2", snapshot_model.state_dict(), ModelConfig())
        ],
        self_play_ratio=0.0,
    )
    assert runner._stacked_snapshot_evaluators == {}
    first = runner._stacked_snapshot_evaluator((0,))
    second = runner._stacked_snapshot_evaluator((0,))
    assert first is second
    assert list(runner._stacked_snapshot_evaluators) == [(0,)]


def test_single_snapshot_uses_joint_policy_evaluator(setup):
    _, cards, registry = setup
    snapshot_model = make_model(registry)
    runner = VectorBatchRunner(
        model=make_model(registry),
        cards=cards,
        registry=registry,
        device=torch.device("cpu"),
        ppo_config=PPOConfig(),
        opponent_spec="random",
        snapshot_state_dicts=[
            ("PPO_2", snapshot_model.state_dict(), ModelConfig())
        ],
        self_play_ratio=1.0,
        pfsp_weights=[1.0],
        seed=29,
    )

    runner.run_episodes(8)

    evaluator = runner._joint_snapshot_evaluators[0]
    assert evaluator.stats["calls"] > 0


def test_trainer_selects_vector_runner(setup):
    data, cards, registry = setup
    model = make_model(registry)

    class Agent:
        simulation_device = torch.device("cpu")

        def __init__(self):
            self.model = model

    runner = _make_runner(
        Agent(),
        cards,
        registry.card_names,
        get_action_space_size(registry.card_names),
        PPOConfig(),
        RunConfig(
            engine="vec",
            episodes=8,
            num_workers=1,
            opponents="random",
        ),
        data,
        lambda: None,
        registry,
    )
    assert isinstance(runner, VectorBatchRunner)


def test_trainer_passes_snapshot_pool_to_vector_runner(setup):
    data, cards, registry = setup
    model = make_model(registry)
    pool = OpponentPool(opponent_spec="random,heuristic,simple")
    pool.add_snapshot(
        model.state_dict(),
        "PPO_4",
        model_config=ModelConfig(),
        update=4,
    )

    class Agent:
        simulation_device = torch.device("cpu")

        def __init__(self):
            self.model = model

    runner = _make_runner(
        Agent(),
        cards,
        registry.card_names,
        get_action_space_size(registry.card_names),
        PPOConfig(),
        RunConfig(
            engine="vec",
            episodes=8,
            num_workers=1,
            opponents="random,heuristic,simple",
            self_play=True,
            self_play_ratio=0.75,
            vec_max_active_snapshots=2,
        ),
        data,
        lambda: None,
        registry,
        pool=pool,
        current_self_play_ratio=0.6,
    )
    assert isinstance(runner, VectorBatchRunner)
    assert runner.self_play_ratio == 0.6
    assert [name for name, _, _ in runner.snapshot_state_dicts] == ["PPO_4"]
    assert runner.pfsp_weights == [1.0]
    assert runner.max_active_snapshots == 2
