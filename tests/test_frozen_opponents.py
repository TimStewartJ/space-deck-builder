"""Frozen checkpoint opponents: configuration, loading, inference, and training."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.__main__ import _run_train, parse_cli_args
from src.config import (
    CHAMPION_CHECKPOINT,
    DataConfig,
    DeviceConfig,
    ModelConfig,
    PPOConfig,
    RunConfig,
    get_training_preset,
    load_checkpoint,
    save_checkpoint,
)
from src.encoding.action_encoder import get_action_space_size
from src.encoding.state_encoder import get_state_size
from src.ppo.opponent_pool import OpponentPool, load_frozen_opponents
from src.ppo.ppo_actor_critic import PPOActorCritic
from src.ppo.vector_batch_runner import VectorBatchRunner
from src.vector_engine.opponents import SNAPSHOT_OFFSET
from src.vector_engine.policy import (
    CurrentSnapshotEvaluator,
    StackedSnapshotEvaluator,
    _snapshot_actions,
    suppress_redundant_end_turn,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def setup():
    data = DataConfig()
    cards = data.load_cards()
    registry = data.build_registry(cards)
    return data, cards, registry


def make_model(registry, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return PPOActorCritic(
        state_dim=get_state_size(registry.card_names),
        action_dim=get_action_space_size(registry.card_names),
        num_cards=registry.num_cards,
        model_config=ModelConfig(),
    )


def save_model(path: Path, model) -> str:
    save_checkpoint(str(path), model.state_dict(), model_config=ModelConfig())
    return str(path)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(engine="vec", frozen_opponents=["a.pth"], self_play=True), "self-play"),
        (dict(engine="object", frozen_opponents=["a.pth"]), "vector engine"),
        (dict(frozen_opponent_ratio=0.5), "need"),
        (dict(frozen_opponent_temperature=0.0), "need"),
        (dict(engine="vec", frozen_opponents=["a.pth"], frozen_opponent_ratio=1.5), r"\[0, 1\]"),
        (dict(engine="vec", frozen_opponents=["a.pth"], frozen_opponent_temperature=-1.0), "non-negative"),
    ],
)
def test_run_config_rejects_invalid_frozen_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RunConfig(**kwargs)


def test_cli_builds_frozen_run_config(monkeypatch):
    captured = {}

    def fake_train(data_cfg, ppo_cfg, run_cfg, dev_cfg, **kwargs):
        captured["run_cfg"] = run_cfg

    monkeypatch.setattr("src.ppo.ppo_trainer.train", fake_train)
    _run_train(parse_cli_args([
        "train", "--engine", "vec", "--frozen-opponents", "a.pth", "b.pth",
        "--frozen-opponent-ratio", "0.25", "--frozen-opponent-temperature", "0",
    ]))

    run_cfg = captured["run_cfg"]
    assert run_cfg.frozen_opponents == ["a.pth", "b.pth"]
    assert run_cfg.frozen_opponent_ratio == 0.25
    assert run_cfg.frozen_opponent_temperature == 0.0
    assert run_cfg.self_play is False


def test_exploiter_preset_trains_only_against_the_frozen_champion(monkeypatch):
    captured = {}

    def fake_train(data_cfg, ppo_cfg, run_cfg, dev_cfg, **kwargs):
        captured.update(kwargs, run_cfg=run_cfg)

    monkeypatch.setattr("src.ppo.ppo_trainer.train", fake_train)
    _run_train(parse_cli_args(["train", "--preset", "exploit-selfplay-seed0"]))

    run_cfg = captured["run_cfg"]
    assert captured["model_path"] == CHAMPION_CHECKPOINT
    assert run_cfg.frozen_opponents == [CHAMPION_CHECKPOINT]
    assert run_cfg.frozen_opponent_ratio == 1.0
    assert run_cfg.frozen_opponent_temperature == 0.0
    assert run_cfg.self_play is False
    assert run_cfg.engine == "vec"

    exploiter = get_training_preset("exploit-selfplay-seed0").train_defaults
    control = get_training_preset("continue-selfplay-seed0").train_defaults
    shared = set(exploiter) & set(control)
    assert {key for key in shared if exploiter[key] != control[key]} == set()
    assert set(control) - set(exploiter) == {
        "self_play", "self_play_ratio", "self_play_schedule", "pfsp",
    }


def test_load_frozen_opponents_names_and_copies_checkpoints(setup, tmp_path):
    _, _, registry = setup
    model = make_model(registry, seed=3)
    path = save_model(tmp_path / "champ.pth", model)
    pool = OpponentPool(opponent_spec="random")

    names = load_frozen_opponents(pool, [path], registry.card_names, registry)

    assert names == ["frozen:champ"]
    [(name, state_dict, model_config)] = pool._snapshots
    assert name == "frozen:champ"
    assert isinstance(model_config, ModelConfig)
    for key, tensor in model.state_dict().items():
        assert torch.equal(state_dict[key], tensor)
    assert pool.to_manifest()["snapshots"] == []


def test_load_frozen_opponents_rejects_bad_checkpoints(setup, tmp_path):
    _, _, registry = setup
    model = make_model(registry, seed=4)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = save_model(tmp_path / "a" / "same.pth", model)
    second = save_model(tmp_path / "b" / "same.pth", model)
    with pytest.raises(ValueError, match="shares the name"):
        load_frozen_opponents(
            OpponentPool(opponent_spec="random"), [first, second],
            registry.card_names, registry,
        )

    with pytest.raises(FileNotFoundError):
        load_frozen_opponents(
            OpponentPool(opponent_spec="random"), [str(tmp_path / "missing.pth")],
            registry.card_names, registry,
        )

    state = model.state_dict()
    key = next(name for name in state if name.startswith("actor_head") and name.endswith("weight"))
    state[key] = state[key][:-1]
    bad = tmp_path / "bad.pth"
    save_checkpoint(str(bad), state, model_config=ModelConfig())
    with pytest.raises(ValueError, match="does not fit"):
        load_frozen_opponents(
            OpponentPool(opponent_spec="random"), [str(bad)],
            registry.card_names, registry,
        )


def test_snapshot_actions_at_temperature_zero_and_one():
    torch.manual_seed(0)
    logits = torch.randn(64, 12)
    logits[:, ::3] = float("-inf")
    assert torch.equal(_snapshot_actions(logits, 0.0), logits.argmax(dim=-1))

    torch.manual_seed(11)
    sampled = _snapshot_actions(logits, 1.0)
    torch.manual_seed(11)
    expected = torch.distributions.Categorical(logits=logits, validate_args=False).sample()
    assert torch.equal(sampled, expected)
    assert torch.isfinite(logits.gather(1, sampled[:, None])).all()


def _observations_and_masks(registry, rows, seed):
    rng = np.random.default_rng(seed)
    state_dim = get_state_size(registry.card_names)
    action_dim = get_action_space_size(registry.card_names)
    observations = rng.random((rows, state_dim), dtype=np.float32)
    masks = rng.random((rows, action_dim)) < 0.2
    masks[:, 1] = True
    return observations, masks


def _greedy(model, observations, masks):
    effective = torch.from_numpy(suppress_redundant_end_turn(masks))
    with torch.no_grad():
        logits = model.policy_logits(torch.from_numpy(observations))
    return logits.masked_fill(~effective, float("-inf")).argmax(dim=-1).numpy()


def test_joint_evaluator_plays_snapshot_greedily_at_temperature_zero(setup):
    _, _, registry = setup
    current = make_model(registry, seed=5)
    snapshot = make_model(registry, seed=6)
    evaluator = CurrentSnapshotEvaluator(
        current, snapshot, torch.device("cpu"), snapshot_temperature=0.0,
    )
    current_observations, current_masks = _observations_and_masks(registry, 9, 1)
    snapshot_observations, snapshot_masks = _observations_and_masks(registry, 13, 2)

    decision = evaluator.decide(
        current_observations, current_masks, snapshot_observations, snapshot_masks,
    )

    np.testing.assert_array_equal(
        decision.snapshot_actions,
        _greedy(snapshot, snapshot_observations, snapshot_masks),
    )
    assert CurrentSnapshotEvaluator(
        current, snapshot, torch.device("cpu"),
    ).snapshot_temperature == 1.0


def test_stacked_evaluator_plays_snapshots_greedily_at_temperature_zero(setup):
    _, _, registry = setup
    models = [make_model(registry, seed=seed) for seed in (7, 8, 9)]
    evaluator = StackedSnapshotEvaluator(
        models, torch.device("cpu"), temperature=0.0,
    )
    groups = [_observations_and_masks(registry, rows, seed) for rows, seed in ((5, 3), (0, 4), (8, 5))]

    decision = evaluator.decide(
        [observations for observations, _ in groups],
        [masks for _, masks in groups],
    )

    for model, (observations, masks), actions in zip(models, groups, decision.actions):
        if len(observations):
            np.testing.assert_array_equal(actions, _greedy(model, observations, masks))
        else:
            assert len(actions) == 0
    assert StackedSnapshotEvaluator(models, torch.device("cpu")).temperature == 1.0


def test_vector_runner_plays_every_game_against_the_frozen_snapshot(setup):
    _, cards, registry = setup
    frozen = make_model(registry, seed=10)
    runner = VectorBatchRunner(
        model=make_model(registry, seed=11),
        cards=cards,
        registry=registry,
        device=torch.device("cpu"),
        ppo_config=PPOConfig(),
        opponent_spec="random,heuristic,simple",
        seed=31,
        snapshot_state_dicts=[("frozen:champ", frozen.state_dict(), ModelConfig())],
        self_play_ratio=1.0,
        pfsp_weights=[1.0],
        snapshot_temperature=0.0,
    )

    runner.run_episodes(8)

    assert (runner._last_opponent_kinds >= SNAPSHOT_OFFSET).all()
    assert runner.opponent_results["frozen:champ"][1] == 8
    assert runner._joint_snapshot_evaluators[0].snapshot_temperature == 0.0


def test_training_and_resume_keep_exactly_the_frozen_opponent(setup, tmp_path, monkeypatch):
    from src.ppo import ppo_trainer

    _, _, registry = setup
    frozen_model = make_model(registry, seed=12)
    frozen_path = save_model(tmp_path / "champ.pth", frozen_model)
    frozen_before = {key: value.clone() for key, value in frozen_model.state_dict().items()}
    runners = []
    original = ppo_trainer._make_runner

    def recording_make_runner(*args, **kwargs):
        runner = original(*args, **kwargs)
        runners.append(runner)
        return runner

    monkeypatch.setattr(ppo_trainer, "_make_runner", recording_make_runner)
    monkeypatch.chdir(tmp_path)
    data_cfg = DataConfig(
        cards_path=str(REPO / "data" / "cards.csv"),
        models_dir=str(tmp_path / "models"),
    )
    dev_cfg = DeviceConfig(main_device="cpu", simulation_device="cpu")

    def run(updates, resume=None):
        run_cfg = RunConfig(
            engine="vec", episodes=8, updates=updates, eval_every=1000,
            eval_games=2, num_workers=1, opponents="random", resume=resume,
            frozen_opponents=[frozen_path], frozen_opponent_ratio=1.0,
            frozen_opponent_temperature=0.0,
        )
        ppo_trainer.train(
            data_cfg, PPOConfig(), run_cfg, dev_cfg,
            model_path=None if resume else frozen_path, seed=5,
        )

    def checkpoint(update):
        [path] = (tmp_path / "models").glob(f"ppo_agent_*_upd{update}_wins*.pth")
        return path

    run(2)
    run(1, resume=str(checkpoint(2)))

    assert len(runners) == 3
    for runner in runners:
        assert [name for name, _, _ in runner.snapshot_state_dicts] == ["frozen:champ"]
        assert runner.self_play_ratio == 1.0
        assert runner.snapshot_temperature == 0.0
        assert (runner._last_opponent_kinds >= SNAPSHOT_OFFSET).all()
        state_dict = runner.snapshot_state_dicts[0][1]
        for key, tensor in frozen_before.items():
            assert torch.equal(state_dict[key], tensor), key

    on_disk = load_checkpoint(frozen_path, map_location="cpu")["model_state_dict"]
    for key, tensor in frozen_before.items():
        assert torch.equal(on_disk[key], tensor), key
    final = load_checkpoint(str(checkpoint(3)), map_location="cpu")
    assert final["update"] == 3
    assert final.get("pool_manifest") is None
    assert any(
        not torch.equal(final["model_state_dict"][key], tensor)
        for key, tensor in frozen_before.items()
    )
