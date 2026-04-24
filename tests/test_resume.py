"""Tests for resume-training: optimizer/scheduler/pool restore round-trips."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.config import (
    CHECKPOINT_VERSION,
    DataConfig,
    ModelConfig,
    PPOConfig,
    RunConfig,
    load_checkpoint,
    save_checkpoint,
)
from src.ppo.opponent_pool import OpponentPool


def _dummy_model() -> torch.nn.Module:
    return torch.nn.Linear(4, 2)


def _dummy_optimizer(model: torch.nn.Module, lr: float = 1e-3) -> torch.optim.Optimizer:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Take a step so optimizer has non-empty Adam moments
    x = torch.randn(8, 4)
    target = torch.randn(8, 2)
    out = model(x)
    loss = ((out - target) ** 2).mean()
    loss.backward()
    opt.step()
    return opt


def test_save_load_includes_optimizer_state(tmp_path: Path):
    model = _dummy_model()
    opt = _dummy_optimizer(model)
    ckpt_path = tmp_path / "ckpt.pth"
    save_checkpoint(
        str(ckpt_path),
        model.state_dict(),
        ppo_config=PPOConfig(),
        model_config=ModelConfig(),
        update=5,
        optimizer_state_dict=opt.state_dict(),
    )
    data = load_checkpoint(str(ckpt_path), map_location="cpu")
    assert data["schema_version"] == CHECKPOINT_VERSION
    assert data["update"] == 5
    assert "optimizer_state_dict" in data
    # Round-trip: rebuild and load
    model2 = _dummy_model()
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    opt2.load_state_dict(data["optimizer_state_dict"])
    # Adam's exp_avg tensors should match
    for g1, g2 in zip(opt.state_dict()["state"].values(), opt2.state_dict()["state"].values()):
        assert torch.allclose(g1["exp_avg"], g2["exp_avg"])


def test_save_load_includes_scheduler_state(tmp_path: Path):
    model = _dummy_model()
    opt = _dummy_optimizer(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=1e-5)
    sched.step()
    sched.step()
    ckpt_path = tmp_path / "ckpt.pth"
    save_checkpoint(
        str(ckpt_path),
        model.state_dict(),
        update=2,
        scheduler_state_dict=sched.state_dict(),
    )
    data = load_checkpoint(str(ckpt_path), map_location="cpu")
    assert data["scheduler_state_dict"]["last_epoch"] == 2


def test_scheduler_load_state_clobbers_t_max():
    """Regression: PyTorch _LRScheduler.load_state_dict does dict.update(),
    which silently overwrites T_max / eta_min / base_lrs with the saved
    values. The trainer must re-pin these after load_state_dict to extend
    the cosine horizon for resumed runs.
    """
    model_a = _dummy_model()
    opt_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)
    sched_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=10, eta_min=0)
    for _ in range(5):
        sched_a.step()
    saved_state = sched_a.state_dict()

    # Construct a scheduler with a NEW horizon (simulating resume
    # extension) — load_state_dict reverts T_max back to 10.
    model_b = _dummy_model()
    opt_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)
    sched_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=100, eta_min=0)
    assert sched_b.T_max == 100
    sched_b.load_state_dict(saved_state)
    # Demonstrate the bug: T_max reverted to the saved value.
    assert sched_b.T_max == 10, "PyTorch behavior changed; rethink the trainer fix"

    # The trainer's fix: re-pin T_max after load_state_dict.
    sched_b.T_max = 99  # would be max(1, total_horizon - 1)
    assert sched_b.T_max == 99
    assert sched_b.last_epoch == 5  # but progress is preserved


def test_optimizer_load_overwrites_lr_param_group():
    """Regression: optim.load_state_dict overwrites param_group['lr'] from
    the checkpoint, silently superseding any --lr the user passed alongside
    --resume. PPOAgent must surface this when the requested LR differs.
    """
    model_a = _dummy_model()
    opt_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)
    # Take a step so optimizer has moments
    x = torch.randn(8, 4); target = torch.randn(8, 2)
    ((model_a(x) - target) ** 2).mean().backward()
    opt_a.step()

    model_b = _dummy_model()
    opt_b = torch.optim.Adam(model_b.parameters(), lr=5e-5)
    assert opt_b.param_groups[0]["lr"] == 5e-5
    opt_b.load_state_dict(opt_a.state_dict())
    # Demonstrate the issue: the user-requested 5e-5 is gone.
    assert opt_b.param_groups[0]["lr"] == 1e-3


def test_load_v2_checkpoint_without_resume_metadata(tmp_path: Path):
    """V2 checkpoints (no optimizer/scheduler/pool fields) still load cleanly."""
    model = _dummy_model()
    ckpt_path = tmp_path / "ckpt_v2.pth"
    # Manually write a v2-shaped checkpoint
    torch.save(
        {
            "schema_version": 2,
            "model_state_dict": model.state_dict(),
            "config": {"model": ModelConfig().to_dict()},
            "update": 10,
        },
        str(ckpt_path),
    )
    data = load_checkpoint(str(ckpt_path), map_location="cpu")
    assert data["schema_version"] == 2
    assert "optimizer_state_dict" not in data
    assert data["update"] == 10


def test_load_v1_checkpoint_rejected(tmp_path: Path):
    """V1 checkpoints have incompatible model shapes — must error."""
    ckpt_path = tmp_path / "ckpt_v1.pth"
    torch.save({"schema_version": 1, "model_state_dict": {}}, str(ckpt_path))
    with pytest.raises(ValueError, match="incompatible"):
        load_checkpoint(str(ckpt_path), map_location="cpu")


def test_pool_manifest_round_trip(tmp_path: Path):
    """OpponentPool can serialize a snapshot list to a manifest and rehydrate."""
    pool = OpponentPool(opponent_spec="random", self_play_ratio=0.5)

    # Create 3 fake snapshots, each with its own checkpoint file on disk
    mcfg = ModelConfig()
    snap_paths = []
    for i in range(3):
        sd = {f"layer{i}.weight": torch.randn(2, 2)}
        path = tmp_path / f"snap_{i}.pth"
        save_checkpoint(str(path), sd, model_config=mcfg, update=i + 1)
        snap_paths.append(str(path))
        pool.add_snapshot(sd, f"PPO_{i + 1}", model_config=mcfg)
        pool.set_snapshot_path(f"PPO_{i + 1}", str(path))

    # Simulate some PFSP feedback so EMA/games are non-default
    pool.update_results({"PPO_1": (3, 10), "PPO_2": (5, 10), "PPO_3": (8, 10)})

    manifest = pool.to_manifest()
    assert len(manifest["snapshots"]) == 3
    assert manifest["snapshots"][0]["name"] == "PPO_1"
    assert manifest["snapshots"][2]["path"] == snap_paths[2]
    assert "PPO_2" in manifest["ema"]
    assert manifest["games"]["PPO_1"] == 10

    # Rehydrate into a fresh pool
    new_pool = OpponentPool(opponent_spec="random", self_play_ratio=0.5)
    loaded = new_pool.load_from_manifest(manifest)
    assert loaded == 3
    assert new_pool.has_snapshots
    assert [n for n, _, _ in new_pool._snapshots] == ["PPO_1", "PPO_2", "PPO_3"]
    # PFSP state preserved
    assert new_pool._snapshot_ema["PPO_1"] == pytest.approx(pool._snapshot_ema["PPO_1"])
    assert new_pool._snapshot_games["PPO_1"] == 10


def test_pool_manifest_skips_missing_paths(tmp_path: Path):
    """Missing snapshot files should be skipped with a warning, not crash."""
    manifest = {
        "snapshots": [
            {"name": "PPO_1", "path": str(tmp_path / "does_not_exist.pth")},
            {"name": "PPO_2", "path": str(tmp_path / "also_missing.pth")},
        ],
        "ema": {"PPO_1": 0.4, "PPO_2": 0.6},
        "games": {"PPO_1": 10, "PPO_2": 20},
    }
    pool = OpponentPool(opponent_spec="random", self_play_ratio=0.5)
    warnings: list[str] = []
    loaded = pool.load_from_manifest(manifest, log_fn=warnings.append)
    assert loaded == 0
    assert not pool.has_snapshots
    assert len(warnings) == 2
    assert all("missing" in w for w in warnings)


def test_pool_manifest_omits_snapshots_without_path(tmp_path: Path):
    """Snapshots added without a recorded path don't appear in the manifest."""
    pool = OpponentPool(opponent_spec="random", self_play_ratio=0.5)
    pool.add_snapshot({"w": torch.zeros(1)}, "PPO_X", model_config=ModelConfig())
    # No set_snapshot_path call
    manifest = pool.to_manifest()
    assert manifest["snapshots"] == []


def test_pool_manifest_respects_snapshot_cap(tmp_path: Path):
    """Loading more snapshots than the cap permits keeps a geometric spread.

    Under the default ``geometric`` eviction strategy, the pool retains
    endpoints plus log-spaced intermediate ages rather than the last N
    entries. With updates 0..4 and cap=3, the ladder pins 0 (oldest) and
    4 (newest) and slots the midpoint at age ~2 — i.e. update 2.
    """
    mcfg = ModelConfig()
    snap_entries = []
    for i in range(5):
        sd = {"w": torch.randn(2)}
        path = tmp_path / f"snap_{i}.pth"
        save_checkpoint(str(path), sd, model_config=mcfg, update=i)
        snap_entries.append({"name": f"PPO_{i}", "path": str(path), "update": i})
    manifest = {
        "snapshots": snap_entries,
        "ema": {f"PPO_{i}": 0.5 for i in range(5)},
        "games": {f"PPO_{i}": 0 for i in range(5)},
    }
    pool = OpponentPool(opponent_spec="random", self_play_ratio=0.5, snapshot_cap=3)
    loaded = pool.load_from_manifest(manifest)
    assert loaded == 5  # all attempted
    names = [n for n, _, _ in pool._snapshots]
    assert names == ["PPO_0", "PPO_2", "PPO_4"]
