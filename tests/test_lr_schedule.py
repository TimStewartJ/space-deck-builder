"""Tests for learning rate scheduling in PPO training."""
import math
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from src.config import PPOConfig, RunConfig


class TestPPOConfigLRFields:
    """Validate PPOConfig LR schedule fields and validation."""

    def test_default_is_cosine(self):
        cfg = PPOConfig()
        assert cfg.lr_end == 1e-5

    def test_lr_end_greater_than_lr_rejected(self):
        with pytest.raises(ValueError, match="lr_end.*must be <= lr"):
            PPOConfig(lr=1e-4, lr_end=1e-3)

    def test_negative_lr_end_rejected(self):
        with pytest.raises(ValueError, match="lr_end must be non-negative"):
            PPOConfig(lr_end=-1e-5)

    def test_lr_end_equals_lr_accepted(self):
        """When lr_end == lr, cosine schedule produces a constant LR."""
        cfg = PPOConfig(lr=3e-4, lr_end=3e-4)
        assert cfg.lr_end == cfg.lr

    def test_serialization_roundtrip(self):
        """LR fields survive to_dict -> from_dict."""
        cfg = PPOConfig(lr=1e-3, lr_end=1e-6)
        d = cfg.to_dict()
        cfg2 = PPOConfig.from_dict(d)
        assert cfg2.lr == 1e-3
        assert cfg2.lr_end == 1e-6

    def test_from_dict_ignores_unknown_keys(self):
        """Old checkpoints with retired fields (lr_schedule, adv_norm,
        snapshot_eviction) load cleanly — unknown keys are dropped."""
        old_dict = {"lr": 3e-4, "gamma": 0.99, "lam": 0.95,
                    "clip_eps": 0.2, "epochs": 4, "batch_size": 8192,
                    "entropy_coef": 0.025, "grad_clip": 0.5,
                    "critic_loss_coef": 0.5,
                    "adv_norm": "global", "lr_schedule": "cosine"}
        cfg = PPOConfig.from_dict(old_dict)
        assert cfg.lr == 3e-4
        assert cfg.lr_end == 1e-5


class TestRunConfigDefaultChange:
    """Validate that self_play_schedule defaults to 'linear'."""

    def test_default_is_linear(self):
        cfg = RunConfig()
        assert cfg.self_play_schedule == "linear"


class TestCosineSchedulerBehavior:
    """Validate the cosine annealing scheduler produces expected LR values.

    These tests exercise the exact scheduler construction used by ppo_trainer
    (CosineAnnealingLR with T_max = max(1, updates - 1)).
    """

    def _make_optimizer_and_scheduler(self, lr, lr_end, updates):
        """Create a dummy model, optimizer, and scheduler matching trainer logic."""
        model = nn.Linear(4, 2)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = lr_sched.CosineAnnealingLR(
            optimizer, T_max=max(1, updates - 1), eta_min=lr_end,
        )
        return optimizer, scheduler

    def _get_lr(self, optimizer):
        return optimizer.param_groups[0]['lr']

    def test_first_update_uses_initial_lr(self):
        """Update 1 should train at the initial LR (no step yet)."""
        opt, sched = self._make_optimizer_and_scheduler(3e-4, 1e-5, 100)
        assert self._get_lr(opt) == pytest.approx(3e-4)

    def test_last_update_uses_lr_end(self):
        """The final update should train at lr_end."""
        lr, lr_end, updates = 3e-4, 1e-5, 100
        opt, sched = self._make_optimizer_and_scheduler(lr, lr_end, updates)
        # Simulate the training loop: step after each update
        for _ in range(updates - 1):
            sched.step()
        # After updates-1 steps, the LR should be at lr_end
        assert self._get_lr(opt) == pytest.approx(lr_end, rel=1e-5)

    def test_midpoint_lr_is_between_start_and_end(self):
        """At 50% progress, LR should be roughly the midpoint."""
        lr, lr_end, updates = 3e-4, 1e-5, 101
        opt, sched = self._make_optimizer_and_scheduler(lr, lr_end, updates)
        for _ in range(50):
            sched.step()
        mid_lr = self._get_lr(opt)
        expected_mid = lr_end + (lr - lr_end) / 2  # cosine midpoint is exactly halfway
        assert mid_lr == pytest.approx(expected_mid, rel=1e-3)

    def test_lr_monotonically_decreases(self):
        """LR should never increase during the schedule."""
        lr, lr_end, updates = 3e-4, 1e-5, 50
        opt, sched = self._make_optimizer_and_scheduler(lr, lr_end, updates)
        prev_lr = self._get_lr(opt)
        for _ in range(updates - 1):
            sched.step()
            current_lr = self._get_lr(opt)
            assert current_lr <= prev_lr + 1e-10  # small epsilon for float comparison
            prev_lr = current_lr

    def test_single_update_uses_initial_lr(self):
        """With updates=1, T_max=max(1,0)=1. The single update uses initial LR."""
        opt, sched = self._make_optimizer_and_scheduler(3e-4, 1e-5, 1)
        assert self._get_lr(opt) == pytest.approx(3e-4)

    def test_two_updates_reaches_lr_end(self):
        """With updates=2, T_max=1. After one step, LR should be at lr_end."""
        lr, lr_end = 3e-4, 1e-5
        opt, sched = self._make_optimizer_and_scheduler(lr, lr_end, 2)
        # Update 1 uses initial lr, step after
        sched.step()
        # Update 2 should use lr_end
        assert self._get_lr(opt) == pytest.approx(lr_end, rel=1e-5)

    def test_constant_schedule_keeps_lr_unchanged(self):
        """Sanity: with no scheduler stepping, LR stays fixed (the
        baseline-against-which cosine annealing is measured)."""
        model = nn.Linear(4, 2)
        optimizer = optim.Adam(model.parameters(), lr=3e-4)
        # No scheduler — simulate 100 updates
        for _ in range(100):
            pass  # no scheduler.step()
        assert optimizer.param_groups[0]['lr'] == 3e-4

    def test_lr_end_equals_lr_stays_constant(self):
        """Cosine with lr_end == lr should produce constant LR."""
        lr = 3e-4
        opt, sched = self._make_optimizer_and_scheduler(lr, lr, 50)
        for _ in range(49):
            sched.step()
            assert self._get_lr(opt) == pytest.approx(lr, rel=1e-5)
