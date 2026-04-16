"""Tests for self-play ratio scheduling."""
import math
import pytest

from src.config import RunConfig
from src.ppo.ppo_trainer import _compute_self_play_ratio


class TestComputeSelfPlayRatio:
    """Validate the schedule interpolation function."""

    def test_constant_returns_end(self):
        assert _compute_self_play_ratio(0.0, 0.5, "constant", 0.0) == 0.5
        assert _compute_self_play_ratio(0.0, 0.5, "constant", 0.5) == 0.5
        assert _compute_self_play_ratio(0.0, 0.5, "constant", 1.0) == 0.5

    def test_linear_endpoints(self):
        assert _compute_self_play_ratio(0.0, 1.0, "linear", 0.0) == 0.0
        assert _compute_self_play_ratio(0.0, 1.0, "linear", 1.0) == 1.0

    def test_linear_midpoint(self):
        assert pytest.approx(_compute_self_play_ratio(0.0, 1.0, "linear", 0.5)) == 0.5
        assert pytest.approx(_compute_self_play_ratio(0.2, 0.8, "linear", 0.5)) == 0.5

    def test_linear_quarter(self):
        assert pytest.approx(_compute_self_play_ratio(0.0, 1.0, "linear", 0.25)) == 0.25

    def test_cosine_endpoints(self):
        assert pytest.approx(_compute_self_play_ratio(0.0, 1.0, "cosine", 0.0)) == 0.0
        assert pytest.approx(_compute_self_play_ratio(0.0, 1.0, "cosine", 1.0)) == 1.0

    def test_cosine_midpoint(self):
        assert pytest.approx(_compute_self_play_ratio(0.0, 1.0, "cosine", 0.5)) == 0.5

    def test_cosine_starts_slow(self):
        """Cosine schedule should be below linear at early progress."""
        cosine_val = _compute_self_play_ratio(0.0, 1.0, "cosine", 0.25)
        linear_val = _compute_self_play_ratio(0.0, 1.0, "linear", 0.25)
        assert cosine_val < linear_val

    def test_cosine_ends_fast(self):
        """Cosine schedule should be above linear at late progress."""
        cosine_val = _compute_self_play_ratio(0.0, 1.0, "cosine", 0.75)
        linear_val = _compute_self_play_ratio(0.0, 1.0, "linear", 0.75)
        assert cosine_val > linear_val

    def test_progress_clamped_below_zero(self):
        assert _compute_self_play_ratio(0.0, 1.0, "linear", -0.5) == 0.0

    def test_progress_clamped_above_one(self):
        assert _compute_self_play_ratio(0.0, 1.0, "linear", 1.5) == 1.0

    def test_nonzero_start(self):
        assert pytest.approx(_compute_self_play_ratio(0.1, 0.9, "linear", 0.0)) == 0.1
        assert pytest.approx(_compute_self_play_ratio(0.1, 0.9, "linear", 1.0)) == 0.9
        assert pytest.approx(_compute_self_play_ratio(0.1, 0.9, "linear", 0.5)) == 0.5

    def test_equal_start_end(self):
        """When start == end, schedule should be constant regardless of type."""
        for schedule in ("constant", "linear", "cosine"):
            for progress in (0.0, 0.5, 1.0):
                assert _compute_self_play_ratio(0.3, 0.3, schedule, progress) == pytest.approx(0.3)

    def test_unknown_schedule_returns_end(self):
        assert _compute_self_play_ratio(0.0, 0.7, "unknown", 0.5) == 0.7


class TestRunConfigValidation:
    """Validate RunConfig self-play schedule input validation."""

    def test_valid_constant_schedule(self):
        cfg = RunConfig(self_play_ratio=0.5, self_play_schedule="constant")
        assert cfg.self_play_schedule == "constant"

    def test_valid_linear_schedule(self):
        cfg = RunConfig(
            self_play_ratio=0.8,
            self_play_ratio_start=0.1,
            self_play_schedule="linear",
        )
        assert cfg.self_play_schedule == "linear"

    def test_valid_cosine_schedule(self):
        cfg = RunConfig(
            self_play_ratio=0.8,
            self_play_ratio_start=0.0,
            self_play_schedule="cosine",
        )
        assert cfg.self_play_schedule == "cosine"

    def test_invalid_schedule_rejected(self):
        with pytest.raises(ValueError, match="Unknown self_play_schedule"):
            RunConfig(self_play_schedule="exponential")

    def test_ratio_start_below_zero_rejected(self):
        with pytest.raises(ValueError, match="self_play_ratio_start"):
            RunConfig(self_play_ratio_start=-0.1)

    def test_ratio_above_one_rejected(self):
        with pytest.raises(ValueError, match="self_play_ratio must be"):
            RunConfig(self_play_ratio=1.5)

    def test_start_greater_than_end_rejected(self):
        with pytest.raises(ValueError, match="must be <="):
            RunConfig(self_play_ratio=0.3, self_play_ratio_start=0.5)

    def test_edge_ratio_zero_to_one(self):
        cfg = RunConfig(self_play_ratio=1.0, self_play_ratio_start=0.0)
        assert cfg.self_play_ratio_start == 0.0
        assert cfg.self_play_ratio == 1.0

    def test_serialization_roundtrip(self):
        """New fields should survive to_dict → from_dict."""
        cfg = RunConfig(
            self_play_ratio=0.8,
            self_play_ratio_start=0.1,
            self_play_schedule="cosine",
        )
        d = cfg.to_dict()
        cfg2 = RunConfig.from_dict(d)
        assert cfg2.self_play_ratio == 0.8
        assert cfg2.self_play_ratio_start == 0.1
        assert cfg2.self_play_schedule == "cosine"

    def test_from_dict_ignores_unknown_keys(self):
        """from_dict should gracefully handle checkpoints with extra keys."""
        d = RunConfig().to_dict()
        d["future_field"] = 42
        cfg = RunConfig.from_dict(d)
        assert cfg.self_play_schedule == "linear"
