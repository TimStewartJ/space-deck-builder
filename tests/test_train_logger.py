"""Tests for training logger and PPO metrics return value."""
import json
import logging
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.ppo.train_logger import (
    MetricsWriter,
    format_ppo_metrics,
    setup_training_logger,
)


class TestSetupTrainingLogger:
    """Tests for setup_training_logger."""

    def test_creates_logger_with_handlers(self, tmp_path):
        logger = setup_training_logger(str(tmp_path))
        assert logger.name == "training"
        assert len(logger.handlers) == 2
        handler_types = {type(h) for h in logger.handlers}
        assert logging.StreamHandler in handler_types
        assert logging.FileHandler in handler_types

    def test_creates_log_file(self, tmp_path):
        setup_training_logger(str(tmp_path))
        log_files = list(tmp_path.glob("training_*_*.log"))
        assert len(log_files) == 1

    def test_idempotent_no_handler_duplication(self, tmp_path):
        """Calling setup multiple times should not duplicate handlers."""
        logger = setup_training_logger(str(tmp_path))
        assert len(logger.handlers) == 2
        logger = setup_training_logger(str(tmp_path))
        assert len(logger.handlers) == 2

    def test_logger_writes_to_file(self, tmp_path):
        logger = setup_training_logger(str(tmp_path))
        logger.info("test message")
        log_files = list(tmp_path.glob("training_*_*.log"))
        content = log_files[0].read_text()
        assert "test message" in content

    def test_propagate_disabled(self, tmp_path):
        logger = setup_training_logger(str(tmp_path))
        assert logger.propagate is False

    def test_console_handler_uses_stdout(self, tmp_path):
        """Console handler should write to stdout, not stderr."""
        import sys
        logger = setup_training_logger(str(tmp_path))
        stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)
                          and not isinstance(h, logging.FileHandler)]
        assert len(stream_handlers) == 1
        assert stream_handlers[0].stream is sys.stdout


class TestMetricsWriter:
    """Tests for MetricsWriter JSONL output."""

    def test_writes_valid_jsonl(self, tmp_path):
        writer = MetricsWriter(str(tmp_path))
        writer.write({"update": 1, "actor_loss": 0.123})
        writer.write({"update": 2, "actor_loss": 0.456})
        writer.close()

        lines = writer.path.read_text().strip().split("\n")
        assert len(lines) == 2
        row1 = json.loads(lines[0])
        assert row1["update"] == 1
        assert row1["actor_loss"] == 0.123
        row2 = json.loads(lines[1])
        assert row2["update"] == 2

    def test_handles_nested_dicts(self, tmp_path):
        writer = MetricsWriter(str(tmp_path))
        writer.write({"rollout": {"win_rate": 0.55, "episodes": 128}})
        writer.close()

        row = json.loads(writer.path.read_text().strip())
        assert row["rollout"]["win_rate"] == 0.55

    def test_handles_torch_tensors(self, tmp_path):
        writer = MetricsWriter(str(tmp_path))
        writer.write({"value": torch.tensor(3.14)})
        writer.close()

        row = json.loads(writer.path.read_text().strip())
        assert abs(row["value"] - 3.14) < 0.01

    def test_creates_metrics_file(self, tmp_path):
        writer = MetricsWriter(str(tmp_path))
        assert writer.path.exists()
        writer.close()


class TestFormatPpoMetrics:
    """Tests for format_ppo_metrics."""

    def test_formats_all_keys(self):
        metrics = {
            "actor_loss": -0.012,
            "critic_loss": 0.34,
            "entropy": 3.21,
            "approx_kl": 0.008,
            "clip_fraction": 0.12,
            "total_grad_norm": 0.42,
            "explained_variance": 0.78,
        }
        result = format_ppo_metrics(metrics)
        assert "actor=-0.012" in result
        assert "critic=0.340" in result
        assert "ent=3.21" in result
        assert "kl=0.0080" in result
        assert "clip=0.120" in result
        assert "gnorm=0.420" in result
        assert "ev=0.78" in result

    def test_handles_missing_keys(self):
        result = format_ppo_metrics({"actor_loss": 0.1})
        assert "actor=0.100" in result
        assert "critic" not in result

    def test_handles_empty_dict(self):
        result = format_ppo_metrics({})
        assert result == ""


class TestPpoAgentUpdateMetrics:
    """Tests that PPOAgent.update() returns the expected metrics dict."""

    @pytest.fixture
    def agent_and_data(self):
        """Create a minimal PPOAgent and synthetic rollout data."""
        from src.config import PPOConfig, DeviceConfig, DataConfig

        data_cfg = DataConfig()
        cards = data_cfg.load_cards()
        registry = data_cfg.build_registry(cards)
        card_names = registry.card_names

        ppo_cfg = PPOConfig(epochs=1, batch_size=32)
        dev_cfg = DeviceConfig(main_device="cpu", simulation_device="cpu")

        from src.ai.ppo_agent import PPOAgent
        agent = PPOAgent(
            "PPO", card_names,
            ppo_config=ppo_cfg,
            device_config=dev_cfg,
            registry=registry,
        )

        # Synthetic rollout data
        n = 64
        state_dim = agent.state_dim
        action_dim = agent.action_dim

        states = torch.randn(n, state_dim)
        actions = torch.randint(0, action_dim, (n,))
        old_lp = torch.randn(n)
        returns = torch.randn(n)
        advs = torch.randn(n)
        masks = torch.ones(n, action_dim, dtype=torch.bool)

        return agent, states, actions, old_lp, returns, advs, masks

    def test_returns_dict_with_expected_keys(self, agent_and_data):
        agent, states, actions, old_lp, returns, advs, masks = agent_and_data
        result = agent.update(states, actions, old_lp, returns, advs, masks)

        expected_keys = {
            "actor_loss", "critic_loss", "mean_ratio", "entropy",
            "approx_kl", "clip_fraction", "total_grad_norm",
            "explained_variance", "nan_detected",
        }
        assert set(result.keys()) == expected_keys

    def test_metrics_are_finite(self, agent_and_data):
        agent, states, actions, old_lp, returns, advs, masks = agent_and_data
        result = agent.update(states, actions, old_lp, returns, advs, masks)

        assert result["nan_detected"] is False
        for key in ["actor_loss", "critic_loss", "entropy", "approx_kl",
                     "clip_fraction", "total_grad_norm", "explained_variance"]:
            val = result[key]
            assert isinstance(val, float), f"{key} should be float, got {type(val)}"
            assert not (val != val), f"{key} is NaN"  # NaN check

    def test_nan_detected_returns_full_dict(self, agent_and_data):
        """Verify the NaN path still returns all expected keys."""
        agent, states, actions, old_lp, returns, advs, masks = agent_and_data

        # Corrupt the model to trigger NaN
        with torch.no_grad():
            for p in agent.model.parameters():
                p.fill_(float("nan"))

        result = agent.update(states, actions, old_lp, returns, advs, masks)
        assert result["nan_detected"] is True
        expected_keys = {
            "actor_loss", "critic_loss", "mean_ratio", "entropy",
            "approx_kl", "clip_fraction", "total_grad_norm",
            "explained_variance", "nan_detected",
        }
        assert set(result.keys()) == expected_keys

    def test_singleton_tail_chunk_does_not_crash(self, agent_and_data):
        """Regression for ppo_agent.py:284 zero-dim concat crash.

        Explained-variance is computed by chunking the rollout in 4096-step
        slices and concatenating per-chunk value predictions. When the rollout
        length leaves a 1-step tail (``len(states) % 4096 == 1``) the model's
        critic returns a tensor that, after an over-eager ``squeeze(-1)``,
        collapses to a 0-d scalar — which ``torch.cat`` rejects with
        "zero-dimensional tensor (at position N) cannot be concatenated".

        Use ``reshape(-1)`` instead so a singleton tail stays 1-d.
        """
        agent, _, _, _, _, _, _ = agent_and_data

        n = 4097  # one full chunk + a 1-step tail
        state_dim = agent.state_dim
        action_dim = agent.action_dim

        states = torch.randn(n, state_dim)
        actions = torch.randint(0, action_dim, (n,))
        old_lp = torch.randn(n)
        returns = torch.randn(n)
        advs = torch.randn(n)
        masks = torch.ones(n, action_dim, dtype=torch.bool)

        result = agent.update(states, actions, old_lp, returns, advs, masks)
        assert result["nan_detected"] is False
        assert isinstance(result["explained_variance"], float)
