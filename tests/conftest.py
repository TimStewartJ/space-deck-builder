"""Fixtures shared across the test suite."""
from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def policy_checkpoint(tmp_path_factory) -> str:
    """Path of a seeded, untrained policy checkpoint under the current rules.

    The published checkpoints were trained under rules version 1 and do not
    load under the current rules (docs/decisions/0016-rulebook-ability-timing.md),
    so tests that need a policy file use this one.
    """
    import torch

    from src.config import DataConfig, ModelConfig, save_checkpoint
    from src.encoding.action_encoder import get_action_space_size
    from src.encoding.state_encoder import get_state_size
    from src.ppo.ppo_actor_critic import PPOActorCritic

    data = DataConfig()
    registry = data.build_registry(data.load_cards())
    names = registry.card_names
    model_config = ModelConfig()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261030)
        model = PPOActorCritic(
            get_state_size(names),
            get_action_space_size(names),
            len(names),
            model_config=model_config,
            card_registry=registry,
        )
    path = tmp_path_factory.mktemp("policy") / "policy.pth"
    save_checkpoint(str(path), model.state_dict(), model_config=model_config)
    return str(path)
