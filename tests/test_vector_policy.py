"""Actual PPO model integration with vector observations and masks."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config import DataConfig, ModelConfig
from src.encoding.action_encoder import get_action_space_size
from src.encoding.state_encoder import get_state_size
from src.ppo.ppo_actor_critic import PPOActorCritic
from src.vector_engine.kernels import PLAY_OFFSET, VectorEngine
from src.vector_engine.policy import (
    VectorPolicyEvaluator,
    suppress_redundant_end_turn,
)
from src.vector_engine.rules import compile_rules
from src.vector_engine.state import VectorState, ZONE_HAND


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_actual_policy_only_samples_legal_vector_actions(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("ROCm/CUDA unavailable")
    data = DataConfig()
    cards = data.load_cards()
    registry = data.build_registry(cards)
    rules = compile_rules(cards, registry)
    games = 64
    state = VectorState.empty(games, rules.num_cards)
    state.health[:] = 50
    scout = registry.card_index_map["Scout"]
    state.zones[:, 0, ZONE_HAND, scout] = 1
    engine = VectorEngine(state, rules)
    masks = engine.build_masks()
    observations = engine.materialize_observations()

    model = PPOActorCritic(
        state_dim=get_state_size(registry.card_names),
        action_dim=get_action_space_size(registry.card_names),
        num_cards=rules.num_cards,
        model_config=ModelConfig(),
    )
    evaluator = VectorPolicyEvaluator(model, torch.device(device_name))
    decision = evaluator.decide(observations, masks)

    assert decision.actions.shape == (games,)
    assert decision.log_probs.shape == (games,)
    assert decision.values.shape == (games,)
    assert np.isfinite(decision.log_probs).all()
    assert np.isfinite(decision.values).all()
    assert masks[np.arange(games), decision.actions].all()
    assert set(np.unique(decision.actions)) == {PLAY_OFFSET + scout}
    assert masks[:, 1].all(), "policy suppression must not mutate engine masks"


def test_production_mask_suppression_matches_worker_policy():
    masks = np.zeros((3, 494), dtype=np.bool_)
    masks[:, 1] = True
    masks[0, PLAY_OFFSET + 2] = True
    attack_player = 3 + 2 * 49
    masks[1, attack_player] = True
    effective = suppress_redundant_end_turn(masks)
    assert not effective[0, 1]
    assert not effective[1, 1]
    assert effective[2, 1]
    assert masks[:, 1].all()
