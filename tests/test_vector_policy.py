"""Actual PPO model integration with vector observations and masks."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config import DataConfig, ModelConfig
from src.ai.agent import Agent
from src.encoding.action_context import build_action_context
from src.encoding.action_encoder import get_action_space_size
from src.encoding.state_encoder import encode_state, get_state_size
from src.engine.game import Game
from src.ppo.ppo_actor_critic import PPOActorCritic
from src.vector_engine.kernels import PLAY_OFFSET, VectorEngine
from src.vector_engine.policy import (
    VectorPolicyEvaluator,
    suppress_redundant_end_turn,
)
from src.vector_engine.rules import compile_rules
from src.vector_engine.state import VectorState, ZONE_HAND, from_games


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


def test_policy_buffers_reuse_capacity_across_variable_batch_sizes():
    data = DataConfig()
    cards = data.load_cards()
    registry = data.build_registry(cards)
    model = PPOActorCritic(
        state_dim=get_state_size(registry.card_names),
        action_dim=get_action_space_size(registry.card_names),
        num_cards=registry.num_cards,
        model_config=ModelConfig(),
    )
    evaluator = VectorPolicyEvaluator(model, torch.device("cpu"))
    observations = np.zeros(
        (64, get_state_size(registry.card_names)), dtype=np.float32
    )
    masks = np.zeros(
        (64, get_action_space_size(registry.card_names)), dtype=np.bool_
    )
    masks[:, 1] = True
    evaluator.decide(observations, masks)
    pointer = evaluator.device_observations.data_ptr()
    evaluator.decide(observations[:63], masks[:63])
    assert evaluator.device_observations.data_ptr() == pointer
    assert evaluator._capacity == 64


def test_current_player_observation_matches_scalar_opponent_perspective():
    data = DataConfig()
    cards = data.load_cards()
    registry = data.build_registry(cards)
    rules = compile_rules(cards, registry)
    game = Game([], card_index_map=registry.card_index_map)
    first = game.add_player("P0", Agent("P0"))
    acting = game.add_player("P1", Agent("P1"))
    for player in game.players:
        game.stats.add_player(player.name)
    game.first_player_name = first.name
    game.current_turn = 1
    game.current_player = acting
    game.is_running = True
    acting.hand.extend(game.create_starting_deck()[:3])

    engine = VectorEngine(
        from_games([game], rules, training_players=[0]),
        rules,
    )
    engine.build_masks()
    vector = engine.materialize_current_player_observations()[0]
    context = build_action_context(
        game,
        acting,
        registry.card_index_map,
        get_action_space_size(registry.card_names),
    )
    scalar = encode_state(
        game,
        is_current_player_training=True,
        cards=registry.card_names,
        card_index_map=registry.card_index_map,
        can_buy=context.can_buy,
        has_actions=context.has_meaningful,
        return_numpy=True,
    )
    np.testing.assert_array_equal(vector, scalar)
