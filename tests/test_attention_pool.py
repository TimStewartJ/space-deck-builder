"""Smoke tests for the AttentionZonePooling module and PPOActorCritic wiring."""
import torch

from src.config import ModelConfig
from src.ppo.ppo_actor_critic import (
    PPOActorCritic, AttentionZonePooling, NUM_ZONES, NUMERIC_DIM,
)
from src.encoding.state_encoder import get_state_size
from src.encoding.action_encoder import get_action_space_size


def _card_names(n: int = 20) -> list[str]:
    return [f"Card{i}" for i in range(n)]


def test_attention_pool_forward_shape_matches_sum():
    cards = _card_names()
    num_cards = len(cards)
    state_dim = get_state_size(cards)
    action_dim = get_action_space_size(cards)

    model = PPOActorCritic(
        state_dim, action_dim, num_cards,
        model_config=ModelConfig(pool_type="attention"),
    )
    x = torch.rand(4, state_dim) * 0.1
    logits, value = model(x)
    assert logits.shape == (4, action_dim)
    assert value.shape == (4,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_attention_pool_empty_zones_no_nan():
    """A zero state vector has all presence=0. Attention softmax would produce
    NaN over all-masked rows; the any_present gate must zero those out."""
    cards = _card_names()
    num_cards = len(cards)
    state_dim = get_state_size(cards)
    action_dim = get_action_space_size(cards)

    model = PPOActorCritic(
        state_dim, action_dim, num_cards,
        model_config=ModelConfig(pool_type="attention"),
    )
    x = torch.zeros(2, state_dim)
    logits, value = model(x)
    assert torch.isfinite(logits).all(), "empty-zone batch produced NaN/Inf logits"
    assert torch.isfinite(value).all(), "empty-zone batch produced NaN/Inf value"


def test_attention_pool_module_unit():
    """Direct test of AttentionZonePooling: shape, gating, and gradient flow."""
    B, Z, C, E = 3, NUM_ZONES, 12, 8
    pool = AttentionZonePooling(Z, E)
    card_w = torch.randn(C, E, requires_grad=True)
    zone_w = torch.randn(Z, E, requires_grad=True)
    presence = torch.zeros(B, Z, C)
    # Only batch-0 zone-0 has any cards present.
    presence[0, 0, :3] = 1.0

    out = pool(card_w, zone_w, presence)
    assert out.shape == (B, Z * E)
    # Zones with no present cards must be exactly zero.
    out_view = out.view(B, Z, E)
    assert torch.all(out_view[1] == 0)
    assert torch.all(out_view[2] == 0)
    assert torch.all(out_view[0, 1:] == 0)
    # The populated zone should be non-zero (probabilistically true with random init).
    assert not torch.all(out_view[0, 0] == 0)

    # Gradient flows back through the query.
    out.sum().backward()
    assert pool.zone_query.grad is not None
    assert torch.isfinite(pool.zone_query.grad).all()


def test_sum_pool_still_works():
    """Default pool_type='sum' must remain functional (no regression)."""
    cards = _card_names()
    num_cards = len(cards)
    state_dim = get_state_size(cards)
    action_dim = get_action_space_size(cards)

    model = PPOActorCritic(
        state_dim, action_dim, num_cards,
        model_config=ModelConfig(),  # defaults: pool_type="sum"
    )
    assert model.pool_type == "sum"
    assert model.zone_pool is None
    x = torch.rand(2, state_dim) * 0.1
    logits, value = model(x)
    assert logits.shape == (2, action_dim)
    assert value.shape == (2,)


def test_attention_state_dict_requires_matching_model_config():
    """Regression test for the batch_runner parallel worker config threading:
    an attention-pool state_dict can only be loaded into another attention-pool
    model. Default-config load must fail; matching-config load must succeed."""
    import pytest
    cards = _card_names()
    num_cards = len(cards)
    state_dim = get_state_size(cards)
    action_dim = get_action_space_size(cards)

    src = PPOActorCritic(
        state_dim, action_dim, num_cards,
        model_config=ModelConfig(pool_type="attention"),
    )
    sd = src.state_dict()
    assert any("zone_pool.zone_query" in k for k in sd.keys())

    default_model = PPOActorCritic(state_dim, action_dim, num_cards)
    with pytest.raises(RuntimeError):
        default_model.load_state_dict(sd)

    matched = PPOActorCritic(
        state_dim, action_dim, num_cards,
        model_config=ModelConfig(pool_type="attention"),
    )
    matched.load_state_dict(sd)  # should not raise
