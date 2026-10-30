"""The folded first trunk layer computes the same function as sum pooling.

``PPOActorCritic`` folds card and zone embeddings, presence-weighted sum
pooling and the first trunk layer into one matrix over the flat observation.
These tests compare it with the unfolded computation, which the model still
uses when ``_foldable`` is off (token features or attention pooling).
"""
import copy

import pytest
import torch

from src.config import ModelConfig
from src.ppo.ppo_actor_critic import PPOActorCritic

NUM_CARDS = 49
STATE_DIM = 4 + 5 + 6 + 10 * NUM_CARDS
ACTION_DIM = 4 + 10 * NUM_CARDS


def _model(seed: int = 0) -> PPOActorCritic:
    torch.manual_seed(seed)
    model = PPOActorCritic(STATE_DIM, ACTION_DIM, NUM_CARDS).double()
    # Default initialisation leaves the embeddings tiny next to the trunk;
    # widen them so the comparison exercises every term of the fold.
    with torch.no_grad():
        model.card_emb.weight.normal_(0.0, 1.0)
        model.zone_emb.weight.normal_(0.0, 1.0)
    return model


def _unfolded(model: PPOActorCritic) -> PPOActorCritic:
    reference = copy.deepcopy(model)
    reference._foldable = False
    return reference


def _observations(batch: int, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    # Card counts and resources are small non-negative integers in practice.
    return torch.randint(
        0, 4, (batch, STATE_DIM), generator=generator
    ).double()


def test_folded_forward_matches_sum_pooling():
    model = _model()
    reference = _unfolded(model)
    x = _observations(64)
    logits, values = model(x)
    expected_logits, expected_values = reference(x)
    torch.testing.assert_close(logits, expected_logits, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(values, expected_values, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(
        model.policy_logits(x), expected_logits, rtol=1e-10, atol=1e-10
    )


def test_folded_forward_accepts_a_single_observation():
    model = _model()
    reference = _unfolded(model)
    x = _observations(1)[0]
    logits, value = model(x)
    expected_logits, expected_value = reference(x)
    assert logits.shape == (ACTION_DIM,)
    assert value.shape == ()
    torch.testing.assert_close(logits, expected_logits, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(value, expected_value, rtol=1e-10, atol=1e-10)


def test_folded_gradients_match_sum_pooling():
    model = _model()
    reference = _unfolded(model)
    x = _observations(32)
    for net in (model, reference):
        logits, values = net(x)
        (logits.square().mean() + values.sin().sum()).backward()
    for (name, param), (_, ref_param) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        torch.testing.assert_close(
            param.grad, ref_param.grad, rtol=1e-9, atol=1e-10, msg=name
        )


def test_folding_keeps_the_checkpoint_format():
    model = _model()
    state = model.state_dict()
    assert not any("folded" in key for key in state)
    restored = PPOActorCritic(STATE_DIM, ACTION_DIM, NUM_CARDS).double()
    restored.load_state_dict(state)
    x = _observations(8)
    torch.testing.assert_close(restored(x)[0], model(x)[0])


def test_wrong_observation_width_is_rejected():
    model = _model()
    with pytest.raises(ValueError, match="observation features"):
        model(torch.zeros(2, STATE_DIM - 1, dtype=torch.float64))


@pytest.mark.parametrize(
    "config",
    [ModelConfig(pool_type="attention")],
)
def test_other_architectures_do_not_fold(config):
    model = PPOActorCritic(STATE_DIM, ACTION_DIM, NUM_CARDS, model_config=config)
    assert not model._foldable
    with pytest.raises(RuntimeError, match="folds"):
        model.folded_input_weight()
    logits, values = model(torch.zeros(3, STATE_DIM))
    assert logits.shape == (3, ACTION_DIM)
    assert values.shape == (3,)
