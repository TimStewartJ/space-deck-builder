"""Graph-replayed rollout policies agree with the eager model."""
import numpy as np
import pytest
import torch

from src.encoding.action_encoder import END_TURN_INDEX
from src.ppo.ppo_actor_critic import PPOActorCritic
from src.vector_engine.graphed_policy import (
    GraphedPolicyBank,
    batch_buckets,
    graphable,
    staged_rows,
)
from src.vector_engine.policy import suppress_redundant_end_turn

NUM_CARDS = 49
STATE_DIM = 4 + 5 + 6 + 10 * NUM_CARDS
ACTION_DIM = 4 + 10 * NUM_CARDS

cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA/HIP device"
)


def _model(seed: int) -> PPOActorCritic:
    torch.manual_seed(seed)
    model = PPOActorCritic(STATE_DIM, ACTION_DIM, NUM_CARDS)
    with torch.no_grad():
        model.card_emb.weight.normal_(0.0, 0.5)
        model.zone_emb.weight.normal_(0.0, 0.5)
        for layer in model.actor_head:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.mul_(4.0)
    return model.eval()


def _inputs(rows: int, seed: int):
    rng = np.random.default_rng(seed)
    observations = (rng.integers(0, 4, (rows, STATE_DIM)) * 0.125).astype(
        np.float32
    )
    masks = rng.random((rows, ACTION_DIM)) < 0.05
    masks[:, END_TURN_INDEX] = True
    masks[rows // 2 :, 3 : 3 + NUM_CARDS] = False  # no PLAY: END_TURN stays legal
    masks[-1] = False  # a row with no legal action at all
    return observations, masks


def _eager(model, observations, masks):
    effective = suppress_redundant_end_turn(masks)
    with torch.no_grad():
        logits, values = model(torch.from_numpy(observations))
    logits = logits.masked_fill(~torch.from_numpy(effective), float("-inf"))
    no_valid = ~torch.from_numpy(effective).any(dim=1)
    logits[no_valid, END_TURN_INDEX] = 0.0
    return logits, values, effective


def test_batch_buckets_cover_every_size():
    buckets = batch_buckets(2000)
    assert buckets[0] == 16 and buckets[-1] == 2000
    assert buckets == sorted(set(buckets))
    for rows in range(1, 2001):
        bucket = next(size for size in buckets if size >= rows)
        assert bucket - rows < max(16, rows // 4 + 1)
    assert batch_buckets(10) == [10]


def test_default_architecture_is_graphable():
    assert graphable(_model(0))


@cuda
def test_learner_bank_records_eager_values_log_probs_and_masks():
    device = torch.device("cuda")
    model = _model(1)
    bank = GraphedPolicyBank(
        model.to(device), 1, 256, device,
        temperature=1.0, with_values=True, storage_rows=1024,
    )
    bank.weights.load(0, model)
    eager_model = model.cpu()
    offset = 0
    for seed, rows in ((2, 200), (3, 37), (4, 256)):
        observations, masks = _inputs(rows, seed)
        staged_rows(bank.host_observations, 0, bank.bucket(rows), rows)[:] = observations
        staged_rows(bank.host_masks, 0, bank.bucket(rows), rows)[:] = masks
        bank.submit(bank.bucket(rows), offset)
        torch.cuda.synchronize()
        logits, values, effective = _eager(eager_model, observations, masks)
        recorded = slice(offset, offset + rows)
        actions = bank.actions[recorded].cpu()
        assert torch.equal(actions, bank.host_actions[:rows])
        assert torch.equal(bank.masks[recorded].cpu(), torch.from_numpy(effective))
        assert torch.equal(bank.states[recorded].cpu(), torch.from_numpy(observations))
        legal = torch.from_numpy(effective)
        legal[~legal.any(dim=1), END_TURN_INDEX] = True
        assert legal.gather(1, actions.unsqueeze(1)).all()
        expected_log_probs = torch.log_softmax(logits, dim=1).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)
        torch.testing.assert_close(
            bank.log_probs[recorded].cpu(), expected_log_probs, rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            bank.values[recorded].cpu(), values, rtol=1e-4, atol=1e-4
        )
        offset += rows


@cuda
def test_snapshot_bank_argmax_matches_each_slots_model():
    device = torch.device("cuda")
    models = [_model(seed) for seed in (5, 6, 7)]
    bank = GraphedPolicyBank(
        models[0].to(device), 3, 128, device,
        temperature=0.0, with_values=False,
    )
    for slot, model in enumerate(models):
        bank.weights.load(slot, model)
    counts = (100, 0, 41)
    bucket = bank.bucket(max(counts))
    inputs = []
    for slot, count in enumerate(counts):
        observations, masks = _inputs(max(count, 1), 10 + slot)
        observations, masks = observations[:count], masks[:count]
        staged_rows(bank.host_observations, slot, bucket, count)[:] = observations
        staged_rows(bank.host_masks, slot, bucket, count)[:] = masks
        inputs.append((observations, masks))
    bank.submit(bucket)
    torch.cuda.synchronize()
    for slot, ((observations, masks), model) in enumerate(zip(inputs, models)):
        if not len(observations):
            continue
        logits, _, _ = _eager(model.cpu(), observations, masks)
        expected = logits.argmax(dim=1)
        actual = staged_rows(bank.host_actions, slot, bucket, len(observations))
        assert np.array_equal(actual, expected.numpy())


@cuda
def test_sampling_follows_the_masked_policy_distribution():
    device = torch.device("cuda")
    model = _model(8)
    bank = GraphedPolicyBank(
        model.to(device), 1, 2048, device,
        temperature=1.0, with_values=False,
    )
    bank.weights.load(0, model)
    observations, _ = _inputs(2, 9)
    row_obs = observations[:1]
    # END_TURN plus three non-PLAY actions, so nothing is suppressed.
    row_mask = np.zeros((1, ACTION_DIM), dtype=bool)
    row_mask[0, [END_TURN_INDEX, 60, 61, 200]] = True
    staged_rows(bank.host_observations, 0, 2048, 2048)[:] = row_obs
    staged_rows(bank.host_masks, 0, 2048, 2048)[:] = row_mask
    counts = np.zeros(ACTION_DIM, dtype=np.int64)
    for _ in range(25):
        bank.submit(2048)
        torch.cuda.synchronize()
        counts += np.bincount(bank.host_actions[:2048].numpy(), minlength=ACTION_DIM)
    logits, _, _ = _eager(model.cpu(), row_obs, row_mask)
    probabilities = torch.softmax(logits[0], dim=0).numpy()
    empirical = counts / counts.sum()
    assert set(np.flatnonzero(counts)) <= set(np.flatnonzero(probabilities))
    assert 0.5 * np.abs(empirical - probabilities).sum() < 0.01


@cuda
def test_storage_growth_keeps_recorded_rows_and_recaptures():
    device = torch.device("cuda")
    model = _model(11)
    bank = GraphedPolicyBank(
        model.to(device), 1, 64, device,
        temperature=1.0, with_values=True, storage_rows=64,
    )
    bank.weights.load(0, model)
    observations, masks = _inputs(64, 12)
    staged_rows(bank.host_observations, 0, 64, 64)[:] = observations
    staged_rows(bank.host_masks, 0, 64, 64)[:] = masks
    bank.submit(64, 0)
    torch.cuda.synchronize()
    recorded = bank.states[:64].clone()
    bank.allocate_storage(256)
    assert bank.captured_buckets == []
    assert torch.equal(bank.states[:64], recorded)
    bank.submit(64, 64)
    torch.cuda.synchronize()
    assert torch.equal(bank.states[64:128].cpu(), torch.from_numpy(observations))
    with pytest.raises(ValueError, match="offset"):
        bank.submit(64, 256 + 64 + 1)
