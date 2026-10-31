"""Graph-replayed rollout policies agree with the eager model."""
import numpy as np
import pytest
import torch

from src.encoding.action_encoder import (
    END_TURN_INDEX,
    get_action_space_layout,
    get_action_space_size,
)
from src.encoding.state_encoder import get_state_size
from src.ppo.ppo_actor_critic import PPOActorCritic
from src.vector_engine.graphed_policy import (
    GraphedRolloutPolicy,
    batch_buckets,
    graphable,
    stage_effective_masks,
    staged_rows,
)
from src.vector_engine.policy import suppress_redundant_end_turn

NUM_CARDS = 49
STATE_DIM = get_state_size([""] * NUM_CARDS)
ACTION_DIM = get_action_space_size([""] * NUM_CARDS)
_LAYOUT = get_action_space_layout(NUM_CARDS)
PLAY = next(
    (offset, offset + count)
    for name, offset, count in _LAYOUT["card_groups"]
    if name == "play_card"
)
ATTACK_PLAYER = _LAYOUT["global_indices"]["attack_player"]

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
    masks[rows // 2 :, PLAY[0] : PLAY[1]] = False
    masks[rows // 2 :, ATTACK_PLAYER] = False  # END_TURN stays legal here
    return observations, masks


def _expected_masks(masks: np.ndarray) -> np.ndarray:
    effective = suppress_redundant_end_turn(masks)
    effective[~effective.any(axis=1), END_TURN_INDEX] = True
    return effective


def _stage(policy, observations, masks, slot, bucket):
    count = len(observations)
    staged_rows(policy.host_observations, slot, bucket, count)[:] = observations
    policy.stage_masks(masks, np.arange(count, dtype=np.int64), slot, bucket)


def _eager(model, observations, masks):
    effective = _expected_masks(masks)
    with torch.no_grad():
        logits, values = model(torch.from_numpy(observations))
    logits = logits.masked_fill(~torch.from_numpy(effective), float("-inf"))
    return logits, values, effective


def test_batch_buckets_cover_every_size():
    buckets = batch_buckets(2000)
    assert buckets[0] == 16 and buckets[-1] == 2000
    assert buckets == sorted(set(buckets))
    for rows in range(1, 2001):
        bucket = next(size for size in buckets if size >= rows)
        assert bucket - rows < max(16, rows // 4 + 1)
    assert batch_buckets(10) == [10]


def test_staged_masks_match_suppress_redundant_end_turn():
    rng = np.random.default_rng(3)
    masks = rng.random((300, ACTION_DIM)) < 0.02
    masks[:, END_TURN_INDEX] = rng.random(300) < 0.8
    masks[:40] = False
    masks[:20, END_TURN_INDEX] = True
    rows = rng.permutation(300)[:250].astype(np.int64)
    out = np.zeros((250, ACTION_DIM), dtype=np.bool_)
    stage_effective_masks(
        masks, rows, out, PLAY[0], PLAY[1], ATTACK_PLAYER, END_TURN_INDEX
    )
    expected = suppress_redundant_end_turn(masks[rows])
    empty = ~expected.any(axis=1)
    assert empty.any()
    expected[empty, END_TURN_INDEX] = True
    assert np.array_equal(out, expected)


def test_default_architecture_is_graphable():
    assert graphable(_model(0))


@cuda
def test_learner_records_eager_values_log_probs_and_masks():
    device = torch.device("cuda")
    model = _model(1)
    policy = GraphedRolloutPolicy(
        model.to(device), 0, 256, device, storage_rows=1024
    )
    policy.load_learner(model)
    eager_model = model.cpu()
    offset = 0
    for seed, rows in ((2, 200), (3, 37), (4, 256)):
        observations, masks = _inputs(rows, seed)
        bucket = policy.bucket(rows)
        _stage(policy, observations, masks, 0, bucket)
        policy.submit(bucket, [rows], offset=offset)
        torch.cuda.synchronize()
        logits, values, effective = _eager(eager_model, observations, masks)
        recorded = slice(offset, offset + rows)
        actions = policy.actions[recorded].cpu()
        assert torch.equal(actions, policy.host_actions[:rows])
        assert torch.equal(policy.masks[recorded].cpu(), torch.from_numpy(effective))
        assert torch.equal(
            policy.states[recorded].cpu(), torch.from_numpy(observations)
        )
        assert torch.from_numpy(effective).gather(1, actions.unsqueeze(1)).all()
        expected_log_probs = torch.log_softmax(logits, dim=1).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)
        torch.testing.assert_close(
            policy.log_probs[recorded].cpu(), expected_log_probs, rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            policy.values[recorded].cpu(), values, rtol=1e-4, atol=1e-4
        )
        offset += rows


@cuda
def test_one_graph_serves_a_sampling_learner_and_greedy_opponents():
    device = torch.device("cuda")
    learner, *opponents = [_model(seed) for seed in (4, 5, 6, 7)]
    policy = GraphedRolloutPolicy(
        learner.to(device), 3, 128, device,
        opponent_temperature=0.0, storage_rows=512,
    )
    policy.load_learner(learner)
    for slot, model in enumerate(opponents, start=1):
        policy.load_opponent(slot, model)
    counts = [50, 100, 0, 41]
    bucket = policy.bucket(max(counts))
    inputs = []
    for slot, count in enumerate(counts):
        observations, masks = _inputs(max(count, 2), 10 + slot)
        observations, masks = observations[:count], masks[:count]
        _stage(policy, observations, masks, slot, bucket)
        inputs.append((observations, masks))
    policy.submit(bucket, counts, offset=7)
    torch.cuda.synchronize()
    for slot, ((observations, masks), model) in enumerate(
        zip(inputs, [learner, *opponents])
    ):
        if not len(observations):
            continue
        logits, values, effective = _eager(model.cpu(), observations, masks)
        actual = staged_rows(policy.host_actions, slot, bucket, len(observations))
        if slot:
            assert np.array_equal(actual, logits.argmax(dim=1).numpy())
            continue
        recorded = slice(7, 7 + len(observations))
        actions = torch.from_numpy(actual.copy())
        assert torch.equal(policy.actions[recorded].cpu(), actions)
        assert torch.from_numpy(effective).gather(1, actions.unsqueeze(1)).all()
        torch.testing.assert_close(
            policy.log_probs[recorded].cpu(),
            torch.log_softmax(logits, dim=1).gather(1, actions.unsqueeze(1)).squeeze(1),
            rtol=1e-4, atol=1e-4,
        )
        torch.testing.assert_close(
            policy.values[recorded].cpu(), values, rtol=1e-4, atol=1e-4
        )


@cuda
def test_sampling_follows_the_masked_policy_distribution():
    device = torch.device("cuda")
    model = _model(8)
    policy = GraphedRolloutPolicy(model.to(device), 0, 2048, device)
    policy.load_learner(model)
    observations, _ = _inputs(2, 9)
    row_mask = np.zeros((1, ACTION_DIM), dtype=bool)
    row_mask[0, [END_TURN_INDEX, 60, 61, 200]] = True
    _stage(
        policy,
        np.repeat(observations[:1], 2048, axis=0),
        np.repeat(row_mask, 2048, axis=0),
        0,
        2048,
    )
    counts = np.zeros(ACTION_DIM, dtype=np.int64)
    for _ in range(25):
        policy.submit(2048, [2048])
        torch.cuda.synchronize()
        counts += np.bincount(policy.host_actions[:2048].numpy(), minlength=ACTION_DIM)
    logits, _, _ = _eager(model.cpu(), observations[:1], row_mask)
    probabilities = torch.softmax(logits[0], dim=0).numpy()
    empirical = counts / counts.sum()
    assert set(np.flatnonzero(counts)) <= set(np.flatnonzero(probabilities))
    assert 0.5 * np.abs(empirical - probabilities).sum() < 0.01


@cuda
def test_storage_growth_keeps_recorded_rows_and_recaptures():
    device = torch.device("cuda")
    model = _model(11)
    policy = GraphedRolloutPolicy(model.to(device), 0, 64, device, storage_rows=64)
    policy.load_learner(model)
    observations, masks = _inputs(64, 12)
    _stage(policy, observations, masks, 0, 64)
    policy.submit(64, [64], offset=0)
    torch.cuda.synchronize()
    recorded = policy.states[:64].clone()
    policy.allocate_storage(256)
    assert policy.captured_graphs == []
    assert torch.equal(policy.states[:64], recorded)
    policy.submit(64, [64], offset=64)
    torch.cuda.synchronize()
    assert torch.equal(policy.states[64:128].cpu(), torch.from_numpy(observations))
    with pytest.raises(ValueError, match="offset"):
        policy.submit(64, [64], offset=256 - 63)


@cuda
def test_capturing_a_graph_leaves_the_random_stream_unchanged():
    device = torch.device("cuda")
    model = _model(14).to(device)
    observations, masks = _inputs(100, 15)

    def sample(capture_before_seeding: bool) -> torch.Tensor:
        policy = GraphedRolloutPolicy(model, 0, 128, device)
        policy.load_learner(model)
        bucket = policy.bucket(100)
        _stage(policy, observations, masks, 0, bucket)
        if capture_before_seeding:
            before = torch.cuda.get_rng_state(device)
            policy._graph(bucket, 1)
            assert torch.equal(torch.cuda.get_rng_state(device), before)
        torch.cuda.manual_seed(16)
        policy.submit(bucket, [100])
        torch.cuda.synchronize()
        return policy.host_actions[:100].clone()

    assert torch.equal(sample(True), sample(False))


@cuda
def test_vector_evaluation_alternates_seats_and_is_deterministic_at_temperature_zero():
    from src.config import DataConfig, PPOConfig
    from src.ppo.ppo_eval import evaluate_with_vector_runner
    from src.ppo.vector_batch_runner import VectorBatchRunner

    data = DataConfig()
    cards = data.load_cards()
    registry = data.build_registry(cards)
    device = torch.device("cuda")
    runner = VectorBatchRunner(
        model=_model(13).to(device),
        cards=cards,
        registry=registry,
        device=device,
        ppo_config=PPOConfig(),
        opponent_spec="random,heuristic,simple",
    )
    assert runner.can_evaluate()
    results = [
        evaluate_with_vector_runner(
            runner,
            opponents="heuristic,random,simple",
            eval_games=61,
            seed=4,
            temperature=0.0,
            log_fn=lambda _: None,
        )
        for _ in range(2)
    ]
    first, second = results
    assert [r.games for r in first.per_opponent] == [21, 20, 20]
    for result in first.per_opponent:
        assert result.first_player_games == (result.games + 1) // 2
        assert result.second_player_games == result.games // 2
        assert result.avg_steps > 0
    assert [(r.wins, r.avg_steps) for r in first.per_opponent] == [
        (r.wins, r.avg_steps) for r in second.per_opponent
    ]
    assert first.total_games == 61
