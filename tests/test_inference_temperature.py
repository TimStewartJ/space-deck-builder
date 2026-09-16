"""Tests for inference-temperature action selection in eval and Elo paths."""
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from src.__main__ import parse_cli_args
from src.ppo.action_selection import select_actions, validate_temperature
from src.ppo.mp_inference_server import (
    InferenceRequest,
    InferenceServer,
    _forward_and_sample,
)

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO / "results" / "optimized_selfplay_baseline" / "checkpoints" / "selfplay_seed0.pth"


class _FixedPolicy(torch.nn.Module):
    """Returns the same logits and value for every row."""

    def __init__(self, logits, value=0.25):
        super().__init__()
        self.register_buffer("row", torch.tensor(logits, dtype=torch.float32))
        self.value = value

    def forward(self, states):
        batch = states.shape[0]
        return self.row.expand(batch, -1).clone(), torch.full((batch,), self.value)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), "hot"])
def test_invalid_temperatures_are_rejected(value):
    with pytest.raises(ValueError, match="temperature"):
        validate_temperature(value)


def test_zero_temperature_selects_first_highest_legal_action():
    logits = torch.tensor([
        [0.0, 3.0, 3.0, -1.0],
        [5.0, float("-inf"), 1.0, 1.0],
    ])
    actions, log_probs = select_actions(logits, 0.0)

    assert actions.tolist() == [1, 0]
    assert torch.equal(log_probs, torch.zeros(2))


def test_unit_temperature_samples_only_legal_actions_with_policy_log_probs():
    torch.manual_seed(0)
    logits = torch.tensor([[0.0, float("-inf"), 1.0, float("-inf")]]).repeat(512, 1)
    actions, log_probs = select_actions(logits, 1.0)

    assert set(actions.tolist()) <= {0, 2}
    expected = torch.distributions.Categorical(logits=logits).log_prob(actions)
    assert torch.allclose(log_probs, expected)


def test_intermediate_temperature_sharpens_the_distribution():
    torch.manual_seed(1)
    logits = torch.tensor([[0.0, math.log(2.0)]]).repeat(40000, 1)
    actions, log_probs = select_actions(logits, 0.5)

    # softmax([0, ln 2] / 0.5) = [1/5, 4/5]
    assert abs(actions.float().mean().item() - 0.8) < 0.01
    assert torch.allclose(log_probs[actions == 1], torch.log(torch.tensor(0.8)))


def test_forward_and_sample_applies_temperature_without_changing_values():
    model = _FixedPolicy([0.1, 0.4, 2.0, 0.3])
    states = torch.zeros((3, 8))
    masks = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 1], [0, 0, 0, 0]], dtype=torch.uint8)

    actions, log_probs, values, masked = _forward_and_sample(model, states, masks, 0.0)

    assert actions.tolist() == [2, 1, 1]  # the empty row is forced to END_TURN
    assert torch.equal(log_probs, torch.zeros(3))
    assert torch.allclose(values, torch.full((3,), 0.25))
    assert masked[1, 2] == float("-inf")


def test_inference_server_applies_temperature_per_model_id():
    import multiprocessing as mp

    model = _FixedPolicy([0.0, 0.0, 1.0, 0.9])
    server = InferenceServer(
        models={"sampled": model, "greedy": _FixedPolicy([0.0, 0.0, 1.0, 0.9])},
        device=torch.device("cpu"),
        num_workers=1,
        ctx=mp.get_context("spawn"),
        temperatures={"greedy": 0.0},
    )
    assert server.temperature_for("sampled") == 1.0
    assert server.temperature_for("greedy") == 0.0
    server.start()
    try:
        states = np.zeros((256, 8), dtype=np.float32)
        masks = np.ones((256, 4), dtype=np.uint8)
        server.request_queue.put(InferenceRequest(
            worker_id=0, request_id=1, states=states, masks=masks, model_id="greedy",
        ))
        greedy = server.response_queues[0].get(timeout=10.0)
        server.request_queue.put(InferenceRequest(
            worker_id=0, request_id=2, states=states, masks=masks, model_id="sampled",
        ))
        sampled = server.response_queues[0].get(timeout=10.0)

        assert greedy.error is None and sampled.error is None
        assert set(greedy.action_indices.tolist()) == {2}
        assert np.all(greedy.log_probs == 0)
        assert len(set(sampled.action_indices.tolist())) > 1

        server.register_model("cool", _FixedPolicy([0.0, 0.0, 1.0, 0.9]), temperature=0)
        assert server.temperature_for("cool") == 0.0
        server.unregister_model("cool")
        assert server.temperature_for("cool") == 1.0
    finally:
        server.stop()


def test_inference_server_rejects_temperatures_for_unknown_models():
    with pytest.raises(ValueError, match="unknown model ids"):
        InferenceServer(
            models={"a": _FixedPolicy([0.0, 1.0])},
            device=torch.device("cpu"),
            num_workers=1,
            temperatures={"b": 0.0},
        )


def test_eval_and_elo_cli_parse_temperatures():
    args = parse_cli_args(["eval", "--temperature", "0", "--output-json", "out.json"])
    assert args.temperature == 0.0
    assert args.output_json == "out.json"
    assert parse_cli_args(["eval"]).temperature == 1.0

    elo = parse_cli_args(["elo", "--checkpoint-temperatures", "1,0.5,0"])
    assert elo.checkpoint_temperatures == [1.0, 0.5, 0.0]
    assert parse_cli_args(["elo"]).checkpoint_temperatures == [1.0]

    with pytest.raises(SystemExit):
        parse_cli_args(["eval", "--temperature", "-0.1"])
    with pytest.raises(SystemExit):
        parse_cli_args(["elo", "--checkpoint-temperatures", "0,0"])


def test_evaluate_passes_temperature_to_runner(monkeypatch):
    from src.config import DataConfig
    from src.ppo import ppo_eval

    calls = []

    class FakeRunner:
        def __init__(self, *args, **kwargs):
            self.eval_seat_counts = {"first": 0, "second": 0}

        def run_eval(self, num_games, *, alternate_seats, temperature):
            calls.append((num_games, alternate_seats, temperature))
            self.eval_seat_counts = {"first": num_games // 2, "second": num_games - num_games // 2}
            return num_games, 0, num_games * 10

    monkeypatch.setattr(ppo_eval, "BatchRunner", FakeRunner)
    result = ppo_eval.evaluate(
        torch.nn.Linear(1, 1), data_cfg=DataConfig(), device="cpu",
        opponents="random,heuristic", eval_games=8, num_workers=1,
        temperature=0.0, log_fn=lambda *_: None,
    )

    assert calls == [(4, True, 0.0), (4, True, 0.0)]
    assert result.temperature == 0.0
    assert result.overall_win_rate == 1.0


def test_write_eval_json_records_provenance(tmp_path):
    from src.ppo.ppo_eval import EvalResult, OpponentResult, write_eval_json

    result = EvalResult(
        per_opponent=[OpponentResult("heuristic", 9, 10, 0.9, 140.0, 5, 5)],
        total_wins=9, total_games=10, overall_win_rate=0.9,
        elapsed_seconds=1.5, temperature=0.0,
    )
    path = tmp_path / "nested" / "eval.json"
    write_eval_json(str(path), result, model_path="m.pth", seed=3, opponents="heuristic")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["temperature"] == 0.0
    assert payload["seed"] == 3
    assert payload["model"] == "m.pth"
    assert payload["per_opponent"][0]["first_player_games"] == 5


def test_build_participants_expands_checkpoint_temperatures():
    from src.ppo.elo_tournament import _participant_temperatures, build_participants

    participants = build_participants(
        [str(CHECKPOINT)], ["heuristic"], checkpoint_temperatures=[1.0, 0.0]
    )

    assert [p.label for p in participants] == ["selfplay_seed0", "selfplay_seed0@T0", "heuristic"]
    assert _participant_temperatures(participants) == {
        "selfplay_seed0": 1.0,
        "selfplay_seed0@T0": 0.0,
    }
    with pytest.raises(ValueError, match="share labels"):
        build_participants([str(CHECKPOINT), str(CHECKPOINT)])


def test_tournament_json_lists_participants_and_pairings(tmp_path):
    from src.ppo.elo_tournament import BuiltinParticipant, _write_tournament_json

    participants = [BuiltinParticipant("random", "random"), BuiltinParticipant("simple", "simple")]
    path = tmp_path / "elo.json"
    _write_tournament_json(
        str(path), participants=participants, ratings=[1000.0, 1200.0],
        match_wins={(0, 1): 10}, match_games={(0, 1): 40}, games_per_pair=40,
        wins_total=[10, 30], losses_total=[30, 10], games_total=[40, 40],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["anchor"] == "random"
    assert payload["participants"][1]["elo"] == 1200.0
    assert payload["pairings"] == [
        {"i": 0, "j": 1, "label_i": "random", "label_j": "simple", "wins_i": 10, "games": 40}
    ]


def test_greedy_batch_runner_eval_is_deterministic_and_seat_balanced():
    from src.ai.random_agent import RandomAgent
    from src.config import DataConfig, ModelConfig, load_checkpoint
    from src.encoding.action_encoder import get_action_space_size
    from src.encoding.state_encoder import get_state_size
    from src.ppo.batch_runner import BatchRunner
    from src.ppo.ppo_actor_critic import PPOActorCritic

    data_cfg = DataConfig()
    cards = data_cfg.load_cards()
    registry = data_cfg.build_registry(cards)
    names = registry.card_names
    checkpoint = load_checkpoint(str(CHECKPOINT), map_location="cpu")
    model = PPOActorCritic(
        get_state_size(names), get_action_space_size(names), len(names),
        model_config=ModelConfig.from_dict(checkpoint["config"]["model"]),
        card_registry=registry,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    outcomes = []
    for _ in range(2):
        random.seed(7)
        runner = BatchRunner(
            model=model, card_names=names, cards=cards,
            action_dim=get_action_space_size(names), device=torch.device("cpu"),
            opponent_factory=lambda: RandomAgent("Rand"),
            num_concurrent=4, registry=registry,
        )
        outcomes.append(runner.run_eval(4, alternate_seats=True, temperature=0.0))
        assert runner.eval_seat_counts == {"first": 2, "second": 2}

    assert outcomes[0] == outcomes[1]
