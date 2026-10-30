"""Tests for the search uplift match harness."""
import json
import math

import pytest

from src.__main__ import parse_cli_args
from src.search.uplift import (
    UpliftConfig,
    elo_from_score,
    game_seed,
    run_uplift,
    sprt_llr,
    summarize,
    wilson_interval,
)


def test_config_rejects_invalid_ladders():
    with pytest.raises(ValueError, match="even"):
        UpliftConfig(model_path="m.pth", budgets=(0,), games=3)
    with pytest.raises(ValueError, match="duplicate"):
        UpliftConfig(model_path="m.pth", budgets=(4, 4), games=2)
    with pytest.raises(ValueError, match="non-negative"):
        UpliftConfig(model_path="m.pth", budgets=(-1,), games=2)
    with pytest.raises(ValueError, match="baseline"):
        UpliftConfig(model_path="m.pth", budgets=(0,), games=2, baseline="oracle")


def test_statistics_helpers():
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-3)
    assert high == pytest.approx(0.5962, abs=1e-3)
    assert elo_from_score(0.5) == pytest.approx(0.0)
    assert elo_from_score(10 / 11) == pytest.approx(400.0)
    assert sprt_llr(0, 0) == 0.0
    assert sprt_llr(600, 1000) > 2.94 > sprt_llr(500, 1000)
    assert game_seed(1, 2) == game_seed(1, 2) != game_seed(1, 3)


def test_summary_counts_seats_and_search_cost():
    rows = [
        {"budget": 4, "index": index, "search_first": index % 2 == 0,
         "search_won": index != 3, "turns": 20,
         "search": {"decisions": 10, "forced": 2, "deviations": 1, "rollouts": 32, "elapsed_s": 0.4}}
        for index in range(4)
    ]
    summary = summarize(rows, (4,))[0]

    assert summary["games"] == 4 and summary["wins"] == 3
    assert summary["first_seat"] == {"games": 2, "wins": 2}
    assert summary["second_seat"] == {"games": 2, "wins": 1}
    assert summary["searched_decisions"] == 32
    assert summary["deviations_from_greedy"] == 4
    assert summary["mean_search_ms"] == pytest.approx(1000 * 1.6 / 32)
    assert math.isfinite(summary["elo"])


def test_search_eval_cli_defaults():
    args = parse_cli_args(["search-eval", "--model", "m.pth"])
    assert args.budgets == [0, 16, 64, 256]
    assert args.baseline == "greedy"
    assert args.games % 2 == 0


def test_run_uplift_plays_paired_exact_seat_games_and_resumes(tmp_path, policy_checkpoint):
    config = UpliftConfig(
        model_path=policy_checkpoint, budgets=(0, 2), games=2,
        baseline="greedy", max_candidates=2, seed=3, device="cpu", workers=1,
    )
    output = tmp_path / "uplift.json"
    logs = []

    payload = run_uplift(config, output_json=str(output), log_fn=logs.append)

    assert len(payload["games"]) == 4
    for row in payload["summary"]:
        assert row["games"] == 2
        assert row["first_seat"]["games"] == 1 and row["second_seat"]["games"] == 1
    control = next(row for row in payload["summary"] if row["budget"] == 0)
    assert control["searched_decisions"] == 0
    searched = next(row for row in payload["summary"] if row["budget"] == 2)
    assert searched["searched_decisions"] > 0
    assert searched["rollouts"] <= 2 * searched["searched_decisions"]
    assert json.loads(output.read_text(encoding="utf-8"))["summary"] == payload["summary"]

    resumed = run_uplift(
        UpliftConfig(**{**config.__dict__, "workers": 1, "numba_threads": 1}),
        output_json=str(output), log_fn=logs.append,
    )
    assert [row["search_won"] for row in resumed["games"]] == [row["search_won"] for row in payload["games"]]
    assert any("remaining=0" in line for line in logs)

    with pytest.raises(ValueError, match="different configuration"):
        run_uplift(UpliftConfig(**{**config.__dict__, "seed": 4}), output_json=str(output), log_fn=logs.append)


def test_run_uplift_rejects_journals_from_another_search_implementation(tmp_path, policy_checkpoint):
    config = UpliftConfig(
        model_path=policy_checkpoint, budgets=(0,), games=2, device="cpu", workers=1,
    )
    output = tmp_path / "uplift.json"
    identity = {
        key: (list(value) if key == "budgets" else value)
        for key, value in config.__dict__.items()
        if key not in ("workers", "numba_threads", "batch_size")
    }
    (tmp_path / "uplift.json.games.jsonl").write_text(json.dumps({"config": identity}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="different configuration"):
        run_uplift(config, output_json=str(output), log_fn=lambda *_: None)
