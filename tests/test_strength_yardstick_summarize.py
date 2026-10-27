"""The strength yardstick's summarizer and the reading rules it applies.

``results/strength_yardstick/summarize.py`` turns tournament, league and
training files into the verdicts pre-registered by decisions 0013 and 0014,
so these tests pin each rule's thresholds, the provisional states, and the
file conventions it relies on.
"""
from __future__ import annotations

import importlib.util
import json
import statistics
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "strength_yardstick_summarize", REPO / "results" / "strength_yardstick" / "summarize.py"
)
summarize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summarize)

CHAMPION = "selfplay_seed0@T0"
PEERS = ("selfplay_seed1@T0", "selfplay_seed2@T0")
GAMES = 4000


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _tournament(rates: dict[tuple[str, str], float]) -> dict:
    """An elo JSON whose pairings give player i the stated win rate against j."""
    labels = sorted({label for pair in rates for label in pair})
    return {
        "participants": [{"label": label, "elo": 1000.0} for label in labels],
        "pairings": [
            {"label_i": i, "label_j": j, "wins_i": round(rate * GAMES), "games": GAMES}
            for (i, j), rate in rates.items()
        ],
    }


def _league(rates: dict[str, float], complete: bool = True) -> dict:
    return {
        "checkpoints": [
            {
                "label": label,
                "opponents": {},
                "pooled": {
                    "wins": round(rate * 24000),
                    "games": 24000,
                    "win_rate": rate,
                    "win_rate_ci95": [rate - 0.005, rate + 0.005],
                    "complete": complete,
                },
            }
            for label, rate in rates.items()
        ]
    }


def _arm_runs(tmp_path: Path, E, C, *, versus=0.52, O_exploiter=0.6, O_control=0.57,
              L_exploiter=0.84, L_control=0.82, league_complete=True) -> Path:
    for seed in range(3):
        exploiter, control = f"exploiter_seed{seed}@T0", f"control_seed{seed}@T0"
        rates = {
            (exploiter, CHAMPION): E[seed],
            (control, CHAMPION): C[seed],
            (exploiter, control): versus,
        }
        for peer in PEERS:
            rates[(exploiter, peer)] = O_exploiter
            rates[(control, peer)] = O_control
        _write(tmp_path / "scores" / f"score_seed{seed}.json", _tournament(rates))
    league = {"selfplay_seed0": 0.792}
    for seed in range(3):
        league[f"exploiter_seed{seed}"] = L_exploiter
        league[f"control_seed{seed}"] = L_control
    _write(tmp_path / "league" / "league.json", _league(league, complete=league_complete))
    return tmp_path


def _ladder_runs(tmp_path: Path, W, *, L=0.81, league_complete=True) -> Path:
    for seed in range(3):
        ladder = f"ladder_seed{seed}@T0"
        rates = {(ladder, CHAMPION): W[seed]}
        for peer in PEERS:
            rates[(ladder, peer)] = 0.6
        _write(tmp_path / "scores" / f"ladder_score_seed{seed}.json", _tournament(rates))
    _write(
        tmp_path / "league" / "league_ladder.json",
        _league({f"ladder_seed{seed}": L for seed in range(3)}, complete=league_complete),
    )
    return tmp_path


def test_mean_interval_uses_student_t_with_two_degrees_of_freedom():
    values = [0.039, 0.069, 0.026]
    stat = summarize.mean_interval(values)
    half = 4.302652729911275 * statistics.stdev(values) / 3 ** 0.5

    assert stat["mean"] == pytest.approx(0.044666666, abs=1e-8)
    assert stat["low"] == pytest.approx(stat["mean"] - half, abs=1e-12)
    assert stat["high"] == pytest.approx(stat["mean"] + half, abs=1e-12)
    assert stat["low"] < 0 < stat["high"]


def test_win_rate_reads_either_seat_order():
    score = _tournament({("a@T0", "b@T0"): 0.6})

    assert summarize.win_rate(score, "a@T0", "b@T0") == pytest.approx(0.6)
    assert summarize.win_rate(score, "b@T0", "a@T0") == pytest.approx(0.4)
    assert summarize.win_rate(score, "a@T0", "c@T0") is None


def test_league_scores_skip_unfinished_and_empty_rows():
    league = _league({"done": 0.8})
    league["checkpoints"] += _league({"partial": 0.9}, complete=False)["checkpoints"]
    league["checkpoints"].append({"label": "empty", "opponents": {}, "pooled": {"games": 0}})

    assert set(summarize.league_scores(league)) == {"done"}


def test_exploiter_verdict_needs_five_points_with_an_interval_above_zero(tmp_path):
    runs = _arm_runs(tmp_path, E=[0.593, 0.603, 0.582], C=[0.555, 0.534, 0.556])

    verdict = summarize.summarize(str(runs))["verdict"]

    assert verdict["provisional"] is False
    assert verdict["E_minus_C"]["mean"] == pytest.approx(0.0443333, abs=1e-6)
    assert verdict["exploitable"] is False
    assert verdict["kind"] is None
    assert verdict["control_helps"] is False


@pytest.mark.parametrize(
    "L_exploiter, O_exploiter, kind",
    [(0.81, 0.55, "narrow"), (0.84, 0.55, "general"), (0.81, 0.60, "general")],
)
def test_exploitable_verdict_is_narrow_only_if_no_better_all_round(tmp_path, L_exploiter, O_exploiter, kind):
    runs = _arm_runs(
        tmp_path, E=[0.62, 0.62, 0.62], C=[0.55, 0.55, 0.55],
        L_exploiter=L_exploiter, L_control=0.82, O_exploiter=O_exploiter, O_control=0.57,
    )

    verdict = summarize.summarize(str(runs))["verdict"]

    assert verdict["exploitable"] is True
    assert verdict["kind"] == kind
    assert verdict["control_helps"] is True


def test_exploiter_verdict_is_provisional_until_the_league_is_finished(tmp_path):
    runs = _arm_runs(tmp_path, E=[0.62, 0.62, 0.62], C=[0.55, 0.55, 0.55], league_complete=False)

    verdict = summarize.summarize(str(runs))["verdict"]

    assert verdict["provisional"] is True
    assert verdict["exploitable"] is True
    assert verdict["kind"] is None
    assert verdict["L_exploiter"] is None


@pytest.mark.parametrize(
    "W, exploitable",
    [([0.57, 0.58, 0.565], True), ([0.56, 0.50, 0.60], False), ([0.54, 0.545, 0.55], False)],
)
def test_ladder_verdict_needs_55_percent_with_an_interval_above_half(tmp_path, W, exploitable):
    runs = _ladder_runs(tmp_path, W)

    verdict = summarize.summarize_ladder(str(runs), champion_league=0.792)["verdict"]

    assert verdict["provisional"] is False
    assert verdict["exploitable"] is exploitable


@pytest.mark.parametrize(
    "L, champion_league, kind",
    [(0.75, 0.792, "narrow"), (0.75, 0.75, "narrow"), (0.81, 0.792, "general")],
)
def test_ladder_verdict_is_narrow_when_its_league_is_no_better_than_the_champion(
    tmp_path, L, champion_league, kind
):
    runs = _ladder_runs(tmp_path, [0.57, 0.58, 0.565], L=L)

    verdict = summarize.summarize_ladder(str(runs), champion_league=champion_league)["verdict"]

    assert verdict["kind"] == kind


def test_ladder_verdict_is_provisional_until_its_league_is_finished(tmp_path):
    runs = _ladder_runs(tmp_path, [0.57, 0.58, 0.565], league_complete=False)

    verdict = summarize.summarize_ladder(str(runs), champion_league=0.792)["verdict"]

    assert verdict["provisional"] is True
    assert verdict["exploitable"] is True
    assert verdict["kind"] is None


def test_summary_reads_the_ladder_beside_the_arms(tmp_path):
    runs = _arm_runs(tmp_path, E=[0.593, 0.603, 0.582], C=[0.555, 0.534, 0.556])
    _ladder_runs(runs, [0.57, 0.58, 0.565], L=0.75)

    summary = summarize.summarize(str(runs))

    assert summary["ladder"]["verdict"]["champion_league"] == pytest.approx(0.792)
    assert summary["ladder"]["verdict"]["kind"] == "narrow"
    assert summarize.summarize_ladder(str(tmp_path / "missing"), champion_league=0.792) is None


def test_training_curve_lets_a_rerun_chunk_replace_the_attempt_it_repeats(tmp_path):
    run_root = tmp_path / "exploit-selfplay-seed0_seed0"
    frozen = "frozen:selfplay_seed0"
    for stamp, wins in (("20261027_010627", 5), ("20261027_014057", 15)):
        metrics = tmp_path / f"metrics_{stamp}.jsonl"
        rows = [
            {"update": update, "rollout": {"per_opponent": {frozen: {"wins": wins, "games": 20}}}}
            for update in (100, 101)
        ]
        metrics.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        log = run_root / "chunks" / f"upd100_to_125_{stamp}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(f"[01:00:00] Metrics file: {metrics}\n", encoding="utf-8")

    curve = summarize.training_curve(str(run_root), frozen)

    assert curve == {100: 0.75, 101: 0.75}
