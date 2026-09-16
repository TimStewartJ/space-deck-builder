"""Summarize search uplift evidence for results/search_uplift.md.

Usage: python results/search_uplift/summarize_search_uplift.py [evidence_dir]

Checks that every run played each (budget, index) exactly once with exact
seat alternation, then reports per-budget win rates, paired McNemar tests on
identical deals, and the budget-64 result pooled across the three seeds.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.search.uplift import elo_from_score, sprt_llr, wilson_interval  # noqa: E402

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
RUNS = (
    ("selfplay_seed0_ladder", "selfplay_seed0"),
    ("selfplay_seed1_budget64", "selfplay_seed1"),
    ("selfplay_seed2_budget64", "selfplay_seed2"),
)


def outcomes(payload: dict, budget: int) -> dict[int, bool]:
    return {row["index"]: row["search_won"] for row in payload["games"] if row["budget"] == budget}


def mcnemar(a: dict[int, bool], b: dict[int, bool]) -> tuple[int, int, float]:
    """Deals won only by arm a, only by arm b, and the exact two-sided p-value."""
    if set(a) != set(b):
        raise ValueError("paired arms must cover the same deals")
    only_a = sum(1 for index in a if a[index] and not b[index])
    only_b = sum(1 for index in a if b[index] and not a[index])
    n = only_a + only_b
    k = min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return only_a, only_b, p


def verify(payload: dict, name: str) -> None:
    config = payload["config"]
    games = config["games"]
    for budget in config["budgets"]:
        rows = [row for row in payload["games"] if row["budget"] == budget]
        indices = sorted(row["index"] for row in rows)
        if indices != list(range(games)):
            raise ValueError(f"{name} budget {budget} does not cover every deal exactly once")
        if any(row["search_first"] != (row["index"] % 2 == 0) for row in rows):
            raise ValueError(f"{name} budget {budget} breaks exact seat alternation")


def rate(wins: int, games: int) -> str:
    low, high = wilson_interval(wins, games)
    return f"{100 * wins / games:.1f}% ({100 * low:.1f}–{100 * high:.1f})"


def elo(wins: int, games: int) -> str:
    low, high = wilson_interval(wins, games)
    return f"{elo_from_score(wins / games):+.0f} ({elo_from_score(low):+.0f} to {elo_from_score(high):+.0f})"


def describe(payload: dict, label: str) -> None:
    config = payload["config"]
    print(f"### {label}: {config['games']} deals per budget, baseline {config['baseline']}")
    print()
    print("| Budget | Search wins | Win rate (95% CI) | Elo vs greedy (95% CI) | SPRT LLR (0 vs +20) | First seat | Second seat | Decisions changed | Mean search |")
    print("|---:|---:|---|---|---:|---:|---:|---:|---:|")
    for row in payload["summary"]:
        first, second = row["first_seat"], row["second_seat"]
        changed = (
            f"{100 * row['deviations_from_greedy'] / row['searched_decisions']:.1f}%"
            if row["searched_decisions"] else "—"
        )
        search_ms = f"{row['mean_search_ms']:.0f} ms" if row["mean_search_ms"] else "—"
        print(
            f"| {row['budget']} | {row['wins']}/{row['games']} | {rate(row['wins'], row['games'])} | "
            f"{elo(row['wins'], row['games'])} | {row['sprt_llr_0_20']:+.1f} | "
            f"{100 * first['wins'] / first['games']:.1f}% | {100 * second['wins'] / second['games']:.1f}% | "
            f"{changed} | {search_ms} |"
        )
    print()
    budgets = config["budgets"]
    for index, low_budget in enumerate(budgets):
        for high_budget in budgets[index + 1:]:
            high, low = outcomes(payload, high_budget), outcomes(payload, low_budget)
            only_high, only_low, p = mcnemar(high, low)
            delta, delta_low, delta_high = paired_difference(only_high, only_low, len(high))
            base = sum(low.values()) / len(low)
            print(
                f"- budget {high_budget} vs budget {low_budget} on identical deals: {only_high} deals "
                f"won only at {high_budget}, {only_low} only at {low_budget}; difference "
                f"{100 * delta:+.1f}pp (paired 95% CI {100 * delta_low:+.1f} to {100 * delta_high:+.1f}pp, "
                f"{elo_from_score(base + delta_low) - elo_from_score(base):+.0f} to "
                f"{elo_from_score(base + delta_high) - elo_from_score(base):+.0f} Elo); exact McNemar p = {p:.2g}"
            )
    print()


def paired_difference(only_a: int, only_b: int, n: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    """Win-rate difference between two arms on the same n deals, with a 95% CI."""
    delta = (only_a - only_b) / n
    variance = ((only_a + only_b) - (only_a - only_b) ** 2 / n) / n**2
    half = z * math.sqrt(max(variance, 0.0))
    return delta, delta - half, delta + half


payloads = {}
for name, label in RUNS:
    path = ROOT / f"{name}.json"
    if not path.exists():
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify(payload, name)
    payloads[name] = payload
    describe(payload, label)

if len(payloads) == len(RUNS):
    wins = games = only_search = only_control = 0
    control_wins = 0
    for name, _ in RUNS:
        payload = payloads[name]
        searched, control = outcomes(payload, 64), outcomes(payload, 0)
        wins += sum(searched.values())
        control_wins += sum(control.values())
        games += len(searched)
        a, b, _ = mcnemar(searched, control)
        only_search += a
        only_control += b
    n = only_search + only_control
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(only_search, only_control) + 1)) / 2**n)
    print("### Budget 64 pooled across seeds 0, 1, and 2")
    print()
    print(f"- search: {wins}/{games} = {rate(wins, games)}, Elo {elo(wins, games)}, SPRT LLR {sprt_llr(wins, games):+.1f}")
    print(f"- control (budget 0): {control_wins}/{games} = {rate(control_wins, games)}")
    print(f"- paired: {only_search} deals won only with search, {only_control} only without; exact McNemar p = {p:.2g}")
