"""Summarize batched search evidence for results/batched_search.md.

Usage: python results/batched_search/summarize_batched_search.py [evidence_dir]
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
PUBLISHED = ROOT.parent / "search_uplift" / "selfplay_seed0_ladder.json"


def games(path: Path, budget: int) -> dict[int, bool]:
    rows = json.loads(path.read_text(encoding="utf-8"))["games"]
    return {row["index"]: row["search_won"] for row in rows if row["budget"] == budget}


def paired(new: dict[int, bool], old: dict[int, bool]) -> tuple[int, int, int]:
    if set(new) != set(old):
        raise ValueError("paired runs must cover the same deals")
    only_new = sum(1 for index in new if new[index] and not old[index])
    only_old = sum(1 for index in new if old[index] and not new[index])
    return only_new, only_old, len(new)


def report(label: str, only_new: int, only_old: int, n: int) -> None:
    delta = (only_new - only_old) / n
    variance = ((only_new + only_old) - (only_new - only_old) ** 2 / n) / n**2
    half = 1.959963984540054 * math.sqrt(variance)
    m = only_new + only_old
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(min(only_new, only_old) + 1)) / 2**m)
    print(
        f"- {label}: {n} deals, {only_new} won only by the new implementation, "
        f"{only_old} only by the old; difference {100 * delta:+.1f}pp "
        f"(paired 95% CI {100 * (delta - half):+.1f} to {100 * (delta + half):+.1f}pp); "
        f"exact McNemar p = {p:.2f}"
    )


print("## Throughput")
for name in ("bench_decisions_v2", "bench_games_512_v2", "bench_games_1024_v2"):
    rows = json.loads((ROOT / f"{name}.json").read_text(encoding="utf-8"))["result"]["rows"]
    for row in rows:
        if "decisions_per_s" in row:
            print(f"- {name} batch {row['batch_size']}: {row['decisions_per_s']:.0f} decisions/s, "
                  f"{row['ms_per_searched_decision']:.2f} ms per searched decision")
        else:
            print(f"- {name} batch {row['batch_size']}: {row['games_per_min']:.0f} games/min, "
                  f"{row['searched_decisions_per_s']:.0f} searched decisions/s")
workers2 = json.loads((ROOT / "bench_workers2.json").read_text(encoding="utf-8"))
print(f"- two workers x batch 1024: {60 * workers2['summary'][0]['games'] / workers2['elapsed_s']:.0f} games/min over "
      f"{workers2['summary'][0]['games']} games ({workers2['elapsed_s']:.0f} s)")
for label, name in (("old implementation, 8 workers", "strength_seed7_budget64_old"), ("new implementation, 2 workers", "strength_seed7_budget64_new")):
    payload = json.loads((ROOT / f"{name}.json").read_text(encoding="utf-8"))
    print(f"- {label}: 1,000 budget-64 games in {payload['elapsed_s']:.0f} s")

print()
print("## Strength")
strength = json.loads((ROOT / "strength_seed0_budget0_64.json").read_text(encoding="utf-8"))
for row in strength["summary"]:
    print(f"- seed 0 new implementation budget {row['budget']}: {row['wins']}/{row['games']}")
seed0 = paired(games(ROOT / "strength_seed0_budget0_64.json", 64), games(PUBLISHED, 64))
seed7 = paired(games(ROOT / "strength_seed7_budget64_new.json", 64), games(ROOT / "strength_seed7_budget64_old.json", 64))
report("seed 0 (published deals)", *seed0)
report("seed 7 (fresh deals)", *seed7)
report("pooled", seed0[0] + seed7[0], seed0[1] + seed7[1], seed0[2] + seed7[2])
control_new = games(ROOT / "strength_seed0_budget0_64.json", 0)
control_old = games(PUBLISHED, 0)
print(f"- seed 0 budget-0 control identical on {sum(control_new[i] == control_old[i] for i in control_new)}/{len(control_new)} deals")

print()
print("## Candidate evaluation on fixed positions")
new = {row["position"]: row for row in json.loads((ROOT / "values_new.json").read_text(encoding="utf-8"))}
old = {row["position"]: row for row in json.loads((ROOT / "values_old.json").read_text(encoding="utf-8"))}
same = sum(
    [(c[0], c[1]) for c in new[p]["candidates"]] == [(c[0], c[1]) for c in old[p]["candidates"]]
    and new[p]["forced"] == old[p]["forced"]
    for p in new
)
gaps_new, gaps_old = [], []
for p in new:
    if len(new[p]["candidates"]) == 2:
        gaps_new.append(new[p]["candidates"][0][2] - new[p]["candidates"][1][2])
        gaps_old.append(old[p]["candidates"][0][2] - old[p]["candidates"][1][2])
differences = [a - b for a, b in zip(gaps_new, gaps_old)]
mean = statistics.mean(differences)
half = 1.959963984540054 * statistics.stdev(differences) / math.sqrt(len(differences))
print(f"- identical candidates and rollout allocations: {same}/{len(new)} positions")
print(f"- value gap correlation {statistics.correlation(gaps_new, gaps_old):.3f}; mean gap difference "
      f"{mean:+.4f} (95% CI {mean - half:+.4f} to {mean + half:+.4f}) over {len(differences)} positions")
