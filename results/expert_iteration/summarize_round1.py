"""Summarize expert iteration round 1 against its pre-registered gates.

Usage: python results/expert_iteration/summarize_round1.py [evidence_dir]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.search.uplift import elo_from_score, wilson_interval  # noqa: E402

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "round1"
PRIMARY_SCORE = 0.529
RETENTION_HEURISTIC = 0.9186


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


selfplay = load("selfplay_summary.json")
print("## Self-play data")
print(f"- {selfplay['games']} games, {selfplay['positions']} positions, {selfplay['shards']} shards, "
      f"{selfplay['elapsed_s'] / 60:.1f} min")
identity = selfplay["identity"]
print(f"- budget {identity['budget']}, {identity['max_candidates']} candidates, seed {identity['seed']}, "
      f"search implementation {identity['search_implementation']}")

history = load("distill_history.json")
print()
print("## Distillation")
print(f"- {history['train_positions']} train and {history['validation_positions']} validation positions; "
      f"best epoch {history['best_epoch']}; {history['elapsed_s']:.0f} s")
print("| Epoch | Policy KL | Search agreement | Initial greedy agreement | KL from initial | Value MSE | Value EV |")
print("|---:|---:|---:|---:|---:|---:|---:|")
for entry in history["history"]:
    v = entry["validation"]
    ev = v["value_explained_variance"]
    print(f"| {entry['epoch']} | {v['policy_kl']:.4f} | {100 * v['search_agreement']:.2f}% | "
          f"{100 * v['initial_greedy_agreement']:.2f}% | {v['kl_initial']:.4f} | {v['value_mse']:.4f} | "
          f"{ev:.3f} |" if ev is not None else "| — |")

elo = load("gate_elo_T0.json")
pairing = elo["pairings"][0]
labels = {participant["index"]: participant["label"] for participant in elo["participants"]}
student_index = next(index for index, label in labels.items() if label.startswith("distilled"))
wins = pairing["wins_i"] if pairing["i"] == student_index else pairing["games"] - pairing["wins_i"]
games = pairing["games"]
low, high = wilson_interval(wins, games)
primary = wins / games >= PRIMARY_SCORE and low > 0.5
print()
print("## Primary gate: distilled vs source, both at temperature 0")
print(f"- {labels[pairing['i']]} vs {labels[pairing['j']]}: distilled won {wins}/{games} = {100 * wins / games:.1f}% "
      f"(Wilson 95% CI {100 * low:.1f}-{100 * high:.1f}%), Elo {elo_from_score(wins / games):+.0f} "
      f"({elo_from_score(low):+.0f} to {elo_from_score(high):+.0f})")
print(f"- threshold: score >= {100 * PRIMARY_SCORE:.1f}% and lower bound > 50% -> **{'PASS' if primary else 'FAIL'}**")

gauntlet = load("gate_gauntlet_T0.json")
per = {row["opponent"]: row for row in gauntlet["per_opponent"]}
heuristic = per["heuristic"]["win_rate"]
retention = heuristic >= RETENTION_HEURISTIC
print()
print("## Retention gate: locked gauntlet at temperature 0")
for name in ("random", "heuristic", "simple"):
    row = per[name]
    print(f"- vs {name}: {row['wins']}/{row['games']} = {100 * row['win_rate']:.2f}% "
          f"(first {row['first_player_games']}, second {row['second_player_games']})")
print(f"- threshold: Heuristic >= {100 * RETENTION_HEURISTIC:.2f}% -> **{'PASS' if retention else 'FAIL'}**")

print()
print(f"## Round 1: **{'PASS' if primary and retention else 'FAIL'}**")

path = ROOT / "diagnostic_search_eval.json"
if path.exists():
    diagnostic = json.loads(path.read_text(encoding="utf-8"))
    print()
    print("## Diagnostic: search on the distilled checkpoint")
    for row in diagnostic["summary"]:
        low, high = row["win_rate_ci95"]
        print(f"- budget {row['budget']}: {row['wins']}/{row['games']} = {100 * row['win_rate']:.1f}% "
              f"(95% CI {100 * low:.1f}-{100 * high:.1f}%), Elo {row['elo']:+.0f}")
