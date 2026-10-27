"""Summarize decision 0013's measurements and apply its reading rule.

Reads, under one runs directory:

- ``scores/score_seed<s>.json``: the per-seed temperature-0 tournaments among
  the champion, selfplay_seed1, selfplay_seed2, exploiter_seed<s> and
  control_seed<s>;
- ``league/league.json``: the held-out league, and ``league/port_check.json``;
- ``audit/audit_selfplay_seed<s>.json``: the purchase audits;
- each arm's run folder, whose logs name its per-update metrics files.

Writes ``summary.json`` there and prints the tables the write-up uses. The
verdict is reported only when all three training seeds are scored; with fewer
it is marked provisional.

Usage: python results/strength_yardstick/summarize.py [<runs dir>]
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import statistics
import sys

T_975_DF2 = 4.302652729911275
SEEDS = (0, 1, 2)
ARMS = {"exploiter": "exploit-selfplay-seed0", "control": "continue-selfplay-seed0"}
CHAMPION = "selfplay_seed0@T0"
PEERS = ("selfplay_seed1@T0", "selfplay_seed2@T0")


def load(path: str):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8-sig") as fh:
        return json.load(fh)


def win_rate(score: dict, player: str, opponent: str) -> float | None:
    """Win rate of *player* against *opponent* in an elo JSON, by label."""
    for pairing in score["pairings"]:
        if pairing["label_i"] == player and pairing["label_j"] == opponent:
            return pairing["wins_i"] / pairing["games"]
        if pairing["label_i"] == opponent and pairing["label_j"] == player:
            return 1.0 - pairing["wins_i"] / pairing["games"]
    return None


def mean_interval(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return {"mean": mean, "low": None, "high": None, "n": len(values)}
    half = (T_975_DF2 if len(values) == 3 else math.nan) * statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "low": mean - half, "high": mean + half, "n": len(values)}


def league_scores(league: dict | None) -> dict[str, dict]:
    if not league:
        return {}
    return {row["label"]: row["pooled"] for row in league["checkpoints"] if row["pooled"]["games"]}


def read_text(path: str) -> str:
    """Read a log written by either Windows PowerShell 5.1 (UTF-16) or Python (UTF-8)."""
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8-sig", errors="replace")


def training_curve(run_root: str, frozen: str = "frozen:selfplay_seed0") -> dict | None:
    """Per-update rollout win rate against the frozen champion, from a run's metrics.

    Chunk logs are read in name order, which is chronological because their
    names end in a timestamp, so a rerun chunk's rows replace those of the
    attempt it repeats.
    """
    logs = sorted(glob.glob(os.path.join(run_root, "**", "*.log"), recursive=True), key=os.path.basename)
    metrics_files = []
    for log in logs:
        for line in read_text(log).splitlines():
            match = re.search(r"Metrics file:\s+(\S+\.jsonl)", line)
            if match:
                metrics_files.append(match.group(1))
    repository = None
    manifest = load(os.path.join(run_root, "reference_manifest.json"))
    if manifest:
        repository = manifest.get("repository")
    rows = {}
    for name in metrics_files:
        path = name if os.path.isabs(name) or repository is None else os.path.join(repository, name)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    rows[int(row["update"])] = row
    curve = {}
    for update, row in sorted(rows.items()):
        result = row.get("rollout", {}).get("per_opponent", {}).get(frozen)
        if result and result["games"]:
            curve[update] = result["wins"] / result["games"]
    return curve or None


def summarize(runs: str) -> dict:
    league = load(os.path.join(runs, "league", "league.json"))
    port = load(os.path.join(runs, "league", "port_check.json"))
    leagues = league_scores(league)
    seeds = {}
    for seed in SEEDS:
        score = load(os.path.join(runs, "scores", f"score_seed{seed}.json"))
        if score is None:
            continue
        exploiter, control = f"exploiter_seed{seed}@T0", f"control_seed{seed}@T0"
        row = {
            "E": win_rate(score, exploiter, CHAMPION),
            "C": win_rate(score, control, CHAMPION),
            "exploiter_vs_control": win_rate(score, exploiter, control),
            "O_exploiter": statistics.fmean(win_rate(score, exploiter, peer) for peer in PEERS),
            "O_control": statistics.fmean(win_rate(score, control, peer) for peer in PEERS),
            "L_exploiter": (leagues.get(f"exploiter_seed{seed}") or {}).get("win_rate"),
            "L_control": (leagues.get(f"control_seed{seed}") or {}).get("win_rate"),
            "ratings": {p["label"]: p["elo"] for p in score["participants"]},
        }
        row["E_minus_C"] = row["E"] - row["C"]
        curve = training_curve(os.path.join(runs, f"{ARMS['exploiter']}_seed{seed}"))
        if curve:
            last = [rate for update, rate in curve.items() if update > max(curve) - 25]
            row["exploiter_training_win_rate"] = {
                "update_1": curve.get(1),
                "last_25_updates": statistics.fmean(last),
                "by_update": {str(u): curve[u] for u in (1, 25, 50, 100, 200, 300, 400) if u in curve},
            }
        seeds[seed] = row

    verdict = None
    if seeds:
        values = lambda key: [row[key] for row in seeds.values() if row.get(key) is not None]
        difference = mean_interval(values("E_minus_C"))
        control = mean_interval(values("C"))
        complete = len(seeds) == len(SEEDS) and all(
            row["L_exploiter"] is not None and row["L_control"] is not None for row in seeds.values()
        )
        exploitable = (
            difference["low"] is not None and difference["mean"] >= 0.05 and difference["low"] > 0
        )
        narrow = None
        if complete:
            narrow = (
                statistics.fmean(values("L_exploiter")) <= statistics.fmean(values("L_control"))
                and statistics.fmean(values("O_exploiter")) <= statistics.fmean(values("O_control"))
            )
        verdict = {
            "provisional": not complete,
            "seeds": sorted(seeds),
            "E_minus_C": difference,
            "E": mean_interval(values("E")),
            "C": control,
            "O_exploiter": mean_interval(values("O_exploiter")),
            "O_control": mean_interval(values("O_control")),
            "L_exploiter": mean_interval(values("L_exploiter")) if values("L_exploiter") else None,
            "L_control": mean_interval(values("L_control")) if values("L_control") else None,
            "exploitable": exploitable,
            "kind": (None if not exploitable else ("narrow" if narrow else "general" if narrow is not None else None)),
            "control_helps": control["low"] is not None and control["mean"] >= 0.55 and control["low"] > 0.50,
        }

    audits = {}
    for seed in SEEDS:
        audit = load(os.path.join(runs, "audit", f"audit_selfplay_seed{seed}.json"))
        if audit:
            audits[f"selfplay_seed{seed}"] = {"counts": audit["counts"], "summary": audit["summary"]}

    return {
        "protocol": "docs/decisions/0013-strength-yardstick.md",
        "seeds": seeds,
        "verdict": verdict,
        "league": {
            label: pooled for label, pooled in sorted(leagues.items())
        },
        "league_by_opponent": {
            row["label"]: {name: cell["wins"] for name, cell in row["opponents"].items()}
            for row in (league or {}).get("checkpoints", [])
        },
        "calibration": (league or {}).get("calibration"),
        "port_check": port,
        "audits": audits,
    }


def _pct(value):
    return "n/a" if value is None else f"{100 * value:.1f}%"


def print_report(summary: dict) -> None:
    verdict = summary["verdict"]
    if summary["seeds"]:
        print("| Seed | E | C | E - C | Exploiter vs control | O exploiter | O control | L exploiter | L control |")
        print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for seed, row in sorted(summary["seeds"].items()):
            print(
                f"| {seed} | {_pct(row['E'])} | {_pct(row['C'])} | {100 * row['E_minus_C']:+.1f} | "
                f"{_pct(row['exploiter_vs_control'])} | {_pct(row['O_exploiter'])} | {_pct(row['O_control'])} | "
                f"{_pct(row['L_exploiter'])} | {_pct(row['L_control'])} |"
            )
    if verdict:
        d = verdict["E_minus_C"]
        interval = (
            f"{100 * d['low']:+.1f} to {100 * d['high']:+.1f}" if d["low"] is not None else "n/a"
        )
        print(
            f"\nverdict ({'provisional' if verdict['provisional'] else 'final'}, seeds {verdict['seeds']}): "
            f"E - C mean {100 * d['mean']:+.1f} points (95% CI {interval}); exploitable={verdict['exploitable']} "
            f"kind={verdict['kind']} control_helps={verdict['control_helps']}"
        )
    if summary["league"]:
        print("\n| Checkpoint | League wins | Win rate | 95% CI |")
        print("|---|---:|---:|---|")
        for label, pooled in summary["league"].items():
            low, high = pooled["win_rate_ci95"]
            print(f"| {label} | {pooled['wins']}/{pooled['games']} | {_pct(pooled['win_rate'])} | {100 * low:.1f}-{100 * high:.1f}% |")
    for label, audit in summary["audits"].items():
        headline = audit["summary"]["all"]
        print(
            f"\n{label}: override {headline['override_gain_points']:+.2f} points "
            f"(SE {headline['override_gain_se_points']:.2f}), clear mistakes "
            f"{headline['clear_mistakes']}/{headline['decisions']}, search "
            f"{headline['search_gain_points']:+.2f} (SE {headline['search_gain_se_points']:.2f})"
        )


def main(argv: list[str]) -> int:
    runs = argv[1] if len(argv) > 1 else r"D:\sdb-yardstick-runs"
    summary = summarize(runs)
    with open(os.path.join(runs, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    print_report(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
