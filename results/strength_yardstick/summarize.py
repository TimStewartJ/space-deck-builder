"""Summarize decisions 0013 and 0014's measurements and apply their reading rules.

Reads, under one runs directory:

- ``scores/score_seed<s>.json``: the per-seed temperature-0 tournaments among
  the champion, selfplay_seed1, selfplay_seed2, exploiter_seed<s> and
  control_seed<s>;
- ``league/league.json``: the held-out league, and ``league/port_check.json``;
- ``audit/audit_<checkpoint>.json``: the purchase audits (the three published
  self-play seeds, and any optional ones);
- each arm's run folder, whose logs name its per-update metrics files;
- for decision 0014's exploiter from random weights,
  ``scores/ladder_score_seed<s>.json`` (the champion, selfplay_seed1,
  selfplay_seed2 and ladder_seed<s>) and ``league/league_ladder.json``.

Writes ``summary.json`` there and prints the tables the write-up uses. A
verdict is reported only when all three training seeds are scored and in the
league; with fewer it is marked provisional.

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

T_975 = {1: 12.706204736174698, 2: 4.302652729911275, 3: 3.182446305284263, 4: 2.776445105197799}
SEEDS = (0, 1, 2)
ARMS = {"exploiter": "exploit-selfplay-seed0", "control": "continue-selfplay-seed0"}
LADDER_PRESET = "exploit-ladder-random"
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
    """Mean and Student-t 95% interval; decision 0013 reads it with three seeds (df 2)."""
    mean = statistics.fmean(values)
    if len(values) < 2:
        return {"mean": mean, "low": None, "high": None, "n": len(values)}
    half = T_975[len(values) - 1] * statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "low": mean - half, "high": mean + half, "n": len(values)}


def league_scores(league: dict | None) -> dict[str, dict]:
    """Pooled league results of the checkpoints that finished every matchup.

    The league rewrites its JSON after each matchup, so a run still in
    progress leaves rows whose pooled rate covers only some opponents.
    """
    if not league:
        return {}
    return {
        row["label"]: row["pooled"]
        for row in league["checkpoints"]
        if row["pooled"]["games"] and row["pooled"].get("complete", True)
    }


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
    for path in sorted(glob.glob(os.path.join(runs, "audit", "audit_*.json"))):
        if path.endswith(".partial.json"):
            continue
        audit = load(path)
        if audit and "summary" in audit:
            label = os.path.splitext(os.path.basename(path))[0][len("audit_"):]
            audits[label] = {"counts": audit["counts"], "summary": audit["summary"]}

    return {
        "protocol": "docs/decisions/0013-strength-yardstick.md",
        "seeds": seeds,
        "verdict": verdict,
        "league": {
            label: pooled for label, pooled in sorted(leagues.items())
        },
        "league_by_opponent": {
            row["label"]: {
                name: {"wins": cell["wins"], "games": cell["games"]}
                for name, cell in row["opponents"].items()
            }
            for row in (league or {}).get("checkpoints", [])
        },
        "calibration": (league or {}).get("calibration"),
        "port_check": port,
        "audits": audits,
        "ladder": summarize_ladder(runs, (leagues.get("selfplay_seed0") or {}).get("win_rate")),
    }


def summarize_ladder(runs: str, champion_league: float | None) -> dict | None:
    """Decision 0014: the exploiter from random weights against the champion's ladder."""
    league = load(os.path.join(runs, "league", "league_ladder.json"))
    leagues = league_scores(league)
    seeds = {}
    for seed in SEEDS:
        score = load(os.path.join(runs, "scores", f"ladder_score_seed{seed}.json"))
        if score is None:
            continue
        ladder = f"ladder_seed{seed}@T0"
        row = {
            "W": win_rate(score, ladder, CHAMPION),
            "O": statistics.fmean(win_rate(score, ladder, peer) for peer in PEERS),
            "L": (leagues.get(f"ladder_seed{seed}") or {}).get("win_rate"),
            "ratings": {p["label"]: p["elo"] for p in score["participants"]},
        }
        curve = training_curve(os.path.join(runs, f"{LADDER_PRESET}_seed{seed}"))
        if curve:
            last = [rate for update, rate in curve.items() if update > max(curve) - 25]
            row["training_win_rate_vs_champion"] = {
                "update_1": curve.get(1),
                "last_25_updates": statistics.fmean(last),
                "by_update": {str(u): curve[u] for u in (1, 25, 50, 100, 200, 300, 400) if u in curve},
            }
        seeds[seed] = row
    if not seeds:
        return None

    values = lambda key: [row[key] for row in seeds.values() if row.get(key) is not None]
    wins = mean_interval(values("W"))
    complete = len(seeds) == len(SEEDS) and all(row["L"] is not None for row in seeds.values())
    exploitable = wins["low"] is not None and wins["mean"] >= 0.55 and wins["low"] > 0.50
    kind = None
    if exploitable and complete and champion_league is not None:
        kind = "narrow" if statistics.fmean(values("L")) <= champion_league else "general"
    return {
        "protocol": "docs/decisions/0014-ladder-exploiter.md",
        "seeds": seeds,
        "verdict": {
            "provisional": not complete,
            "seeds": sorted(seeds),
            "W": wins,
            "O": mean_interval(values("O")),
            "L": mean_interval(values("L")) if values("L") else None,
            "champion_league": champion_league,
            "exploitable": exploitable,
            "kind": kind,
        },
        "league": {label: pooled for label, pooled in sorted(leagues.items())},
    }


def _pct(value):
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _interval(stat: dict | None, signed: bool = False) -> str:
    """A mean in points with its 95% interval, or n/a."""
    if not stat:
        return "n/a"
    fmt = "{:+.1f}" if signed else "{:.1f}"
    text = fmt.format(100 * stat["mean"])
    if stat.get("low") is None:
        return f"{text} (one seed)"
    return f"{text} ({fmt.format(100 * stat['low'])} to {fmt.format(100 * stat['high'])})"


def _points(value, se=None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}" + (f" ({se:.2f})" if se is not None else "")


def print_report(summary: dict) -> None:
    verdict = summary["verdict"]
    if summary["seeds"]:
        print("## Exploiter versus control\n")
        print("| Seed | E | C | E - C | Exploiter vs control | O exploiter | O control | L exploiter | L control | Exploiter training win rate, update 1 / last 25 |")
        print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for seed, row in sorted(summary["seeds"].items()):
            curve = row.get("exploiter_training_win_rate") or {}
            print(
                f"| {seed} | {_pct(row['E'])} | {_pct(row['C'])} | {100 * row['E_minus_C']:+.1f} | "
                f"{_pct(row['exploiter_vs_control'])} | {_pct(row['O_exploiter'])} | {_pct(row['O_control'])} | "
                f"{_pct(row['L_exploiter'])} | {_pct(row['L_control'])} | "
                f"{_pct(curve.get('update_1'))} / {_pct(curve.get('last_25_updates'))} |"
            )
    if verdict:
        state = "provisional" if verdict["provisional"] else "final"
        print(f"\nVerdict ({state}, seeds {verdict['seeds']}); means in points with Student-t 95% intervals:\n")
        print(f"- E - C: {_interval(verdict['E_minus_C'], signed=True)}")
        print(f"- E: {_interval(verdict['E'])}; C: {_interval(verdict['C'])}")
        print(f"- O exploiter: {_interval(verdict['O_exploiter'])}; O control: {_interval(verdict['O_control'])}")
        print(f"- L exploiter: {_interval(verdict['L_exploiter'])}; L control: {_interval(verdict['L_control'])}")
        print(
            f"- exploitable: {verdict['exploitable']}; kind: {verdict['kind']}; "
            f"control_helps: {verdict['control_helps']}"
        )

    if summary["league"]:
        opponents = []
        for cells in summary["league_by_opponent"].values():
            opponents.extend(name for name in cells if name not in opponents)
        print("\n## Held-out league\n")
        print("| Checkpoint | " + " | ".join(opponents) + " | Pooled | 95% CI |")
        print("|---|" + "---:|" * len(opponents) + "---:|---|")
        for label, pooled in summary["league"].items():
            cells = summary["league_by_opponent"].get(label, {})
            rates = [
                _pct(cells[name]["wins"] / cells[name]["games"]) if name in cells else "n/a"
                for name in opponents
            ]
            low, high = pooled["win_rate_ci95"]
            print(
                f"| {label} | " + " | ".join(rates) +
                f" | {_pct(pooled['win_rate'])} | {100 * low:.1f}-{100 * high:.1f}% |"
            )
        calibration = summary.get("calibration") or {}
        if calibration:
            rates = [
                _pct(calibration[name]["win_rate"]) if name in calibration else "n/a"
                for name in opponents
            ]
            print("| *each opponent vs Heuristic* | " + " | ".join(rates) + " | | |")
        port = summary.get("port_check")
        if port is not None:
            print(f"\nPort check against the prototype: {'passed' if port['passed'] else 'FAILED'}")

    if summary["audits"]:
        print("\n## Purchase audit\n")
        print("| Checkpoint | Decisions | Override gain (SE) | Overrides | Clear mistakes | Their mean gain | Search gain (SE) | Search differs | Skipped |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for label, audit in summary["audits"].items():
            headline = audit["summary"]["all"]
            counts = audit["counts"]
            skipped = ", ".join(
                f"{key} {counts[key]}" for key in (
                    "no_eligible_decision", "diverged", "search_greedy_mismatch", "search_choice_mismatch",
                ) if counts.get(key)
            ) or "none"
            print(
                f"| {label} | {headline['decisions']} | "
                f"{_points(headline['override_gain_points'], headline['override_gain_se_points'])} | "
                f"{_pct(headline['override_used_share'])} | "
                f"{headline['clear_mistakes']} ({_pct(headline['clear_mistake_share'])}) | "
                f"{_points(headline['clear_mistake_mean_gain_points'])} | "
                f"{_points(headline['search_gain_points'], headline['search_gain_se_points'])} | "
                f"{_pct(headline['search_differs_share'])} | {skipped} |"
            )
        bins = list(next(iter(summary["audits"].values()))["summary"]["by_turn_counter"])
        print("\nOverride gain (SE) by turn counter, in points:\n")
        print("| Checkpoint | " + " | ".join(f"turns {name} (decisions)" for name in bins) + " |")
        print("|---|" + "---:|" * len(bins))
        for label, audit in summary["audits"].items():
            by_turn = audit["summary"]["by_turn_counter"]
            print(
                f"| {label} | " + " | ".join(
                    f"{_points(by_turn[name]['override_gain_points'], by_turn[name]['override_gain_se_points'])} "
                    f"({by_turn[name]['decisions']})"
                    for name in bins
                ) + " |"
            )

    ladder = summary.get("ladder")
    if ladder:
        print("\n## Exploiter from random weights (decision 0014)\n")
        print("| Seed | W: vs champion | O: vs seeds 1, 2 | L: league | Training win rate vs champion, update 1 / last 25 |")
        print("|---:|---:|---:|---:|---|")
        for seed, row in sorted(ladder["seeds"].items()):
            curve = row.get("training_win_rate_vs_champion") or {}
            print(
                f"| {seed} | {_pct(row['W'])} | {_pct(row['O'])} | {_pct(row['L'])} | "
                f"{_pct(curve.get('update_1'))} / {_pct(curve.get('last_25_updates'))} |"
            )
        verdict = ladder["verdict"]
        state = "provisional" if verdict["provisional"] else "final"
        print(f"\nVerdict ({state}, seeds {verdict['seeds']}); means in points with Student-t 95% intervals:\n")
        print(
            f"- W: {_interval(verdict['W'])}; O: {_interval(verdict['O'])}; "
            f"L: {_interval(verdict['L'])} (champion {_pct(verdict['champion_league'])})"
        )
        print(f"- exploitable: {verdict['exploitable']}; kind: {verdict['kind']}")


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
