"""Compare PPO training runs by diffing their metrics.jsonl files.

Loads two or more metrics files (one per training run) and prints a
side-by-side table at key update milestones for the most diagnostic PPO
metrics: rollout win rate, entropy, actor/critic loss, explained variance,
approx_kl, clip_fraction, and total grad norm.

Also reports total wall-time spent on rollouts vs PPO updates and the
final-update rollout win rate per run.

Usage:
    # Two runs with auto-derived labels (file stem)
    python scripts/compare_metrics.py logs/metrics_a.jsonl logs/metrics_b.jsonl

    # With explicit labels (label=path)
    python scripts/compare_metrics.py sum_mlp=logs/metrics_a.jsonl attn_attn=logs/metrics_b.jsonl

    # Custom milestone updates
    python scripts/compare_metrics.py --milestones 1,10,50,100 logs/a.jsonl logs/b.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_MILESTONES = [1, 5, 10, 25, 50, 75, 100]


def parse_run_arg(arg: str) -> tuple[str, Path]:
    """Accept either 'label=path' or just 'path' (label = file stem)."""
    if "=" in arg:
        label, _, path = arg.partition("=")
        return label, Path(path)
    p = Path(arg)
    return p.stem, p


def load_metrics(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="Run specs as 'label=path' or just 'path'")
    parser.add_argument(
        "--milestones",
        type=lambda s: [int(x) for x in s.split(",")],
        default=DEFAULT_MILESTONES,
        help="Comma-separated update numbers to compare at (default: 1,5,10,25,50,75,100)",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=20,
        help="Show win-rate trajectory for the last N updates (default: 20, 0 to disable)",
    )
    args = parser.parse_args()

    if len(args.runs) < 2:
        print("Need at least two runs to compare.", file=sys.stderr)
        return 2

    runs: dict[str, list[dict]] = {}
    for spec in args.runs:
        label, path = parse_run_arg(spec)
        if not path.exists():
            print(f"{label}: missing file {path}", file=sys.stderr)
            return 2
        runs[label] = load_metrics(path)
        print(f"{label}: {len(runs[label])} updates  ({path})")

    labels = list(runs.keys())
    col_w = 14

    metric_specs = [
        ("win_rate (vs pool)", lambda r: r["rollout"]["win_rate"]),
        ("entropy",            lambda r: r["ppo"]["entropy"]),
        ("actor_loss",         lambda r: r["ppo"]["actor_loss"]),
        ("critic_loss",        lambda r: r["ppo"]["critic_loss"]),
        ("explained_var",      lambda r: r["ppo"]["explained_variance"]),
        ("approx_kl",          lambda r: r["ppo"]["approx_kl"]),
        ("clip_fraction",      lambda r: r["ppo"]["clip_fraction"]),
        ("grad_norm",          lambda r: r["ppo"]["total_grad_norm"]),
    ]

    header = f"\n{'upd':>4} | {'metric':>22}"
    for label in labels:
        header += f" | {label:>{col_w}}"
    print(header)
    print("-" * len(header))

    for upd in args.milestones:
        rows_at_upd = {label: next((r for r in rs if r["update"] == upd), None) for label, rs in runs.items()}
        if not all(rows_at_upd.values()):
            continue
        for label_text, getter in metric_specs:
            line = f"{upd:>4} | {label_text:>22}"
            for label in labels:
                line += f" | {getter(rows_at_upd[label]):>{col_w}.4f}"
            print(line)
        print("-" * len(header))

    print("\n--- Throughput and wall ---")
    for label, rows in runs.items():
        total_ppo = sum(r["ppo_duration_s"] for r in rows)
        total_roll = sum(r["rollout"]["duration_s"] for r in rows)
        avg_thru = sum(r["rollout"]["throughput_eps"] for r in rows) / len(rows)
        print(
            f"{label}: ppo={total_ppo:.0f}s rollout={total_roll:.0f}s "
            f"avg_throughput={avg_thru:.0f}ep/s "
            f"final_win_rate={rows[-1]['rollout']['win_rate']:.3f}"
        )

    if args.tail > 0:
        print(f"\n--- Win rate trajectory (last {args.tail} updates) ---")
        head = f"{'upd':>4}"
        for label in labels:
            head += f" | {label:>10}"
        print(head)
        max_upd = max(rows[-1]["update"] for rows in runs.values())
        for upd in range(max(1, max_upd - args.tail + 1), max_upd + 1):
            line = f"{upd:>4}"
            ok = True
            for label in labels:
                row = next((r for r in runs[label] if r["update"] == upd), None)
                if row is None:
                    ok = False
                    break
                line += f" | {row['rollout']['win_rate']:>10.3f}"
            if ok:
                print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
