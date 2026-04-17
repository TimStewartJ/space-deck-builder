"""Print architecture metadata for one or more PPO checkpoints.

For each checkpoint, reports update number and the model config fields most
relevant to architecture decisions: pool_type, actor_type, trunk hidden sizes,
and card embedding dimension.

Usage:
    # Pass explicit paths
    python scripts/probe_checkpoints.py models/ppo_agent_0417_0245_upd100_wins200.pth

    # Or a glob (PowerShell expands globs by default; quote on bash to expand inside Python)
    python scripts/probe_checkpoints.py models/ppo_agent_*upd100*.pth

    # Sort by update number ascending
    python scripts/probe_checkpoints.py --sort-by-update models/ppo_agent_*.pth
"""
from __future__ import annotations

import argparse
import sys
from glob import glob
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="Checkpoint paths or glob patterns")
    parser.add_argument(
        "--sort-by-update",
        action="store_true",
        help="Sort checkpoints by their stored 'update' field ascending",
    )
    args = parser.parse_args()

    # Expand any glob patterns (so the script works the same on shells that
    # don't auto-expand, e.g. cmd.exe).
    paths: list[str] = []
    for raw in args.paths:
        expanded = glob(raw)
        paths.extend(expanded if expanded else [raw])

    if not paths:
        print("No checkpoint paths supplied.", file=sys.stderr)
        return 2

    rows: list[tuple[int | float, str, str]] = []
    for c in paths:
        p = Path(c)
        if not p.exists():
            rows.append((float("inf"), p.name, "MISSING"))
            continue
        try:
            d = torch.load(c, map_location="cpu", weights_only=False)
        except Exception as exc:  # pragma: no cover - diagnostic script
            rows.append((float("inf"), p.name, f"LOAD_ERROR: {exc}"))
            continue
        m = d.get("config", {}).get("model")
        upd = d.get("update", "?")
        if not m:
            rows.append(
                (upd if isinstance(upd, int) else float("inf"), p.name, f"upd={upd:<4} (legacy checkpoint, no model config)")
            )
            continue
        line = (
            f"upd={upd:<4} pool={m.get('pool_type', '?'):<10} "
            f"actor={m.get('actor_type', '?'):<5} "
            f"trunk={m.get('trunk_hidden_sizes', '?')} "
            f"emb={m.get('card_emb_dim', '?')}"
        )
        rows.append((upd if isinstance(upd, int) else float("inf"), p.name, line))

    if args.sort_by_update:
        rows.sort(key=lambda r: r[0])

    name_width = max(len(name) for _, name, _ in rows)
    for _, name, line in rows:
        print(f"{name:<{name_width}}  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
