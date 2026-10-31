"""Compare per-update training metrics of two reference-recipe runs."""
import json, sys
from pathlib import Path
import numpy as np

def load(root):
    rows = {}
    for path in sorted(Path(root).rglob("*metrics*.jsonl")):
        if "recovered" in path.name:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["update"]] = row
    return rows

def series(rows, getter):
    out = {}
    for update, row in rows.items():
        try:
            out[update] = float(getter(row))
        except (KeyError, TypeError, ValueError):
            pass
    return out

old, new = load(sys.argv[1]), load(sys.argv[2])
metrics = {
    "entropy": lambda r: r["ppo"]["entropy"],
    "approx_kl": lambda r: r["ppo"]["approx_kl"],
    "clip_fraction": lambda r: r["ppo"]["clip_fraction"],
    "explained_variance": lambda r: r["ppo"]["explained_variance"],
    "critic_loss": lambda r: r["ppo"]["critic_loss"],
    "decisions/ep": lambda r: r["rollout"]["num_samples"] / r["rollout"]["episodes"],
    "rollout win rate": lambda r: r["rollout"]["win_rate"],
    "win vs Heuristic": lambda r: r["rollout"]["per_opponent"]["Heuristic"]["wins"] / r["rollout"]["per_opponent"]["Heuristic"]["games"],
}
print(f"updates: old {len(old)} ({min(old)}..{max(old)}), new {len(new)} ({min(new) if new else '-'}..{max(new) if new else '-'})")
windows = [(1, 25), (26, 50), (51, 100), (101, 200), (201, 300), (301, 400)]
for name, getter in metrics.items():
    a, b = series(old, getter), series(new, getter)
    parts = []
    for lo, hi in windows:
        av = [a[u] for u in range(lo, hi + 1) if u in a]
        bv = [b[u] for u in range(lo, hi + 1) if u in b]
        if av and bv:
            parts.append(f"{lo}-{hi}: {np.mean(av):.4g} vs {np.mean(bv):.4g}")
    print(f"{name:20s} " + " | ".join(parts))
