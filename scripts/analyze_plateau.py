"""Plateau diagnostic across all training metrics jsonl files.

Concatenates every ``logs/metrics_*.jsonl`` into a single timeline, dedupes by
update number, identifies distinct training runs by timestamp gaps, and prints
a compact table of LR, entropy, approximate KL, clip fraction, actor/critic
loss, explained variance, gradient norm, win rate, and self-play ratio at a
set of representative update checkpoints.

Run from the repo root::

    .venv\\Scripts\\python.exe scripts\\analyze_plateau.py
"""
import json, glob, os

files = sorted(glob.glob('logs/metrics_*.jsonl'), key=os.path.getmtime)
rows = []
for f in files:
    for line in open(f, encoding='utf-8'):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass

# Later-written entries for the same update win so resumed runs overwrite
# stale entries from prior chunks.
by_update = {}
for r in rows:
    by_update[r['update']] = r
rows = sorted(by_update.values(), key=lambda r: r['update'])

print(f'Total unique updates: {len(rows)}')
print(f'Update range: {rows[0]["update"]} -> {rows[-1]["update"]}')
print()

# A >30-minute gap between consecutive rows demarcates a separate run.
prev_ts = None
runs = []
current = []
for r in rows:
    ts = r['timestamp']
    if prev_ts and (ts[:10] != prev_ts[:10] or abs(int(ts[11:13]) * 60 + int(ts[14:16]) - int(prev_ts[11:13]) * 60 - int(prev_ts[14:16])) > 30):
        if current:
            runs.append(current)
        current = []
    current.append(r)
    prev_ts = ts
if current:
    runs.append(current)
print(f'Detected {len(runs)} distinct runs')
for run in runs:
    print(f'  updates {run[0]["update"]:4d} -> {run[-1]["update"]:4d}   ({run[0]["timestamp"][:19]})')
print()

print(f'{"upd":>5} {"lr":>10} {"entropy":>8} {"kl":>10} {"clip%":>7} {"actor":>10} {"critic":>8} {"EV":>7} {"grad":>7} {"win%":>6} {"sp":>5} {"emb_grad":>9}')
for u in [1, 10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 850, 875, 900, 950, 1000]:
    r = by_update.get(u)
    if r is None:
        continue
    p = r['ppo']
    print(f'{u:>5} {r["lr"]:>10.2e} {p["entropy"]:>8.3f} {p["approx_kl"]:>10.2e} {p["clip_fraction"] * 100:>6.2f}% {p["actor_loss"]:>+10.5f} {p["critic_loss"]:>8.4f} {p["explained_variance"]:>7.3f} {p["total_grad_norm"]:>7.3f} {r["rollout"]["win_rate"] * 100:>5.1f}% {r.get("self_play_ratio", 0):>5.2f} {r.get("card_emb_grad", 0):>9.2e}')
