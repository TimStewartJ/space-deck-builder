import re, sys
def parse_run(log_path, label):
    updates = []
    with open(log_path, 'r', encoding='utf-16', errors='ignore') as f:
        for line in f:
            m = re.search(r'PPO: actor=(-?[\d.]+)\s+critic=([\d.]+)\s+ent=([\d.]+)\s+kl=([\d.]+)\s+clip=([\d.]+)\s+gnorm=([\d.]+)\s+ev=(-?[\d.]+)', line)
            if m:
                updates.append({k: float(v) for k, v in zip(['actor','critic','ent','kl','clip','gnorm','ev'], m.groups())})
            wm = re.search(r'win_rate=(\d+)%', line)
            if wm and updates:
                updates[-1]['rollout_win'] = int(wm.group(1))
    evals = []
    with open(log_path, 'r', encoding='utf-16', errors='ignore') as f:
        for line in f:
            em = re.search(r'Overall: (\d+)/(\d+) wins \((\d+)%\)', line)
            if em:
                evals.append((int(em.group(1)), int(em.group(2)), int(em.group(3))))
    print(f'=== {label} ===  (updates parsed: {len(updates)})')
    if updates:
        u = updates[-1]
        print(f"  Final  ent={u['ent']:.2f}  ev={u['ev']:.2f}  critic={u['critic']:.3f}  actor={u['actor']:.4f}  rollout_win={u.get('rollout_win')}%")
    print(f'  Eval results: {evals}')
    print(f"  Rollout win trajectory: {[u.get('rollout_win') for u in updates]}")
    print(f"  Entropy trajectory:     {[round(u['ent'],2) for u in updates]}")
    print()

parse_run('logs/cmp_sum_20260416_222852.log', 'SUM')
parse_run('logs/cmp_attn_20260416_222852.log', 'ATTENTION')
