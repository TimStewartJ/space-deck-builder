"""Rollout throughput benchmark.

The single validation loop for performance work. Runs a small number of
real training updates and reports the metric that matters -- decisions per
second during rollout -- alongside the server and worker phase breakdowns
that explain it.

The historical target is recorded below: the April 2026 runs sustained
~66,900 decisions/s at 16000 episodes / 20 workers. Any change to the
rollout architecture should be judged against that number, measured the
same way.

Usage:
    python scripts/bench_rollout.py                    # default config
    python scripts/bench_rollout.py --updates 3
    python scripts/bench_rollout.py --episodes 8000 --json out.json
"""
import argparse
import json
import pathlib
import re
import statistics
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent

# Two reference points, because they mean different things.
#
# HISTORICAL is what this workload achieved in April 2026
# (logs/curriculum_default_seed0_20260427_172015.log: "16000 episodes,
# 2167337 steps in 32.4s"). It is the aspirational number.
#
# BASELINE_TODAY is the same April commit (9cd76b8) re-run on this machine
# during the August performance work: ~100.2 s for the same 16000 episodes,
# i.e. ~22,200 decisions/s. The identical code is ~3x slower than it was,
# which is a machine-level regression no code change can recover. Judge
# rollout changes against BASELINE_TODAY; treat HISTORICAL as the target to
# chase only once the machine regression is understood.
HISTORICAL_STEPS_PER_SEC = 66_900
HISTORICAL_ROLLOUT_S = 32.4
BASELINE_TODAY_STEPS_PER_SEC = 22_200
BASELINE_TODAY_ROLLOUT_S = 100.2

ROLLOUT_RE = re.compile(
    r"Rollout: (\d+) episodes, (\d+) steps in ([\d.]+)s \(([\d.]+) ep/s\)"
)
SERVER_RE = re.compile(
    r"\[InferenceServer\] batches=(\d+) avg_bs=([\d.]+) max_bs=(\d+) "
    r"compute=([\d.]+)s wall=([\d.]+)s"
)
PHASES_RE = re.compile(
    r"\[InferenceServer\] phases: idle=([\d.]+)s drain=([\d.]+)s "
    r"stage=([\d.]+)s h2d=([\d.]+)s kernel=([\d.]+)s sync=([\d.]+)s "
    r"dispatch=([\d.]+)s alloc=([\d.]+)s \| accounted=(\d+)% reqs=(\d+) "
    r"stage=([\d.]+)MB@([\d.]+)MB/s"
)
WORKERS_RE = re.compile(
    r"\[Workers\] avg/worker: advance=([\d.]+)s finish=([\d.]+)s "
    r"encode=([\d.]+)s send=([\d.]+)s wait=([\d.]+)s apply=([\d.]+)s \| "
    r"total=([\d.]+)s busy=(\d+)% blocked=(\d+)%"
)
PPO_RE = re.compile(r"PPO: .*?([\d.]+)s  samples=(\d+)")


def run(args) -> str:
    cmd = [
        str(REPO / ".venv" / "Scripts" / "python.exe")
        if (REPO / ".venv" / "Scripts" / "python.exe").exists()
        else sys.executable,
        "-u", "-m", "src", "train",
        "--updates", str(args.updates),
        "--episodes", str(args.episodes),
        "--eval-every", "9999",
        "--seed", "0",
    ]
    if args.workers is not None:
        cmd += ["--num-workers", str(args.workers)]
    if args.extra:
        cmd += args.extra.split()

    print(f"$ {' '.join(cmd)}\n", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, cwd=str(REPO), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:])
        raise SystemExit(f"training failed with exit code {proc.returncode}")
    print(f"(subprocess wall: {wall:.1f}s)\n")
    return proc.stdout


def parse(out: str, skip_first: bool) -> dict:
    rollouts = [
        {"episodes": int(m[0]), "steps": int(m[1]),
         "seconds": float(m[2]), "eps_per_s": float(m[3])}
        for m in ROLLOUT_RE.findall(out)
    ]
    servers = [
        {"batches": int(m[0]), "avg_bs": float(m[1]), "max_bs": int(m[2]),
         "compute_s": float(m[3]), "wall_s": float(m[4])}
        for m in SERVER_RE.findall(out)
    ]
    phases = [
        {"idle": float(m[0]), "drain": float(m[1]), "stage": float(m[2]),
         "h2d": float(m[3]), "kernel": float(m[4]), "sync": float(m[5]),
         "dispatch": float(m[6]), "alloc": float(m[7]),
         "accounted_pct": int(m[8]), "requests": int(m[9]),
         "stage_mb": float(m[10]), "stage_mbps": float(m[11])}
        for m in PHASES_RE.findall(out)
    ]
    workers = [
        {"advance": float(m[0]), "finish": float(m[1]), "encode": float(m[2]),
         "send": float(m[3]), "wait": float(m[4]), "apply": float(m[5]),
         "total": float(m[6]), "busy_pct": int(m[7]), "blocked_pct": int(m[8])}
        for m in WORKERS_RE.findall(out)
    ]
    ppo = [float(m[0]) for m in PPO_RE.findall(out)]

    # The first update pays GPU/ROCm warmup and process spawn, so it is not
    # representative of steady state.
    if skip_first:
        rollouts, servers, phases, workers, ppo = (
            rollouts[1:], servers[1:], phases[1:], workers[1:], ppo[1:],
        )
    return {"rollouts": rollouts, "servers": servers, "phases": phases,
            "workers": workers, "ppo_s": ppo}


def mean(xs):
    return statistics.mean(xs) if xs else 0.0


def report(d: dict) -> dict:
    rollouts = d["rollouts"]
    if not rollouts:
        raise SystemExit("no rollout lines parsed — did training actually run?")

    steps = sum(r["steps"] for r in rollouts)
    secs = sum(r["seconds"] for r in rollouts)
    sps = steps / secs if secs else 0.0

    print("=" * 68)
    print("ROLLOUT THROUGHPUT")
    print("=" * 68)
    print(f"  updates measured      : {len(rollouts)}")
    print(f"  mean rollout time     : {mean([r['seconds'] for r in rollouts]):8.1f} s")
    print(f"  decisions/sec         : {sps:8,.0f}")
    print()
    same_machine = sps / BASELINE_TODAY_STEPS_PER_SEC
    print(f"  vs April code, today  : {same_machine:7.2f}x "
          f"({BASELINE_TODAY_STEPS_PER_SEC:,} dec/s)   <-- like-for-like")
    print(f"  vs April, in April    : {sps / HISTORICAL_STEPS_PER_SEC * 100:7.1f}% "
          f"({HISTORICAL_STEPS_PER_SEC:,} dec/s)  <-- machine has since regressed ~3x")

    if d["servers"]:
        s = d["servers"]
        p = d["phases"]
        wall = mean([x["wall_s"] for x in s])
        print("\n" + "-" * 68)
        print("INFERENCE SERVER (mean per update)")
        print("-" * 68)
        print(f"  wall                  : {wall:8.1f} s")
        print(f"  batches               : {mean([x['batches'] for x in s]):8,.0f}"
              f"   avg_bs {mean([x['avg_bs'] for x in s]):,.0f}"
              f"   max_bs {max(x['max_bs'] for x in s):,}")
        if p:
            for k, label in (
                ("idle", "idle (starved)"), ("drain", "drain"),
                ("stage", "stage"), ("h2d", "h2d"), ("kernel", "kernel"),
                ("sync", "sync"), ("dispatch", "dispatch"), ("alloc", "alloc"),
            ):
                v = mean([x[k] for x in p])
                print(f"    {label:<20}: {v:8.1f} s  ({v / wall * 100:5.1f}%)")
            print(f"  stage throughput      : {mean([x['stage_mbps'] for x in p]):8,.0f} MB/s"
                  f"  ({mean([x['stage_mb'] for x in p]):,.0f} MB)")
            print(f"  requests/batch        : "
                  f"{mean([x['requests'] for x in p]) / mean([x['batches'] for x in s]):8.1f}")
            print(f"  accounted             : {mean([x['accounted_pct'] for x in p]):7.0f}%")

    if d["workers"]:
        w = d["workers"]
        tot = mean([x["total"] for x in w])
        print("\n" + "-" * 68)
        print("WORKERS (mean per worker per update)")
        print("-" * 68)
        for k in ("advance", "finish", "encode", "send", "wait", "apply"):
            v = mean([x[k] for x in w])
            print(f"    {k:<20}: {v:8.1f} s  ({v / tot * 100:5.1f}%)")
        print(f"  busy                  : {mean([x['busy_pct'] for x in w]):7.0f}%")
        print(f"  blocked on server     : {mean([x['blocked_pct'] for x in w]):7.0f}%"
              "   <-- lost throughput")

    if d["ppo_s"]:
        print(f"\n  PPO update (mean)     : {mean(d['ppo_s']):8.1f} s")

    print("=" * 68)
    return {
        "decisions_per_sec": sps,
        "pct_of_historical": sps / HISTORICAL_STEPS_PER_SEC * 100,
        "x_vs_april_code_today": sps / BASELINE_TODAY_STEPS_PER_SEC,
        "mean_rollout_s": mean([r["seconds"] for r in rollouts]),
        "mean_ppo_s": mean(d["ppo_s"]),
        "beats_april_code_today": sps >= BASELINE_TODAY_STEPS_PER_SEC,
        "matches_historical": sps >= HISTORICAL_STEPS_PER_SEC,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--updates", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=16000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--extra", type=str, default="",
                    help="extra flags passed through to `src train`")
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--keep-first", action="store_true",
                    help="include update 1 (warmup) in the averages")
    ap.add_argument("--label", type=str, default="")
    args = ap.parse_args()

    out = run(args)
    d = parse(out, skip_first=not args.keep_first and args.updates > 1)
    summary = report(d)
    summary["label"] = args.label
    summary["episodes"] = args.episodes
    summary["updates"] = args.updates

    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps({"summary": summary, "detail": d}, indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
