# Scripts

Developer tools and orchestrators for training, profiling, inspecting, and
analyzing PPO runs. None of these are imported by the production code in
`src/`; they're standalone CLIs you invoke directly.

Run with the venv python so the local package and its deps resolve:

```powershell
.venv\Scripts\python.exe scripts\<name>.py [args]
```

PowerShell scripts run with `pwsh -File`:

```powershell
pwsh -File scripts\<name>.ps1
```

## Setup

| Script | Purpose |
|---|---|
| `setup_gpu.py` | Wrapper around `uv sync --extra <backend>` (rocm / cuda / cpu / detect) plus an install verification step. Because `torch` lives only in extras, a bare `uv sync` uninstalls it — always go through this script or pass `--extra` yourself. |

## Launching training

Launch long workloads outside the caller's process tree. A run started as a
child of the agent runtime measured ~3.7x slower than the same run started
through Task Scheduler, and `Start-Process` does not escape that context.

| Script | Purpose |
|---|---|
| `train_reference.ps1` | One command for a published training preset: `.\scripts\train_reference.ps1 -Seed 0`. Registers a Task Scheduler launch, trains the first chunk from scratch with the preset's full LR and self-play horizons, continues through `chunked_resume.ps1`, and finishes with the preset's locked exact-seat gauntlet. Preset values come from `TRAINING_PRESETS` in `src/config.py`; re-running with the same run root resumes, and mismatched parameters are rejected. Smoke-size overrides (`-Updates`, `-ChunkSize`, `-Episodes`, eval sizes, workers) are available. |
| `launch_training.ps1` | Launches `python -m src train` (`-TrainArgs`) or any other command (`-Command`, with `{python}` replaced by the resolved GPU interpreter) through Windows Task Scheduler at Normal priority, outside the caller's process tree. Each launch has a unique identity and owns a kill-on-close Windows Job Object so every descendant is cleaned up atomically. On the project's i7-12700K host, its Auto affinity uses logical CPUs 0-15 so training runs on P-cores. `-ProcessorAffinityMask None` explicitly restores all logical CPUs; explicit hexadecimal/decimal masks are also supported. |
| `stop_training.ps1` | Stops one exact launcher-owned task instance after verifying its launch manifest and action-process identity. Closing the launcher's Job Object terminates its full process tree without process-id ancestry races. Use this instead of `Stop-ScheduledTask`, which can orphan Python workers. |
| `training_process_placement.ps1` | Shared priority/affinity helper used by launch and chunk orchestration. Auto is deliberately hardware-safe: it applies `0xFFFF` only to the known 20-thread i7-12700K layout. |
| `training_task_lock.ps1` | Shared exclusive task-transaction lock used to serialize registration/startup against stopping the same scheduled task name. |

## Result artifacts

| Script | Purpose |
|---|---|
| `release_artifacts.py` | `publish` bundles a result's checkpoints and snapshot pools (default `**/*.pth`) into a deterministic zip with a `SHA256SUMS` manifest, refuses files that differ from an existing `ARTIFACTS.sha256`, and uploads the zip plus its checksum to the GitHub release for the result's tag. `fetch` downloads a bundle, verifies the archive and every entry, and extracts it safely. See `docs/eval_protocol.md` §8. |

## Inspection

| Script | Purpose |
|---|---|
| `probe_checkpoints.py` | Print pool/actor/trunk/emb metadata for one or more `.pth` checkpoints. Accepts paths or globs; `--sort-by-update` orders by training step. |
| `gen_arch_visual.py` | Generate a static HTML page documenting the current actor-critic architecture, the proposed attention head, and how attention maps to LLM patterns. |

## Analysis

| Script | Purpose |
|---|---|
| `compare_metrics.py` | Side-by-side diff of two or more `metrics_*.jsonl` files at key update milestones. Reports rollout win rate, entropy, losses, KL, clip fraction, grad norm, throughput. |
| `analyze_plateau.py` | Concatenates every `logs/metrics_*.jsonl` into one timeline, splits it into runs by timestamp gap, and prints LR / entropy / KL / clip / losses / explained variance / grad norm / win rate at representative updates. Use it to see whether a run has actually stopped improving. |

## Benchmarks

| Script | Purpose |
|---|---|
| `bench_rollout.py` | Runs a short training and reports rollout decisions/sec plus a full per-phase breakdown of the inference server and the workers. The first place to look when a rollout is slow, and the yardstick any rollout change should be measured against. |
| `bench_vec_primitives.py` | Measures the engine-only throughput ceiling of a possible struct-of-arrays game engine. Runs one deterministic fake workload through NumPy SIMD, Numba SPMD, and eager torch CPU/CUDA/ROCm backends; requires exact per-step masks and final state equality before timing, rejects branch-degenerate workloads, and interleaves repetitions. Torch results describe eager execution only, not a fused GPU compiler. It is a go/no-go gate for a rules-correct vector-engine spike, not a training benchmark. |
| `launch_vec_benchmark.ps1` | Launches `bench_vec_primitives.py` through Windows Task Scheduler at Normal priority, outside the caller's process tree. Use this for the intended 16,000-game gate; running the same workload as a child of the agent runtime can distort throughput by several times. |
| `bench_vector_engine.py` | Gate 2 benchmark using the compiled real-rules subset. Measures Numba engine + exact legal masks + current 505-float observations, then the same path with the actual PPO policy and production END_TURN suppression. |
| `launch_vector_engine_benchmark.ps1` | Launches `bench_vector_engine.py` through Task Scheduler at Normal priority and writes timestamped log/JSON evidence. |
| `benchmark.py` | End-to-end episode/throughput benchmark for the BatchRunner. Compares sequential vs batched modes. |
| `device_benchmark.py` | Lower-level per-device timing for the inference path. |
| `bench_token_path.py` | Training-shaped step benchmark (forward + backward + optimizer, real batch size) comparing `token_features=False` vs `True`. Written after a forward-only microbenchmark badly understated the tokenized path's real cost. |

## Profiling

| Script | Purpose |
|---|---|
| `profile_training.py` | Wraps a training invocation with py-spy and writes a profile to `analysis/`. |
| `analyze_pyspy.py` | Post-process a py-spy profile into hot-path summaries. |

## Sweep orchestrators

These April 2026 orchestrators predate the vector engine, presets, and the
Task Scheduler launcher. They drive the object engine with the flags of that
era and are kept for provenance of the architecture ablations; prefer
`train_reference.ps1` and `chunked_resume.ps1` for new work. They all share
the same shape: launch one detached training run per configuration, write a
manifest entry per run, and grace-kill the python process ~3 minutes after
`Training complete.` to make sure ROCm/CUDA resources are released before the
next launch.

| Script | Purpose |
|---|---|
| `overnight_sweep.ps1` | Cross-architecture sweep (sum_mlp / attn_attn / etc.) — used to build the original architecture comparison. |
| `overnight_master.ps1` | Manifest-driven multi-phase orchestrator (baseline wait → checkpoint safeguard → tournament → chained resumes). Idempotent: completed phases are skipped on restart. |
| `hp_sweep_attn.ps1` | Hyperparameter sweep for attention pool+actor (lr / epochs / clip-eps variants). |
| `hp3_full_resume.ps1` | Two-phase fresh-then-`--resume` run, written to validate `--resume` under real conditions. |
| `run_pool_comparison.ps1` | Quick A/B pool-type comparison (smaller scale than `overnight_sweep.ps1`). |
| `chunked_resume.ps1` | Runs a resumed training arc in fixed-size fresh-process chunks, with a durable manifest, fixed overall LR/curriculum horizons, restored RNG streams, RNG-neutral cheap interim evaluations, production process placement, and a full final evaluation. Launch it through Task Scheduler to avoid both inherited execution-context slowdown and long-process ROCm latency drift. |

Each orchestrator is **resumable**: it reads its own manifest JSON in
`logs/` and skips entries already marked `completed` so you can rerun safely
after a crash.

## Conventions

- All scripts that load checkpoints use `weights_only=False` because we
  store config dataclasses in the checkpoint, not just weights.
- Output for analysis scripts goes to stdout in human-readable form. Pipe
  to `Tee-Object` if you want both terminal output and a saved log.
- Long-running workloads must run outside the caller's process tree: launch
  them through `launch_training.ps1` (or `train_reference.ps1`) and stop them
  with `stop_training.ps1`.
