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
| `setup_gpu.py` | Install the right PyTorch wheel for your hardware (rocm / cuda / cpu / detect). Run after every `uv sync` because uv prunes torch when extras change. |

## Inspection

| Script | Purpose |
|---|---|
| `probe_checkpoints.py` | Print pool/actor/trunk/emb metadata for one or more `.pth` checkpoints. Accepts paths or globs; `--sort-by-update` orders by training step. |
| `gen_arch_visual.py` | Generate a static HTML page documenting the current actor-critic architecture, the proposed attention head, and how attention maps to LLM patterns. |

## Analysis

| Script | Purpose |
|---|---|
| `compare_metrics.py` | Side-by-side diff of two or more `metrics_*.jsonl` files at key update milestones. Reports rollout win rate, entropy, losses, KL, clip fraction, grad norm, throughput. |

## Benchmarks

| Script | Purpose |
|---|---|
| `benchmark.py` | End-to-end episode/throughput benchmark for the BatchRunner. Compares sequential vs batched modes. |
| `device_benchmark.py` | Lower-level per-device timing for the inference path. |

## Profiling

| Script | Purpose |
|---|---|
| `profile_training.py` | Wraps a training invocation with py-spy and writes a profile to `analysis/`. |
| `analyze_pyspy.py` | Post-process a py-spy profile into hot-path summaries. |

## Sweep orchestrators

These all share the same shape: launch one detached training run per
configuration, write a manifest entry per run, and grace-kill the python
process ~3 minutes after `Training complete.` to make sure ROCm/CUDA
resources are released before the next launch.

| Script | Purpose |
|---|---|
| `overnight_sweep.ps1` | Cross-architecture sweep (sum_mlp / attn_attn / etc.) — used to build the original architecture comparison. |
| `hp_sweep_attn.ps1` | Hyperparameter sweep for attention pool+actor (lr / epochs / clip-eps variants). |
| `run_pool_comparison.ps1` | Quick A/B pool-type comparison (smaller scale than `overnight_sweep.ps1`). |

Each orchestrator is **resumable**: it reads its own manifest JSON in
`logs/` and skips entries already marked `completed` so you can rerun safely
after a crash.

## Conventions

- All scripts that load checkpoints use `weights_only=False` because we
  store config dataclasses in the checkpoint, not just weights.
- Output for analysis scripts goes to stdout in human-readable form. Pipe
  to `Tee-Object` if you want both terminal output and a saved log.
- Long-running orchestrators must be launched with `Start-Process` in a
  hidden window so they survive shell exits — see the **Launching Detached
  Training Runs** section in agents.md.
