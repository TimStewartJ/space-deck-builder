# ROCm 7.15 / torch 2.12 stack probe

**Status:** Rejected for production
**Date:** 2026-08-18
**Evidence:** [`results/rocm715_stack_probe/`](rocm715_stack_probe/)

## Candidate

- Python 3.11;
- torch `2.12.0+rocm7.15.0a20260728`;
- ROCm `7.15.0a20260728`;
- device-specific `gfx1201` packages;
- `triton-windows 3.7.0.post26`.

The candidate lived in an isolated worktree and venv. GPU detection, gfx1201
kernel loading, existing checkpoint loading, TorchInductor, and all 548 tests
passed. The production environment was never modified.

## Interleaved eager A/B

Four old/new process pairs each ran identical cap-3 rollouts and PPO updates.
The workload showed substantial machine-level variance:

- paired rollout deltas: -5.40%, +19.98%, -0.52%, -4.24%;
- median warm rollout: 5.83s production, 5.69s candidate;
- median warm PPO: 2.88s production, 3.30s candidate.

This does not establish a reliable eager-mode improvement.

## TorchInductor

Windows Triton successfully compiled the real learner and stacked snapshot
models, but production-shaped variable batches repeatedly recompiled:

- first compiled rollout: 137.4s;
- warm compiled rollout: 5.82s;
- first compiled PPO update: 22.6s;
- warm compiled PPO update: 2.89s.

Compilation did not improve end-to-end warm performance and added unacceptable
fresh-process cost.

## Matched 30-update test

Both stacks resumed the same update-213 checkpoint with the same RNG,
curriculum, P-core placement, and 2,000-episode workload.

| Metric | Production 2.11/7.13 | Candidate 2.12/7.15 |
|---|---:|---:|
| Total wall | **280.4s** | 467.9s |
| Mean rollout | **5.91s** | 7.60s |
| Mean PPO | **3.33s** | 7.89s |
| Early warm rollout | **5.51s** | 7.06s |
| Late rollout | **5.84s** | 7.92s |

The candidate was 66.9% slower overall. It also retained process-age
degradation rather than fixing it.

## Decision

Production remains exact-pinned to torch 2.11 and ROCm 7.13. Future stack
upgrades must repeat the full compatibility, interleaved eager, compiler, and
30-update decay gates before changing that pin.
