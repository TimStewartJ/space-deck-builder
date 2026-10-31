# Evidence for `training_throughput.md`

Logs, metrics and results behind
[`../training_throughput.md`](../training_throughput.md), copied from the run
folders on the project machine (Windows, RX 9070, torch
2.11.0+rocm7.13.0a20260408). Logs were converted to UTF-8. The run folders,
with every checkpoint, are archived there at
`D:\space-deck-builder-archive\training-throughput-20261030`. Every training
run used the reference recipe (`--preset reference-selfplay`, or
`continue-selfplay-seed0` resumed from the timing fixture). Per-update timings
are the gaps between consecutive `timestamp` fields in a metrics file, plus
each row's `rollout.duration_s` and `ppo_duration_s`.

## `timing/`: one lane, master against this branch

All three resumed the same fixture: update 200 of the phase 1
`continue-selfplay-seed0` seed-0 run with its ten-snapshot pool, built from
the archived run by this folder's `scripts/make_fixture.py`. Each trained 12
updates (201-212) through Task Scheduler pinned to logical CPUs 0-7
(`--updates 12 --lr-horizon 400 --self-play-horizon 400 --eval-every 1000`).
Only update 212 evaluates, and reported means cover updates 202-211, so the
evaluation size (600 games for `master`, 3,200 for this branch) does not enter
them.

| Files | Code | Wall / rollout / PPO per update |
|---|---|---|
| `master_2fb71e1*` | `master` (`2fb71e1`) | 9.64 / 6.26 / 3.18 s |
| `candidate_cd9d034*` | this branch at `cd9d034` | 2.17 / 1.23 / 0.87 s |
| `final_code_*` | this branch at `25ecf5d`, run twice in one task: default hipBLASLt, then `TORCH_BLAS_PREFER_HIPBLASLT=0` (rocBLAS) | 2.28 / 1.36 / 0.85 s and 2.13 / 1.33 / 0.73 s |

The commits after `cd9d034` keep PPO minibatches eager (graph capture gained
nothing) and share one warm-up stream per device (the memory fix); the final
code's PPO time varies between processes (see `microbench/`).

## `validation/`: a complete reference run, seed 0

`train_reference.ps1 -Seed 0` on this branch at `c96a9dc`, with production
placement (logical CPUs 0-15), in 25-update processes.

- `reference_manifest.json`, `chunks_manifest.json`: the run's manifests.
- `metrics/`: per-update metrics of updates 26-400. This folder's
  `scripts/compare_curves.py` compares them with the published seed 0's
  metrics in 100-update windows (the curve table in the write-up).
- `final_chunk_upd375_to_400.log`: includes the locked 15,000-game gauntlet
  (14,335 wins; Heuristic 4,397 of 5,000) on the vector engine.
- `league.json`: `python -m src league` (4,000 games per held-out opponent,
  six workers) for this run's final checkpoint and the published
  `selfplay_seed0`.
- `elo_temperature0.json`: `python -m src elo` at temperature 0, 4,000 games
  per pair, with this run's final checkpoint, the three published self-play
  seeds and Heuristic.

## `lanes/`: running several lanes

- `sweep/`: one, two, three, four and five lanes side by side at `25ecf5d`,
  each resuming the timing fixture with its own seed (10 + lane) for 30
  updates (`--lr-horizon 400 --self-play-horizon 400`), production placement.
  This folder's `scripts/multilane_report.ps1 -Label ml2 -LaneCounts 1,2,3,4,5`
  produced the lane table from them (means over updates 205-226). A six-lane
  run was stopped early and is not reported.
- `process_age/`: three lanes at `25ecf5d`, each training updates 201-600 in
  one process (horizons 600).
- `chunk_length/`: three lanes training 100 updates of the reference recipe
  from scratch through the repository's `scripts/run_queue.ps1`, in 25-update
  processes (`c25/`) and in one process (`c100/`); the queue lane logs record
  each arc's start and end. This folder's `scripts/chunk_ab.ps1` ran both
  arms.

## `microbench/`

Scripts named here are in this folder's `scripts/`.

- `merged_vs_separate_graphs.log` (`bench_padding.py`): one merged
  learner-and-snapshots replay against a learner replay plus a snapshot
  replay, per decision step, for four row counts.
- `gpu_step.log` (`bench_gpu_step.py`): an attempt to time pure GPU work by
  queueing replays behind a spin kernel. The host could not queue them before
  the spin ended (`queued in time: False`), so its numbers are not used.
- `blas_*.log` (`bench_blas.py`): the PPO update on synthetic data under
  hipBLASLt, rocBLAS, `allow_tf32`, and
  `torch.backends.cuda.preferred_blas_library`; `blas_alternating.log`
  alternates the two libraries over six processes.

## `scripts/`

The scratch scripts that produced these files. They name the run folders on
the project machine and are kept as a record of the method, not as tools.
