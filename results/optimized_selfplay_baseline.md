# Optimized self-play baseline

**Status:** Protocol-compliant reference result
**Training commit:** `fe1d2a5`
**Protocol:** [`docs/eval_protocol.md`](../docs/eval_protocol.md)
**Evidence:** [`results/optimized_selfplay_baseline/`](optimized_selfplay_baseline/)

## Configuration

- three independent training seeds: 0, 1, 2;
- 400 updates × 2,000 episodes;
- Random, Heuristic, and Simple fixed opponents;
- constant 50% snapshot self-play;
- variance PFSP;
- three active snapshots per resident population and ten persistent snapshots;
- one fresh Python process every 25 updates;
- 3,200-game routine evaluations every 25 updates;
- final 15,000-game gauntlet per seed;
- Normal process priority and P-core affinity on the i7-12700K host.

Every process boundary restored optimizer, scheduler, snapshot-pool, Python,
NumPy, Torch CPU, and ROCm RNG state. Evaluation preserved those RNG streams.

## Fixed-opponent gauntlet

Each seed played 5,000 games against each fixed opponent. Confidence intervals
are Student-t intervals across the three independent training seeds.

| Metric | Optimized self-play | Fixed-opponent baseline | Paired delta |
|---|---:|---:|---:|
| Random | 99.79% ± 0.15pp | 99.79% ± 0.08pp | -0.01pp ± 0.20pp |
| Heuristic | 86.36% ± 2.03pp | 86.77% ± 0.45pp | -0.41pp ± 2.06pp |
| Simple | 97.06% ± 1.04pp | 97.55% ± 0.63pp | -0.49pp ± 0.45pp |
| Overall | 94.40% ± 0.74pp | 94.71% ± 0.10pp | -0.30pp ± 0.64pp |

The scripted gauntlet does not establish a self-play improvement. Random,
Heuristic, and overall paired intervals include zero; Simple shows a small
retention loss. Random and Simple are saturated and contribute little useful
strength separation.

Every per-opponent result above contains exactly 2,500 first-player and 2,500
second-player games per seed.

## Cross-play Elo

The tournament included the three fixed-baseline seeds, the three self-play
seeds, and all fixed agents. It used 1,000 games per pairing and anchored
`fixed_seed0` at Elo 1000.

| Rank | Participant | Elo |
|---:|---|---:|
| 1 | selfplay_seed0 | 1149 |
| 2 | selfplay_seed2 | 1130 |
| 3 | selfplay_seed1 | 1097 |
| 4 | fixed_seed1 | 1022 |
| 5 | fixed_seed2 | 1005 |
| 6 | fixed_seed0 | 1000 |
| 7 | heuristic | 725 |
| 8 | simple | 423 |
| 9 | random | 1 |

Across the nine fixed-versus-self-play checkpoint pairings, self-play won
67.9% of games. The self-play seeds occupy all top three Elo positions despite
the near-tie on scripted opponents. This confirms that cross-play is the useful
strength signal once the fixed gauntlet saturates.

Every Elo pairing contains 500 games in each seat. The published `elo.log`
contains all 36 pairings.

## Training performance

- warm rollout: 5.22s/update;
- warm PPO optimization: 2.97s/update;
- warm total: 8.18s/update;
- legacy routine sample efficiency: every seed first exceeded 70% against
  Heuristic at the 50-update evaluation. Those pre-fix routine evaluations
  randomized seats and are retained as training diagnostics, not as part of
  the locked exact-seat result;
- one complete seed, including evaluations and process resets: about one hour;
- three seeds plus Elo and analysis: about 3h 11m.

Seed 0 exposed the launcher's inherited scheduling problem during updates
1-22. The run resumed from validated checkpoints while priority and P-core
affinity were made explicit; RNG, optimizer, scheduler, and pool continuity
were preserved and no completed update was repeated. Seeds 1 and 2 used the
final placement for their complete runs. Timing aggregates include the seed-0
correction, while strength comparisons remain protocol-valid.

The 25-update process boundary remains intentional. A matched 30-update test of
torch 2.12/ROCm 7.15 took 468s versus 280s for the pinned production stack and
did not remove process-age degradation.

## Reference artifacts

- `summary.json` — per-seed values, aggregate intervals, paired deltas, timing,
  and Elo leaderboard;
- `manifest.json` — run topology and final artifact locations;
- `balanced_eval/` — authoritative exact-seat JSON and logs for all six
  gauntlets plus the complete Elo tournament;
- `seed_<N>_final_metrics.jsonl` — original updates 376-400 training metrics;
- `elo.log` — authoritative all-36-pair exact-seat tournament and leaderboard;
- `legacy_random_seat_elo.log` — preserved pre-fix tournament provenance;
- `checkpoints/selfplay_seed<N>.pth` — final model, optimizer, scheduler,
  portable pool manifest, and RNG state for each seed;
- `snapshots/seed_<N>/` — every historical policy required to resume each
  self-play seed without external files;
- `checkpoints/fixed_seed<N>.pth` — the three control checkpoints required to
  reproduce the cross-play tournament.

The optimized self-play baseline is the reference checkpoint league for future
strength work. The fixed-opponent baseline remains the control for scripted
opponent retention.
