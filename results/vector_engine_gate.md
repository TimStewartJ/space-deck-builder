# Vector Engine Gate

**Date:** 2026-08-14  
**Branch:** `perf/vec-engine-spike`  
**Benchmark implementation:** `1337d21`

## Conclusion

**Gate 1 passes emphatically. Build the rules-correct vertical slice using
Numba SPMD on CPU, with 16 threads on the i7-12700K.**

The reviewed primitive workload executes the real array shapes and awkward
operations a struct-of-arrays engine needs, but intentionally implements no
Star Realms rules. It is a throughput ceiling and architecture gate, not an
end-to-end training prediction.

## Full gate

Command was launched through `scripts/launch_vec_benchmark.ps1`, outside the
agent process tree at Normal priority:

- 16,000 concurrent games
- 200 steps/game = 3.2M decisions/repetition
- 5 rotated/interleaved repetitions/backend
- exact NumPy/Numba/torch CPU/torch ROCm verification before timing

The workload maintained:

| branch | decisions | share |
| --- | ---: | ---: |
| play | 1,128,302 | 35.26% |
| draw-card play | 92,920 | 2.90% |
| buy | 514,209 | 16.07% |
| attack | 144,203 | 4.51% |
| end turn | 490,451 | 15.33% |
| pending mode | 922,835 | 28.84% |
| terminal resets | 6,233 | - |

Results:

| backend | median decisions/s | coefficient of variation | engine-only ceiling vs current |
| --- | ---: | ---: | ---: |
| Numba SPMD (20 threads) | **66.54M** | 3.1% | **700x** |
| NumPy masked SIMD | 1.714M | 1.6% | 18.0x |
| eager torch CPU | 1.149M | 2.0% | 12.1x (gate-ineligible) |
| eager torch ROCm | 0.910M | 2.2% | 9.6x (gate-ineligible) |

All backends produced the same full-state BLAKE2b digest. Verification compared
every pre-step `[games, 494]` action mask and every final mutable array exactly
through all modes and terminal resets.

An independent NumPy/Numba reproduction produced 70.72M / 1.465M decisions/s
with the same digest.

Raw data:

- [`full_gate.json`](vector_engine_gate/full_gate.json)
- [`reproduction.json`](vector_engine_gate/reproduction.json)

## Numba thread scaling

Fifteen rotated/interleaved repetitions per thread count:

| threads | median decisions/s | speedup | parallel efficiency |
| ---: | ---: | ---: | ---: |
| 1 | 13.10M | 1.00x | 100% |
| 2 | 25.85M | 1.97x | 98.7% |
| 4 | 51.01M | 3.90x | 97.4% |
| 8 | 75.93M | 5.80x | 72.5% |
| 12 | 89.00M | 6.80x | 56.6% |
| **16** | **95.98M** | **7.33x** | 45.8% |
| 20 | 73.21M | 5.59x | 28.0% |

Use 16 threads. Twenty is materially slower. Even one Numba thread exceeds
the 0.95M decisions/s 10x gate by roughly 14x, so the decision does not depend
on optimistic multicore scaling.

Raw data: [`numba_thread_scaling.json`](vector_engine_gate/numba_thread_scaling.json)

## Exposed bottlenecks

The primitive engine omitted materialization of the current 505-float policy
observation. A 16-thread Numba prototype measured:

- 22.51M observations/s
- 0.711ms for all 16,000 observations

Raw data: [`observation_ceiling.json`](vector_engine_gate/observation_ceiling.json)

The current 0.28M-parameter GPU policy, including forward, legal-mask
application, categorical sample, log-prob, and value, measured:

| batch | full decisions/s | latency |
| ---: | ---: | ---: |
| 256 | 0.149M | 1.72ms |
| 1,024 | 0.429M | 2.39ms |
| 4,096 | 0.897M | 4.57ms |
| 16,000 | **1.186M** | **13.49ms** |

Raw data: [`policy_ceiling.json`](vector_engine_gate/policy_ceiling.json)

Therefore engine + observation can be hidden comfortably under policy
inference. A current ~2.2M-decision rollout has a rough lower bound near two
seconds instead of ~22 seconds. PPO optimization still costs ~22 seconds per
update, so unchanged end-to-end training is closer to 24-25 seconds/update
(~1.8x), not hundreds of times faster.

## Gate 2 requirements

Build an Effect-to-IR compiler and Numba SPMD vertical slice covering the
compiler-validated no-target-selection subset plus real play, buy, attack, end
turn, draw, ally effects, legal masks, and observation generation.

Requirements:

1. Compare full state against the scalar engine after every decision.
2. Differential-test every supported card.
3. Sustain at least 5M engine+mask+observation decisions/s at G=16,000.
4. Demonstrate an integrated rollout that is policy-bound rather than
   environment-bound.

## Gate 2 result

Gate 2 completed on `feat/vector-engine-gate2`.

The compiler supports **26 canonical cards** and rejects 23 with explicit
target-selection/pending/OR reasons. The earlier "27 no-choice cards" estimate
was not a valid semantic category and is superseded by compiler validation.

Correctness:

- one differential case for every supported card;
- four simultaneous scalar/vector games over 250 decisions;
- exact legal masks and 505-float observations before every step;
- exact canonical state after every step;
- actual PPO policy smoke on CPU and ROCm;
- 434 repository tests passing.

Detached branch-diverse benchmark at G=16,000:

| path | median decisions/s | range |
| --- | ---: | ---: |
| engine + legal mask + observation | **7.31M** | 6.62M-7.49M |
| integrated actual GPU policy | **542k** | 538k-546k |

The 5M engine gate passes by 1.43x, and the environment is roughly 13x faster
than the integrated policy path. The rollout is policy-bound as required.

Raw data: [`gate2_full.json`](vector_engine_gate/gate2_full.json)

## Gate 3 result

Gate 3 extends the IR and state machine to all 49 canonical cards under current
scalar semantics:

- hand/discard/trade scrap targets;
- deferred forced discard;
- destroy-base targets;
- discard-draw and scrap-draw completion continuations;
- current two-stage OR resolution;
- ordered duplicate-card activation state.

The guarded `train --engine vec --opponents random` path collects linked PPO
rollouts, computes GAE, runs an optimizer update, evaluates through the scalar
reporting path, and saves a checkpoint. Other opponent types fail explicitly.

Final detached full-card benchmark:

| path | median decisions/s | range |
| --- | ---: | ---: |
| full engine + legal mask + observation | **5.88M** | 5.44M-6.49M |
| integrated actual GPU policy | **587k** | 578k-602k |

The environment is about 10.0x faster than policy inference. Raw data:
[`gate3_full.json`](vector_engine_gate/gate3_full.json).

### Default-size training reality

A detached two-update run at 16,000 episodes proves the guarded path survives
consecutive default-size updates without OOM after consumed rollout tensors
are deleted:

| update | rollout | PPO | samples |
| ---: | ---: | ---: | ---: |
| 1 (JIT/cold) | 80.6s | 129.9s | 2.76M |
| 2 (warm) | **29.2s** | **42.6s** | 2.73M |

This is functional but **not faster than the current object rollout at the
legacy 16,000-episode budget**. The engine itself is fast; millions of
observations/masks, variable-size policy calls, rollout storage, and PPO
optimization dominate and cause memory pressure. At 4,000 episodes, the warm
vector rollout is 9.6s and PPO is 9.2s.

Therefore the next performance work is not more simulation optimization. It
is the compute-budget pilot plus bounded/streamed rollout and PPO storage.

Raw log: [`two_update_smoke.log`](vector_engine_gate/two_update_smoke.log)

## Authoritative Rule IR / physical Core Set gate

ADR 0005 removes the scalar effect-type switch and implements the formerly
absent physical rules for Blob Carrier, Brain World, Mech World, Stealth
Needle, Fleet HQ, Central Office, and Freighter in both engines.

The detached 16,000-game benchmark includes free acquisition, copy selection,
ordered top-deck draws, next-ship-top tokens, and Fleet HQ passives. Five
repetitions produced:

| path | median decisions/s | range |
| --- | ---: | ---: |
| engine + legal mask + observation | **6.53M** | 6.35M-6.61M |
| integrated actual GPU policy | **588k** | 568k-592k |

The expanded engine remains above the 5M gate and about 11.1x faster than the
integrated policy path. Every repetition produced the same BLAKE2b digest over
all mutable SoA arrays, masks, and observations.

Raw artifacts:

- [`rule_ir_v1_gate.json`](vector_engine_gate/rule_ir_v1_gate.json)
- [`rule_ir_v1_gate.log`](vector_engine_gate/rule_ir_v1_gate.log)

Detached one-update ROCm CLI smokes used the same seed, 64 episodes, and random
opponent:

| engine | rollout | PPO | decisions |
| --- | ---: | ---: | ---: |
| vector | 5.6s | 2.3s | 11,393 |
| object | 10.1s | 2.1s | 10,829 |

Both paths completed evaluation and checkpoint serialization successfully;
the disposable smoke checkpoints were then removed.

- [`rule_ir_v1_vector_training_smoke.log`](vector_engine_gate/rule_ir_v1_vector_training_smoke.log)
- [`rule_ir_v1_object_training_smoke.log`](vector_engine_gate/rule_ir_v1_object_training_smoke.log)

## Fixed vector opponents

ADR 0006 ports Random, Heuristic, and Simple into one batched Numba selector.
Weighted opponent specifications assign one stable type per resident game and
rollout metrics retain independent wins/game totals for every type.

A detached 192-episode ROCm smoke completed vector rollout, PPO, three-way
scalar evaluation, and checkpoint serialization. Opponent selection took
0.06s of the 6.26s rollout loop, versus 5.95s in the GPU policy.

The same end-to-end gate at 2,000 episodes, the smallest compute-pilot batch,
completed without memory pressure:

| phase | wall |
| --- | ---: |
| rollout (288,040 policy samples) | 7.2s |
| PPO | 4.5s |
| three-way evaluation | 0.8s |
| total | 12.6s |

The 2,000 games were distributed as 694 Heuristic, 671 Random, and 635 Simple.
Fixed-opponent selection consumed 0.14s while GPU policy inference consumed
5.99s, so mixed opponents do not move the rollout bottleneck away from policy
inference.

Raw artifacts:

- [`fixed_opponents_training_smoke.log`](vector_engine_gate/fixed_opponents_training_smoke.log)
- [`fixed_opponents_training_smoke.jsonl`](vector_engine_gate/fixed_opponents_training_smoke.jsonl)
- [`fixed_opponents_2k_training_smoke.log`](vector_engine_gate/fixed_opponents_2k_training_smoke.log)
- [`fixed_opponents_2k_training_smoke.jsonl`](vector_engine_gate/fixed_opponents_2k_training_smoke.jsonl)

## Dense vector snapshot self-play

ADR 0008 ports snapshot/PFSP self-play directly onto the fast dense vector
rollout. Games keep stable fixed-or-snapshot assignments, snapshot policies
receive acting-player observations, and result names feed the existing EMA,
geometric pool, manifest, and resume machinery.

An uncapped 12-update, 2,000-episode probe grew the persistent pool to ten
snapshots. Snapshot inference increasingly fragmented:

| active snapshots | snapshot calls | avg snapshot batch | rollout |
| ---: | ---: | ---: | ---: |
| 1 | 454 | 46 | 2.3s |
| 6 | 3,194 | 47 | 8.9s |
| 9 | 4,291 | 33 | 11.2s |
| 10 | 4,371 | 32 | 10.4s |

The default now activates at most three PFSP-weighted snapshots per resident
population while retaining the complete log-spaced pool across updates. The
identical seed and 12-update protocol then measured:

| metric | uncapped | capped at 3 |
| --- | ---: | ---: |
| total run | 132.4s | **100.3s** |
| late-update rollout | 10.4s | **5.0s** |
| late snapshot calls | 4,371 | **1,476** |
| late average snapshot batch | 32 | **93** |

The cap does not discard snapshots. A resume smoke restored ten pool entries;
PFSP selected three (`PPO_7`, `PPO_8`, and `PPO_11`) for that resident
population, and update 13 completed through train, eval, checkpoint, and result
accounting.

Raw artifacts:

- [`dense_self_play_uncapped_probe.log`](vector_engine_gate/dense_self_play_uncapped_probe.log)
- [`dense_self_play_uncapped_probe.jsonl`](vector_engine_gate/dense_self_play_uncapped_probe.jsonl)
- [`dense_self_play_capped_probe.log`](vector_engine_gate/dense_self_play_capped_probe.log)
- [`dense_self_play_capped_probe.jsonl`](vector_engine_gate/dense_self_play_capped_probe.jsonl)
- [`dense_self_play_capped_resume.log`](vector_engine_gate/dense_self_play_capped_resume.log)
- [`dense_self_play_capped_resume.jsonl`](vector_engine_gate/dense_self_play_capped_resume.jsonl)
