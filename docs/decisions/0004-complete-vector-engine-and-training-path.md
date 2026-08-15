# 0004 - Complete vector rules and expose a guarded training path

**Status:** Accepted on `feat/vector-engine-gate2` (2026-08-14), superseded in part by ADR 0005

## Context

ADR 0003 established a compiler-backed Numba engine for 26 cards. It failed
closed on target-selection, deferred-discard, destroy-base, discard-draw, and
OR effects, and was not reachable from the training CLI.

Stopping there left two separate systems: a fast experimental slice and the
object engine used for all real work.

## Decision

Extend the Effect IR and SoA state to match all current scalar semantics:

- pending SCRAP actions over hand, discard, and trade row;
- deferred opponent discard debt materialized at turn start;
- pending destroy-base without outpost priority;
- discard-then-draw and scrap-then-draw completion continuations;
- current scalar OR behavior: first child action, then parent action applies
  remaining children;
- ordered per-copy effect activation for duplicate cards.

Pending state is a four-entry queue per player. The differential oracle records
the scalar engine's realized draws and market refills, then applies the same
encoded action to Numba. All 49 canonical cards compile and have per-card
differential coverage.

Expose `RunConfig.engine` and `train --engine {object,vec}`. The vector runner:

- currently accepts only `--opponents random`;
- implements RandomAgent's exact action priority (play first, then direct
  attack, then uniform random);
- stores linked per-game transitions and computes GAE in Numba;
- uses the production PPO mask policy that suppresses END_TURN while play or
  direct attack is available;
- returns the same tensors as existing batch runners.

Unsupported vector-runner configurations (heuristic/simple/self-play) raise
explicit errors. The object engine remains the default.

## Verification

- all 49 cards compile;
- targeted tests cover trade scrap/refill, destroy-base queues, deferred
  discard, completion draws, and two-stage OR;
- four simultaneous full-card games match scalar masks, observations, and
  canonical state through complete games;
- complete CLI train -> PPO update -> scalar eval -> checkpoint smoke;
- vector rollout statistics are plausible against a matched object smoke;
- 117 vector-specific tests.
- 483 repository tests passing.

Detached full-card gate at G=16,000:

| path | median decisions/s | range |
| --- | ---: | ---: |
| full engine + mask + observation | **5.88M** | 5.44M-6.49M |
| actual PPO-integrated path | **587k** | 578k-602k |

The environment path is about 10.0x faster than the policy path, so full-card
vector rollout is policy-bound.

A consecutive two-update default-size smoke is memory-safe, but the warm
16,000-episode rollout is 29.2s and PPO is 42.6s. Gate throughput excludes
rollout storage/concatenation and optimizer work. The vector path is therefore
functional, not yet a default-size training speedup; reduced compute budgets
and streamed/bounded PPO storage are the next performance layer.

## Consequences

- The vector engine is usable end to end for random-opponent PPO work.
- Existing default training and reporting remain unchanged.
- Heuristic/simple/self-play support is the next integration scope, not hidden
  behind a fallback.
- PPO optimization remains the dominant whole-update cost after rollout is
  reduced.
