# 0006 - Execute every fixed opponent in the vector runner

**Status:** Accepted on `feat/vector-opponents` (2026-08-15)

## Context

The vector runner originally accepted only `random`. That was sufficient to
prove the engine and PPO integration, but not for useful compute-budget work:
Random is saturated, while Heuristic is the discriminating fixed opponent.
Falling back to object simulation for selected games would erase the vector
architecture's throughput and complicate result accounting.

The scalar fixed pool contains three policies:

- Random: play first, then direct attack, then sample uniformly from physical
  `Action` objects;
- Heuristic: strategic play/effect/attack/buy priorities plus phase-aware
  scrap and discard choices;
- Simple: take the first available physical action.

## Decision

Compile the scalar heuristic's static card scores once, then execute all three
fixed policies in a Numba batch kernel beside the vector game state.

Each resident game receives one opponent kind sampled from the normalized
`OpponentPool` weights. The assignment remains fixed for that game and results
are reported independently as `Random`, `Heuristic`, and `Simple`.

The selector consumes canonical legal masks and physical-action
multiplicities. Random preserves its exact weighted distribution. Heuristic
uses the same play, effect, lethal, base-defense, faction-synergy, buy, scrap,
and discard scoring as the scalar agent. Simple preserves scalar action-group
order.

The object engine stores stable list order for hands, played cards, and bases;
the SoA engine intentionally stores counts. When otherwise equal choices
depend only on that hidden order, vector policies use canonical card order.
Those choices have equal scalar heuristic score. This is an explicit
tie-breaking contract rather than an object-engine fallback.

Snapshot self-play is not a fixed opponent type. It remains unsupported by the
vector runner and continues to fail explicitly.

## Verification

- direct scalar/vector policy fixtures cover every action group and heuristic
  priority;
- weighted assignment is reproducible and statistically matches configured
  probabilities;
- Random/Heuristic/Simple cross-play completes full vector games using only
  legal actions;
- mixed vector PPO rollouts report per-opponent wins and totals;
- the full repository suite and detached ROCm mixed-training smoke are
  required before merge.

## Consequences

- The compute-budget pilot can run on the authoritative vector path against
  the full fixed gauntlet.
- No scalar games or Python agents execute inside vector rollout.
- Per-opponent rollout metrics remain comparable to object-runner reporting.
- Adding another fixed policy requires a batch selector and parity fixtures;
  unknown names continue to fail through `OpponentPool` validation.
