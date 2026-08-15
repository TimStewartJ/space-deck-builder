# 0003 - Compile declarative effects to a Numba SPMD engine

**Status:** Accepted for Gate 2 (2026-08-14), superseded in part by ADR 0005
**Context branch:** `feat/vector-engine-gate2`

## Context

The object engine expresses each game as `Game -> Player -> list[Card]`.
Training then spends CPU time enumerating actions and encoding those objects,
uses twenty worker processes, copies state through shared memory, and funnels
policy requests through an inference server.

Gate 1 measured the primitive alternatives at the real dimensions:

- Numba SPMD: 66.5M decisions/s, peaking at 96.0M with 16 threads
- NumPy masked SIMD: 1.71M decisions/s
- eager torch ROCm: 0.91M decisions/s (not a fused GPU ceiling)

The existing `Effect` trees are already a declarative source language. Writing
another independent card implementation would create the highest-risk failure
mode in this project: silently training a strong policy for subtly wrong rules.

## Decision

Compile the existing `Card`/`Effect` objects into fixed-width tables:

```
card -> effect bundles -> opcodes
```

Bundles preserve one encoded activation boundary and carry a trigger:

- `AUTO` - applied when the card is played
- `ALLY` - explicit action gated by faction count
- `SCRAP` - explicit self-scrap action
- `ACTIVATE` - explicit non-ally activation (including current COMPLEX no-ops)

Gate 2 opcodes are `NOP`, resource additions, draw, draw-per-faction, and
conditional draw. Cards containing target-selection, deferred-discard, or OR
effects fail closed with a reason.

Both a scalar IR interpreter and the Numba engine consume the compiled tables.
The existing object engine remains an external oracle. Differential tests:

1. compare every legal-action mask;
2. capture realized scalar draws and market refills;
3. execute the same encoded action in the Numba engine;
4. compare every canonical state field and 505-float observation after every
   decision.

The vector state uses card-count arrays:

```
zones [games, 2, 5, 49]
effect_ready [games, 2, 49, max_bundles]
```

Deck order is not stored. Differential mode supplies the scalar engine's
realized draws; production samples from the multiset. This is distributionally
exact for the currently implemented rules because implemented cards do not
observe deck order.

## Gate 2 scope correction

The design estimate said "27 no-choice cards." Auditing the real effect trees
showed that count mixed different definitions of choice. The compiler, rather
than a hand-maintained list, is authoritative:

- **26 canonical cards supported** under current scalar semantics, including
  starters, simple self-scrap abilities, deterministic conditional effects,
  and COMPLEX effects that are intentionally no-ops today;
- **23 cards rejected** because they require target selection, pending state,
  or OR choice.

Every supported card has its own scalar/vector differential test.

## Consequences

**Positive**

- Rules have one declarative source.
- Per-game logic uses readable real branches rather than hand-written masks.
- One CPU process replaces the worker/IPC architecture for supported rules.
- Current PPO observations and action indices stay compatible.

**Negative**

- Gate 2 is a vertical slice, not the full game. Unsupported cards cannot enter
  a vector game.
- Effect readiness is an additional state dimension required to preserve
  per-card activation semantics when card identity is collapsed to counts.
- Choice effects require continuation/pending-state IR in the next phase.

## Verification

- 26 per-card differential cases
- four simultaneous scalar/vector games through 250 decision steps
- legal masks and observations compared before every step
- all canonical state fields compared after every step
- actual PPO model integration on CPU and ROCm
- 434 repository tests passing

The detached 16,000-game Gate 2 benchmark measured:

- **7.31M decisions/s** for engine + mask + current 505-float observation
- **542k decisions/s** with the actual GPU policy in the loop

The environment path exceeds the integrated policy path by about 13x, meeting
the requirement that rollout becomes policy-bound.
