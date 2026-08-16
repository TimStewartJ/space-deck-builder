# 0005 - Make Rule IR authoritative and complete Core Set rules

**Status:** Accepted (2026-08-15)

## Context

The object engine applied `Effect` objects through a large Python type switch,
while the vector engine separately translated the same objects to opcodes.
Although differential tests constrained drift, two semantic implementations
still existed. Seven Core Set cards also depended on `COMPLEX` effects that
were intentionally treated as no-ops.

Those limitations were acceptable for the vector-engine performance gate, but
not for training a policy intended to play the physical game.

## Decision

Version the shared Rule IR in `src/rules/ir.py`. It owns:

- effect opcodes and trigger classes;
- target and pending flags;
- `Effect` to opcode compilation;
- auto, ally, scrap, explicit-activation, and passive trigger classification.

`Effect.apply()` now delegates to `src/rules/scalar.py`. The scalar interpreter
and Numba compiler consume the same opcode and trigger contract; neither keeps
an independent effect-type switch.

Implement the previously absent physical mechanics:

- Blob Carrier free ship acquisition directly to the top of the deck;
- Brain World draw-on-completed-scrap accounting;
- Mech World wildcard ally identity without changing its faction;
- Stealth Needle per-instance copied rules and faction for the turn;
- Fleet HQ's later-ship event trigger, including stacking;
- Central Office and Freighter next-acquired-ship top-deck effects.

The vector state therefore adds:

- an ordered top-deck segment above the existing deck multiset;
- per-active-copy effective rule-card and faction fields;
- copy and free-acquisition pending modes;
- next-ship-top counters.

Physical card identity remains separate from effective rules. A copied scrap
ability scraps Stealth Needle, and action indices continue to refer to the
physical card. Mech World's wildcard contributes to ally checks only.

## Verification

- a versioned compiler snapshot covers every canonical card;
- focused physical-rule fixtures cover every formerly complex card;
- scalar/vector tests compare masks, observations, pending state, ordered deck
  state, effective per-copy rules, and complete canonical state after actions;
- complete seeded differential games remain the cross-backend oracle;
- the guarded fixed-opponent vector training path remains a required smoke.

## Consequences

- Rule semantics have one executable source of truth.
- New card mechanics must first receive an IR opcode/trigger and scalar
  interpreter behavior, then a vector implementation under differential tests.
- Stealth Needle and ordered top-deck effects add modest SoA state and branches
  to the hot path.
- ADRs 0003 and 0004 remain the performance and rollout history, but their
  references to intentional Core Set no-ops are superseded by this decision.
