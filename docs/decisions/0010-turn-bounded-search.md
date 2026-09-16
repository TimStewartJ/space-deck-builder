# 0010 - Turn-bounded information-set search

**Status:** Accepted on `feat/turn-search` (2026-09-16)

## Context

PPO training has plateaued: architecture, width, and token-feature ablations
returned null results, and the reference self-play league gains little after
update ~200. Search can improve decisions at inference time without retraining,
and search results can later become training targets.

Star Realms has hidden information. A search must never read the opponent's
true hand, either player's true deck order, or the trade-deck order. Phase A
(`src/search/information_set.py`) projects a live decision into the acting
player's public knowledge and samples playable worlds from it.

Two properties of the rules shape the design:

- **The opponent takes no decisions during your turn.** Forced discards are
  deferred to the victim's own turn, so within one turn only your draws and
  trade-row refills are random.
- **The vector engine has no hidden order.** Draws and refills sample from
  zone counts with each game row's own RNG, so deck and trade-deck order are
  resampled naturally per row.

## Decision

`TurnSearch.choose(game)`:

1. projects the live game and converts the canonical public view to one
   vector row, asserting that its legal mask equals the scalar mask of the
   same view;
2. ranks legal actions by the policy prior (production END_TURN suppression
   applied) and keeps at most `max_candidates`;
3. evaluates candidates by truncated rollouts: apply the candidate, continue
   the searcher's own turn greedily with the policy, and stop when the turn
   passes, the game ends, or `max_turn_steps` is reached;
4. scores a finished game with its exact outcome (±1). A passed turn is
   scored with the **negated critic value of the opponent's first decision,
   evaluated from the opponent's own observation**. That is the acting-player
   kind of state the critic was trained on. The searcher's own view of an
   opponent-to-move state never occurs in training;
5. allocates `budget` rollouts by sequential halving and selects
   `argmax logits + (c_visit + max N) · c_scale · q̂`, with q̂ = (value + 1)/2.
   This is the Gumbel MuZero improvement rule evaluated deterministically,
   without Gumbel noise.

Rows for different candidates share world ids (common random numbers). Each
world re-deals only the opponent's unseen hand, from the canonical unseen
multiset. Publicly known hand cards stay in hand and the known top-deck
segment stays in the deck. Row RNG states derive from the search seed and
world id. The search seed combines the configured seed with the public
information fingerprint, so results depend only on public information.

`SearchAgent` wraps a search as a scalar agent. `python -m src search-eval`
plays the searcher against the same checkpoint without search (greedy by
default) with exact seat alternation and per-index deals shared across
budgets. Budget 0 is a fairness control.

## Rejected alternatives

**Critic value of the searcher's last pre-END_TURN state.** `V(s)` averages
over the policy's continuation from `s` and is not the value of ending the
turn there.

**The searcher's critic view after END_TURN.** The critic never trained on
observations where the observer is not the player to move.

**Per-simulation scalar determinization.** Phase A deep copies cost ~15 ms per
world. The vector re-deal needs only the projected root and count arithmetic.

**A full PUCT tree inside the turn.** Truncated rollouts with a value
evaluator are a well-established policy-improvement operator (Tesauro and
Galperin, 1996), vectorize across all candidates and worlds in the Numba
engine, and avoid the Python tree overhead that dominates small-budget
searches. A tree remains a follow-up if uplift saturates.

## Known limitations

- The critic estimates the acting player's outcome against its training
  opponent mixture, so both seats' values are optimistic and do not sum to
  zero. Candidates at one decision are compared on the same kind of leaf, so
  the offset does not change their ranking. It does not calibrate absolute
  values.
- Continuations are greedy, so a candidate is scored by the policy's line
  rather than the best line after it.
- Worlds are sampled per decision, not reused across decisions in a turn.

## Verification

- vector legality parity with the scalar information view at every root;
- world re-deals conserve the unseen multiset, keep known hand and top-deck
  constraints, share allocations by world id, and depend only on the unseen
  multiset;
- identical results for repeated searches and for games that differ only in
  hidden information;
- tactical fixtures where the prior prefers the wrong action: immediate lethal
  over attacking a base, destroying a blocking outpost to reach lethal, and
  buying the card the critic values;
- complete games with legal actions and budget accounting;
- `search-eval` summaries with Wilson intervals, Elo deltas, and an SPRT
  log-likelihood ratio.

## Result

At 64 rollouts per decision the searcher beats the same checkpoint playing
greedily on 56.4% of 2,000 paired deals across three seeds (+44 Elo). On
`selfplay_seed0`, 256 rollouts showed no detectable gain over 64 (paired 95%
CI −2.8 to +2.4pp), consistent with saturation but not proof of it. See
[`results/search_uplift.md`](../../results/search_uplift.md).
