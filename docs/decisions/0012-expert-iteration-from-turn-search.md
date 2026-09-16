# 0012 - Expert iteration from turn-bounded search

**Status:** Accepted on `feat/expert-iteration` (2026-09-16). Round 1 was
pre-registered before any data was generated.

## Context

Turn-bounded search beats the same checkpoint playing greedily by +44 Elo
([`results/search_uplift.md`](../../results/search_uplift.md)), and batched
search generates searched games about 30× more cheaply
([`results/batched_search.md`](../../results/batched_search.md)). PPO training
has plateaued. Expert iteration trains the network on what search found, so
the improved instinct can guide stronger search in later rounds.

## Decision

**Data.** `python -m src search-selfplay` records games where both seats are
the same checkpoint searching at a fixed budget. Every decision becomes a
training position: the acting player's observation, legal actions with
production END_TURN suppression, the search's improved policy, the critic
root value, and the game outcome from the acting player's perspective.

- Deals depend only on the master seed and game index.
- Shards contain complete games only and are written atomically, so
  generation resumes or extends by replaying missing games.
- Resume identity covers the checkpoint hash, search settings, search
  implementation, and shard format.

**Policy target.** The completed-Q improved policy of Gumbel MuZero
(Danihelka et al., 2022):
`softmax(logits + σ(completedQ))` over legal actions, where:

- evaluated candidates use their rollout mean;
- every other legal action uses
  `v_mix = (v_root + ΣN · Σ π(a)q(a) / Σ π(a)) / (1 + ΣN)` over evaluated
  actions;
- `σ(q) = (c_visit + max N) · c_scale · (q + 1) / 2` is the transform search
  uses to select moves.

The target keeps the prior's relative preferences among actions search did not
evaluate. A forced decision targets its only legal action.

**Value target.** The game outcome (+1 win, −1 loss), the undiscounted
return PPO trains the critic on.

**Distillation.** `python -m src distill` starts from the checkpoint that
guided the search and minimizes masked policy cross-entropy plus
`value_coef` × outcome squared error with Adam and gradient clipping.

- Games with `index % validation_modulus == 0` are held out.
- Training stops after `patience` epochs without validation improvement.
- The best validation checkpoint is saved with its history.
- Validation reports policy KL to the target, KL from the initial policy,
  agreement with the searched and initial greedy moves, and critic explained
  variance.

## Pre-registered round 1

Fixed before generating data. None of these settings change after seeing
distillation or gate results.

| Setting | Value |
|---|---|
| Source checkpoint | `selfplay_seed0` from `v0.3-selfplay-baseline` |
| Self-play | 8,192 games, budget 64, 8 candidates, seed 0 |
| Distillation | lr 1e-4, batch 8,192, at most 8 epochs, patience 2, value coefficient 0.5, gradient clip 0.5, validation modulus 20, seed 0 |

**Primary gate.** The distilled checkpoint at temperature 0 plays the source
checkpoint at temperature 0 through `python -m src elo`
(`--checkpoint-temperatures 0`, 2,000 games, exact seat alternation). Round 1
**passes** if the distilled checkpoint scores at least 52.9% (+20 Elo) and its
Wilson 95% lower bound is above 50%.

**Retention gate.** The distilled checkpoint's locked 15,000-game gauntlet at
temperature 0 (seed 0, exact seats) must score at least 91.86% against
Heuristic: the source checkpoint's published temperature-0 result (92.86%,
[`results/greedy_inference.md`](../../results/greedy_inference.md)) minus one
point.

**Diagnostic (not gating).** `search-eval` with the distilled checkpoint at
budgets 0 and 64 on 1,000 deals, to see whether search still adds strength on
top of the distilled policy.

**Consequence.** If both gates pass, repeat round 1 for seeds 1 and 2 and
start round 2 from the distilled checkpoints. If either gate fails, improve
leaf evaluation before more rounds.

## Rejected alternatives

**One-hot targets on the searched move.** They discard the prior's ordering
among unevaluated actions, which later rounds need to rank candidates.

**Visit-count targets.** Sequential halving allocates visits by schedule, not
preference, so visit counts are a poor policy estimate.

**Critic targets from search values.** Rollout means are biased by greedy
continuations and critic leaves; outcomes are unbiased.
