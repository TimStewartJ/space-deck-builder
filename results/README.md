# Results Index

Every experiment writeup, newest first. All are measured against
[`docs/eval_protocol.md`](../docs/eval_protocol.md) — read that first if you
are comparing numbers across files.

Each writeup keeps its own eval/Elo logs beside it. Training logs and
checkpoints are gitignored. Logs a writeup depends on are force-added; from
`v0.3-selfplay-baseline` onward, checkpoints and snapshot pools are published
as verified GitHub release bundles on the result's tag (see
[`docs/eval_protocol.md`](../docs/eval_protocol.md#8-storage-layout)).

| Date | Writeup | Question | Outcome |
|---|---|---|---|
| 2026-09-16 | [`expert_iteration_round1.md`](expert_iteration_round1.md) | Does training a checkpoint on its own search's improved policies and outcomes make its greedy play stronger? | **Pre-registered FAIL, narrowly.** The distilled checkpoint beat its source 52.8% at temperature 0 (+19 Elo, 95% CI +4 to +35) against a +20 Elo threshold, with no detectable gauntlet loss. It adopted only about one in five of search's move changes, and search on it adds no detectably less than on its source. |
| 2026-09-16 | [`batched_search.md`](batched_search.md) | Can search run many games' decisions together from public-count roots without losing strength? | **Yes.** On identical deals: 34× less search time per decision and 7.6× faster wall time from two processes instead of eight; the two-worker topology plays 382 games/min in a separate benchmark. Paired strength against the previous implementation: −0.8pp (95% CI −2.1 to +0.5) over 2,000 deals; the greedy control reproduced all 1,000 published games. |
| 2026-09-16 | [`search_uplift.md`](search_uplift.md) | Does turn-bounded information-set search beat the same checkpoint playing greedily, and how does it scale with rollouts? | **Yes, modestly.** At 64 rollouts per decision search wins 56.4% of 2,000 paired deals across three seeds (+44 Elo, McNemar p = 8 × 10⁻¹⁰). On one checkpoint, 256 rollouts showed no detectable gain over 64. |
| 2026-09-16 | [`greedy_inference.md`](greedy_inference.md) | Do the published checkpoints play better when they choose their highest-probability action instead of sampling? | **Yes.** Temperature 0 raises win rate against Heuristic by +6.15pp (self-play) and +6.18pp (fixed), removing about 45% of those losses, and each checkpoint beats its sampled self (+43 and +72 Elo). Self-play still leads cross-play. Greedy play is the baseline search must beat. |
| 2026-08-18 | [`rocm715_stack_probe.md`](rocm715_stack_probe.md) | Does the July torch 2.12/ROCm 7.15/gfx1201/Triton stack improve training or eliminate runtime degradation? | **No.** It passed all compatibility gates but made the matched 30-update workload 66.9% slower and retained process-age degradation. Production remains exact-pinned to torch 2.11/ROCm 7.13. |
| 2026-08-18 | [`optimized_selfplay_baseline.md`](optimized_selfplay_baseline.md) | Does optimized snapshot self-play improve the selected 400×2,000 fixed-opponent baseline across three seeds? | **Yes in cross-play, tied overall on scripted opponents.** Under exact 50/50 seating, self-play occupies the top three Elo positions and wins 67.9% against fixed-baseline checkpoints. Its fixed gauntlet is 94.40% overall vs. 94.71% for the control. |
| 2026-08-17 | [`snapshot_inference_optimization.md`](snapshot_inference_optimization.md) | Can vector PFSP self-play avoid fragmented snapshot inference and long-process ROCm drift? | **Yes.** Actor-only stacked inference plus 25-update process chunks cuts degraded cap-3 rollout from 12.41s to 5.42s (2.29x). A paired seed-0 selection pilot keeps cap 3 as the conservative default; cap 1 is an optional 3.27s fast path. |
| 2026-08-16 | [`compute_budget_pilot.md`](compute_budget_pilot.md) | At a fixed 800k-episode budget, how should updates trade against episodes per update? | **400 × 2,000.** The preregistered Heuristic rule selected the smallest batch; its three fixed-opponent seeds pooled 87.18% against Heuristic and became the control for the self-play baseline. |
| 2026-08-14 | [`vector_engine_gate.md`](vector_engine_gate.md) | Is a struct-of-arrays compiler engine fast enough and rules-exact end to end? | **Yes.** All 49 cards compile and match scalar state/masks/observations. The full engine runs at 5.88M/s, the actual-policy path at 587k/s, and `train --engine vec --opponents random` completes rollout, PPO update, scalar eval, and checkpoint save. |
| 2026-04-28 | [`curriculum_v02.md`](curriculum_v02.md) | Should the default training curriculum be the fixed mixed pool, or self-play? | Mixed fixed opponents became the default (+6.8pp gauntlet, +136 Elo vs random-only). Self-play is stronger still (+203 Elo, 59.7% head-to-head) but ~1.8x the wall time, so it stays opt-in. |
| 2026-04-24 | [`token_features_phase1.md`](token_features_phase1.md) | Do static per-card feature tokens beat a pure embedding lookup? | **Tied.** Kept behind `--token-features`, off by default. See the correction at the top — the parameter delta was ~1%, not 2.7x, so this did not test a capacity increase. |
| 2026-04-23 | [`capacity_probe.md`](capacity_probe.md) | Was the attention actor just underfit — capacity- or training-time-bound? | Both hypotheses refuted. Attention is genuinely worse at this task, at every scale tested. Attention investigation closed. |
| 2026-04-23 | [`attention_ablation.md`](attention_ablation.md) | Does an attention actor or attention pooling beat `mlp`/`sum`? | No cell beat baseline. Attention pooling is ~free (−12 Elo); the attention *actor* is harmful (−25 to −29 Elo). Defaults stay `mlp`/`sum`. |
| 2026-04-22 | [`baseline_v3.md`](baseline_v3.md) | What is the reference baseline? | v0.1 baseline: `mlp`/`sum`, 200 updates, 3 seeds. Superseded as the default recipe by `curriculum_v02`, but still the random-only reference point. |

## Standing conclusions

- **Architecture is settled for now.** `actor_type=mlp`, `pool_type=sum`.
  Three separate experiments (ablation, capacity probe, token features) all
  returned null. Treat further architecture variation as low-yield until
  something below it changes.
- **The fixed gauntlet is saturating.** `random` (~99.9%) and `simple`
  (~98.9%) no longer separate strong checkpoints; `heuristic` carries almost
  all the discriminating signal. Cross-play Elo is the more sensitive metric.
- **Snapshot self-play is the stronger checkpoint league.** Its three seeds
  occupy the top three positions against the fixed-baseline seeds even though
  the scripted gauntlet is statistically tied.
- **Sampled evaluation understates the policies.** The locked protocol samples
  each checkpoint's policy; greedy decoding of the same checkpoints is a
  separate, stronger variant ([`greedy_inference.md`](greedy_inference.md)).
- **Search adds strength on top of greedy play.** Turn-bounded search gains
  about +44 Elo over the greedy policy at 64 rollouts. On one checkpoint, 256
  rollouts showed no detectable gain over 64. That is consistent with leaf
  evaluation limiting search, but the experiments do not isolate the cause
  ([`search_uplift.md`](search_uplift.md)).
- **There is no absolute strength anchor.** Every number here is relative to
  weak scripted opponents or to other checkpoints. Nothing establishes how
  well these agents actually play Star Realms.
