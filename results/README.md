# Results Index

Every experiment writeup, newest first. All are measured against
[`docs/eval_protocol.md`](../docs/eval_protocol.md) — read that first if you
are comparing numbers across files.

Each writeup keeps its own eval/Elo logs beside it. Training logs and
checkpoints are gitignored, so anything cited here was force-added or lives
only on the machine that produced it.

| Date | Writeup | Question | Outcome |
|---|---|---|---|
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
- **There is no absolute strength anchor.** Every number here is relative to
  weak scripted opponents or to other checkpoints. Nothing establishes how
  well these agents actually play Star Realms.
