# Capacity/training-time probe — cell D @ 400 updates

**Date:** 2026-04-23
**Companion:** [`results/attention_ablation.md`](attention_ablation.md)
**Branch:** `ablation/attention-results`

## Hypothesis

The 4-cell ablation showed A (mlp/sum) beats D (attn/attn) by 27 Elo, but all cells were plateaued at similar gauntlet numbers. Two competing explanations:

1. **Capacity-bound**: attention models need more parameters/width to express their advantage. (Original plan: [512,512] trunk.)
2. **Training-time-bound**: attention models need more updates to converge to the same level.
3. **Fundamental disadvantage**: attention is worse for this task at any reasonable compute.

The planned width probe required a new `--trunk-hidden-sizes` CLI flag. Adding one at 03:00 AM would have violated the worktree-isolation rule, so we pivoted to a **training-time probe** using existing CLI: cell D with `--updates 400 --lr-horizon 400` (2× v0.1 training, cosine LR spans the full horizon).

## Setup

| | Probe | Reference D0 | Reference A0 |
|---|---|---|---|
| Architecture | attn / attn | attn / attn | mlp / sum |
| Updates | **400** | 200 | 200 |
| LR horizon | 400 (cosine 3e-4 → 1e-5 over 400) | 200 | 200 |
| Seed | 0 | 0 | 0 |
| Wall time | 12121 s (3h22m) | 6475 s | 6345 s |
| Checkpoint | `ppo_agent_0423_0628_upd400_wins3195.pth` | `ppo_agent_0422_1102_upd200_wins3199.pth` | `ppo_agent_0421_1915_upd200_wins3193.pth` |

All other hyperparameters identical to the actual random-only v0.1/attention-ablation setup.

## Gauntlet (5000 games × 3 opponents, seed 0)

| Config | vs random | vs heuristic | vs simple | overall |
|---|---:|---:|---:|---:|
| A0 (mlp/sum, 200 upd) | 99.8% | **67.8%** | 96.0% | 87.9% |
| D0 (attn/attn, 200 upd) | 99.9% | 65.7% | 94.7% | 86.8% |
| **D_400 (attn/attn, 400 upd)** | 99.9% | **67.4%** | 95.7% | 87.6% |

**D_400 closes the heuristic gap almost completely** (+1.7pp vs D0, only −0.4pp vs A0). Gauntlet alone would suggest the gap is training-time-bound.

## Mini cross-play Elo (1000 games / pairing, 3 ckpts + 3 anchors)

| Rank | Checkpoint | Elo | Win % |
|---:|---|---:|---:|
| 1 | A0 | 1000 | 75.1% |
| 2 | **D_400** | **977** | 72.4% |
| 3 | D0 | 958 | 70.1% |
| — | heuristic | 838 | 55.8% |
| — | simple | 501 | 24.9% |
| — | random | 26 | 1.7% |

**D_400 gained +19 Elo over D0** — a real improvement from extra training.

## Head-to-head (decisive result)

| Matchup | Win rate (row) | Games |
|---|---:|---:|
| A0 vs D0 (200 upd) | **54.1%** | 1000 |
| A0 vs D_400 | **54.3%** | 1000 |
| D_400 vs D0 | 54.0% (D_400 wins) | 1000 |

**The A↔D gap is identical regardless of D's training time.** Extra training made D better in absolute terms (beats D0 at 54%), but against A the margin is unchanged at 54.3%. A just matches D's gains, because A is higher on the learning curve too — except A doesn't actually get to run 400 updates here, yet still holds the same lead.

## Interpretation

Hypothesis ranking after this probe:

- ❌ **Training-time-bound** (refuted): D_400 gains improve D's absolute strength but do not close the A↔D head-to-head gap. A extracted the same advantage in half the compute.
- ❓ **Capacity-bound** (untested): the width probe was deferred. Still plausible if attention needs wider trunk to express its advantage.
- ✅ **Fundamental disadvantage at this scale** (supported): at [256,256] trunk / 32-d embeddings / 494-dim action space / Star Realms complexity, the attention actor is strictly dominated by the MLP actor. Extra training helps D improve, but the ceiling is lower.

---

# Width probe (2026-04-23 evening)

**Branch:** `feature/width-probe-cli`
**Companion CLI:** new `--trunk-hidden-sizes`, `--actor-head-sizes`, `--critic-head-sizes`, `--card-emb-dim` flags (commit `f4366b0`).

## Hypothesis

The training-time probe above ruled out *training-time-bound*. The remaining live alternative was *capacity-bound*: maybe attention needs a wider trunk to express its advantage. Test by widening cell A (mlp/sum) and cell D (attn/attn) from `[256,256]+[128]` to `[512,512]+[256]` (~3× params: 0.26M → 0.79M) and re-running the head-to-head.

## Setup

| | A_512 | D_512 | Reference A0 | Reference D0 |
|---|---|---|---|---|
| Architecture | mlp / sum | attn / attn | mlp / sum | attn / attn |
| Trunk hidden | [512, 512] | [512, 512] | [256, 256] | [256, 256] |
| Actor/critic heads | [256] | [256] | [128] | [128] |
| Params | 0.79M | 0.79M | 0.26M | 0.26M |
| Updates | **175** ⚠ | 200 | 200 | 200 |
| Seed | 0 | 0 | 0 | 0 |
| Training opponents | random | random | random | random |
| Self-play | disabled | disabled | disabled | disabled |
| Wall time | ~105 min (crashed at upd177) | 128 min | 6345 s | 6475 s |
| Checkpoint | `ppo_agent_0423_1930_upd175_wins3196.pth` | `ppo_agent_0423_2139_upd200_wins3196.pth` | `ppo_agent_0421_1915_upd200_wins3193.pth` | `ppo_agent_0422_1102_upd200_wins3199.pth` |

Checkpoint files are local run artifacts and are not committed because `*.pth` is ignored.

**⚠ A_512 caveat:** crashed at upd 177/200 with `RuntimeError: zero-dimensional tensor (at position 342) cannot be concatenated` in `src/ai/ppo_agent.py:284` (`torch.cat(all_vals)`). Pre-existing PPO bug, manifested at higher capacity and fixed separately in `d036ba1`. Salvaged via the upd175 checkpoint, which is acceptable because cosine LR was already at 2.11e-05 (close to the 1e-5 floor) and gauntlet vs random was saturated at 99.96%. The last 25 updates would have been cosmetic.

## Gauntlet (15 000 games × 3 opponents, seed 0)

| Cell | vs random | vs heuristic | vs simple | overall |
|---|---:|---:|---:|---:|
| A0 (256) | 99.8% | 67.7% | 96.0% | 87.9% |
| D0 (256) | 99.9% | 65.7% | 94.7% | 86.8% |
| **A_512** | 100.0% | 67.1% | 96.2% | 87.7% |
| **D_512** | 99.9% | **68.2%** | 95.5% | **87.8%** |

**Read:** width does ~nothing for A (87.9% → 87.7%, well within noise). D *appears* to gain +1.0pp overall and edge A_512 by 1.1pp on heuristic. If we stopped here, "attention scales better with width" would look plausible.

But the ablation already taught us that gauntlet alone is misleading — D_400 closed the heuristic gap and still lost to A head-to-head 54%. Decisive test is cross-play.

## Cross-play Elo (1000 games per pair, seed 0)

| Rank | Participant | Elo | Win% |
|---:|---|---:|---:|
| 1 | A0_256 | 1000 | 70.6% |
| 2 | A_512 | 999 | 70.4% |
| 3 | D_512 | 970 | 66.9% |
| 4 | D0_256 | 968 | 66.7% |
| 5 | heuristic | 856 | 53.0% |
| 6 | simple | 484 | 20.8% |
| 7 | random | 35 | 1.6% |

**Width is essentially neutral for both architectures** (A_512 −1 Elo vs A0; D_512 +2 Elo vs D0 — both inside noise). The 30-Elo A↔D gap from the original ablation is preserved.

## Head-to-head (key pairings, win% for row vs column)

| | A0_256 | A_512 | D0_256 | D_512 |
|---|---:|---:|---:|---:|
| A0_256 | — | 50.9% | 53.0% | 55.7% |
| A_512 | 49.1% | — | 55.0% | **55.4%** |
| D0_256 | 47.0% | 45.0% | — | 51.1% |
| D_512 | 44.3% | **44.6%** | 48.9% | — |

The critical pairing **A_512 vs D_512: 55.4% / 44.6%** for the MLP. Per the launch agent's decision tree, that is the **D < A by ≥52%** branch.

## Verdict

✅ **Capacity-bound hypothesis: refuted.** Tripling the parameter budget did not let attention catch the MLP. A still beats D by 55% head-to-head at [512, 512] — within 1pp of the original 54% margin at [256, 256].

Combined with the training-time probe:

| Hypothesis | Status |
|---|---|
| Training-time-bound | ❌ refuted (D_400 probe) |
| Capacity-bound | ❌ refuted (this probe) |
| **Fundamental disadvantage at this task** | ✅ supported by both probes |

The MLP-actor + sum-pool combination is the clear default for Star Realms at every scale we have tested. The attention actor is not "underfit at 256" — it is genuinely worse at the action-selection problem this game presents.

## Recommendation

- Keep `actor_type=mlp` and `pool_type=sum` as the locked defaults.
- The architecture toggles are still useful as research instrumentation but should not be treated as live tuning knobs for the main training pipeline.
- Close the attention investigation. Future capacity work should explore other axes (more updates with larger batches, richer state features, opponent diversity) rather than wider attention trunks.
- Fix the `ppo_agent.py:284` zero-dim concat bug in a separate worktree before any future longer-horizon training. It only triggered here at higher width but is a latent crash for any run.

## Caveats

- n=1 seed per cell. The 30-Elo gap is consistent with the 3-seed ablation, but a multi-seed width sweep would tighten the conclusion.
- A_512 stopped 25 updates short. Highly unlikely to flip the verdict (LR was already at the cosine floor) but worth noting.
- Only one width step tested (512 vs 256). We have no evidence about [1024, 1024] or deeper trunks. The fundamental-disadvantage conclusion is *at the scales relevant to this hardware budget*, not a universal claim.

Additional observations:
- D_400's gauntlet heuristic result (67.4%) is within noise of A0's (67.7%), so the gauntlet alone is misleading — you need head-to-head play to see the gap. This is a useful methodological note for future ablations.
- The constant-margin result also means "attention catches up on heuristic but not on A" is a signature pattern. If we saw D_400 matching A head-to-head despite worse gauntlet, that would flip the story. We didn't.

## Recommendation

- **No change to defaults** beyond what the ablation already recommended (`actor_type=mlp`, `pool_type=sum`).
- **Capacity probe (width) remains open** and worth a proper worktree if we want a complete story. Suggested scope:
  - Add `--trunk-hidden-sizes` + `--actor-head-sizes` + `--critic-head-sizes` CLI passthrough
  - Train A and D at [512,512] trunk, 200 updates, seed 0
  - If D_512 still loses to A_512 head-to-head → attention is fundamentally disadvantaged on this task at any reasonable capacity. Close the book on attention.
  - If D_512 beats A_512 → capacity was the issue, revisit the ablation.
- **Do not run D at 400 updates as the main baseline** — the 2× compute for +19 Elo is a poor trade compared to running a new seed of A (which gives +6 Elo worth of noise reduction per seed for the same cost).

## Reproduction

Training log: `logs/timeprobe_D_400upd_seed0_20260423_030640.log`
Gauntlet log: `results/timeprobe_eval/D0_400.log`
Elo log: `results/timeprobe_elo.log`

```
python -m src train --actor-type attention --pool-type attention --seed 0 --updates 400 --lr-horizon 400
python -m src eval  --model <probe_ckpt> --opponents random,heuristic,simple --games 15000 --seed 0 --num-workers 16
python -m src elo   --checkpoints <probe> <A0> <D0> --agents random,heuristic,simple --games-per-pair 1000 --num-workers 16
```

