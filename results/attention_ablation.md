# Attention ablation — 4 architectures × 3 seeds

**Date:** 2026-04-23
**Baseline reference:** [`results/baseline_v3.md`](baseline_v3.md) (v0.1, tag `baseline/2026-04-22`)
**Codebase tag at training time:** post-cleanup (`b6f4523`)

## Design

Two architectural toggles, four cells, three seeds each:

| Cell | `actor_type` | `pool_type` | Notes |
|------|:------------:|:-----------:|-------|
| A | mlp | sum | v0.1 baseline (reused checkpoints) |
| B | mlp | attention | swap pooling only |
| C | attention | sum | swap actor only |
| D | attention | attention | swap both (full attention) |

All other hyperparameters identical to v0.1: 200 updates × 16k episodes, cosine LR 3e-4 → 1e-5, self-play ratio 0.5 (cosine), opponents `random,heuristic,simple`, trunk `[256,256]`, actor/critic heads `[128]`, card embedding 32-d. Seeds 0/1/2.

**Compute:** 9 new training runs × ~108 min on AMD RX 9070 + ROCm 7.2 (single-stream InferenceServer). Cell A reused v0.1 checkpoints (3 runs, ~5h saved).

## Gauntlet (5000 games × 3 opponents per checkpoint, seed 0)

Win rate vs each rule-based opponent. All cells beat random ≥99.7% and simple ≥94%; only `heuristic` discriminates.

| Cell | Architecture | vs random | vs heuristic | vs simple | overall (15k) |
|------|---|---:|---:|---:|---:|
| A | mlp / sum | 99.9% | **67.7%** ± 0.5 | 95.8% | 87.9% |
| B | mlp / attn | 99.8% | 65.7% ± 0.9 | 94.4% | 86.5% |
| C | attn / sum | 99.9% | **67.6%** ± 1.0 | 95.9% | 87.8% |
| D | attn / attn | 99.9% | 66.7% ± 0.9 | 95.1% | 87.2% |

(Mean across 3 seeds; std reported for `vs heuristic` only.)

## Cross-play Elo (105 pairings × 1000 games, anchors: random/heuristic/simple)

15 participants, 105k games total. Results sorted; 7-char IDs map to cells via the legend below.

| Rank | Checkpoint        | Cell | Elo  | Win % |
|-----:|-------------------|:----:|-----:|------:|
| 1    | 0421_2055_upd200 | A1   | 1006 | 61.0% |
| 2    | 0421_1915_upd200 | A0   | 1000 | 60.5% |
| 3    | 0421_2237_upd200 | A2   |  997 | 59.8% |
| 4    | 0423_0153_upd200 | C2   |  994 | 59.4% |
| 5    | 0422_1446_upd200 | C0   |  991 | 59.0% |
| 6    | 0422_2040_upd200 | C1   |  983 | 57.9% |
| 7    | 0422_2233_upd200 | D2   |  978 | 57.3% |
| 8    | 0422_1842_upd200 | B1   |  978 | 57.2% |
| 9    | 0422_1102_upd200 | D0   |  978 | 57.2% |
| 10   | 0422_1254_upd200 | B0   |  973 | 56.5% |
| 11   | 0423_0008_upd200 | B2   |  966 | 55.6% |
| 12   | 0422_1652_upd200 | D1   |  966 | 55.6% |
| —    | heuristic         | —    |  852 | 40.9% |
| —    | simple            | —    |  452 | 11.2% |
| —    | random            | —    |  −41 |  0.7% |

**Cell mean Elo (± seed-stdev across 3 runs):**

| Cell | Mean Elo | σ | Δ vs A |
|------|---------:|---:|------:|
| A | **1001** | 4.5 |   — |
| C |   989 | 5.7 | −12 |
| D |   974 | 6.9 | −27 |
| B |   972 | 6.0 | −29 |

## Head-to-head: cell-aggregated win rate (row vs column)

Each off-diagonal pairing aggregates 9 cross-seed pairings × 1000 games = **9000 games**. With binomial σ ≈ 0.5%, differences ≥1.5% are statistically meaningful.

|       |   A   |   B   |   C   |   D   |
|:-----:|:-----:|:-----:|:-----:|:-----:|
| **A** |   —   | 54.3% | 51.8% | 54.4% |
| **B** | 45.7% |   —   | 47.6% | 49.3% |
| **C** | 48.2% | 52.4% |   —   | 52.5% |
| **D** | 45.6% | 50.7% | 47.5% |   —   |

**Pecking order:** A > C > B ≈ D, with all gaps from A statistically significant (≥3.5σ).

## Findings

1. **Baseline mlp/sum (A) wins outright.** It tops the Elo leaderboard, leads the gauntlet, and beats every other cell head-to-head with statistical significance.
2. **Attention pooling (C) is nearly free.** Only −12 Elo and −0.1pp gauntlet vs A. If we wanted attention pooling for downstream reasons (interpretability, transfer), the cost is small.
3. **Attention actor (B, D) is harmful.** Both cells using `actor_type=attention` underperform A by ~25-29 Elo and ~2pp gauntlet. Adding attention pooling on top of an attention actor (D) does not recover the loss.
4. **Seed variance is real but smaller than cell gaps.** Within-cell σ ≈ 4-7 Elo; between-cell mean gap A↔B is 29 Elo (~4× the noise floor). Three seeds was sufficient to call this.
5. **No architecture cell beats baseline** on any metric. The cleanest action is to **leave defaults at `mlp` / `sum`** and remove attention from consideration unless a future task (multi-card chains, larger card pool) reveals a niche where it pays.

## Open questions

- **Is the [256,256] trunk saturated?** All four cells plateau at very similar overall numbers (86-88% gauntlet). Untested with this experiment. A training-time probe with cell D + `--updates 400` is launching now to test whether attention models specifically need more compute. See `results/capacity_probe.md` (forthcoming) for results.
- **Does attention help on harder opponents?** Only `heuristic` discriminates the cells; `random` and `simple` are saturated. We do not have a stronger rule-based opponent yet. Open question for future v0.2 work.
- **Is the attention-actor regression a learning-rate or capacity issue?** The default LR was tuned for the mlp actor. A small LR sweep on cell D could resolve this but is out of scope here.

## Recommendation

- **Keep defaults `actor_type=mlp`, `pool_type=sum`.** Already the case in `src/config.py`.
- **Do NOT remove the attention code paths** — they are validated to train cleanly, and may pay off on a richer task or with capacity changes. They are now a regression-tested option.
- **Tag this result** as `ablation/attention/2026-04-23` once the writeup is reviewed.

## Reproduction

Eval logs: `results/ablation_eval/{A0,…,D2}.log`
Elo log: `results/ablation_elo_full.log`
Pairings (extracted): `results/ablation_elo_pairings.txt`
Per-run training logs: `logs/ablation_<CELL>_20260422_111746.log`

Checkpoint paths: see Elo table above. v0.1 baseline checkpoints are also referenced from `results/baseline_v3.md`.

To re-run a single cell from scratch:
```
python -m src train --actor-type <mlp|attention> --pool-type <sum|attention> --seed <0|1|2>
```

To re-run gauntlet/Elo only, see the inline commands in this session's history (the `eval` subcommand and `elo --checkpoints`).
