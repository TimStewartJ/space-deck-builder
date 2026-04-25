# Token Features Phase 1 — A vs A_tok

## Setup
- Branch: `feature/token-features-phase1` @ `8664e4f` (cube-free fast-path fix).
- Two PPO runs, `--seed 0 --updates 200`, default config, opponents = random:
  - **A baseline** — no token features. ckpt: `models/ppo_agent_0424_1351_upd200_wins3196.pth` (~0.10M params).
  - **A_tok** — `--token-features`. ckpt: `models/ppo_agent_0424_1911_upd200_wins3197.pth` (~0.27M params).
- A_tok wall time: 7673s (2h08m) vs A baseline ~1h50m → **1.16x slowdown** (predicted ~1.28x from GPU bench).

## Backstory: cube-free fast-path fix
The naive token encoding caused a 18-20x slowdown vs legacy on GPU. Root cause: cube allocation in the encoding loop. After the fix:
- CPU bench: tokens 0.64x of legacy speed (faster, even).
- GPU bench: tokens 5.26 ms/step vs legacy 4.10 ms/step → **1.28x slowdown** (acceptable).

Wall-time observed (1.16x) tracks the GPU bench prediction.

## Gauntlet (15,000 games, seed 0, 16 workers)

| Opponent  | A baseline | A_tok    | Δ (pp)  |
|-----------|------------|----------|---------|
| random    | 99.90%     | 99.88%   | -0.02   |
| heuristic | 67.64%     | 68.48%   | +0.84   |
| simple    | 95.78%     | 95.50%   | -0.28   |
| **overall** | **87.77%** | **87.95%** | **+0.18** |

All deltas within ±2% → **TIED** per decision tree.

## Mini-Elo (1000 games per pairing, full round-robin)

| Rank | Agent              | Elo  | Win%  |
|------|--------------------|------|-------|
| 1    | 0424_1911 (A_tok)  | 1015 | 79.2% |
| 2    | 0424_1351 (A)      | 1000 | 77.6% |
| 3    | heuristic          | 866  | 62.4% |
| 4    | simple             | 504  | 28.6% |
| 5    | random             | 53   | 2.2%  |

- Head-to-head: **A_tok 509–491 vs A** (51%, +15 Elo — within noise on 1k games).
- Per-opponent in Elo run: A_tok beats heuristic 70.1% vs A's 65.3% (+4.8pp). The 15k gauntlet showed only +0.84pp on heuristic, so the +4.8pp here is likely sample variance from the smaller (1k) Elo subgame.

## Verdict
**Tied.** Token features neither help nor hurt at this capacity/training budget. The 2.7x parameter increase (0.10M → 0.27M) does not produce a measurable win rate edge after 200 updates against a random-only opponent pool.

## Recommendation
- **Keep `--token-features` behind a flag**, do NOT make it the default.
- Branch worth keeping for potential follow-ups; the encoding pipeline is now fast enough to be a reasonable choice.
- If we want to know whether tokens scale better, the next experiment should be:
  - **Multi-seed (3 seeds)** at the same budget to confirm the tie holds.
  - **Higher-capacity / longer training** (e.g. 1000 updates, mixed opponents) where the extra params might actually be utilized.
- Negative-result document; do not merge to master without one of the above experiments.
