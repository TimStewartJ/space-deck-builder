# Eval Protocol

**Status:** Locked as of `v0.1-baseline`. This document is a contract.
Any change to the protocol invalidates prior results — every result must
either be re-run under the new protocol or explicitly marked legacy
with a pointer to the protocol version that produced it.

This protocol exists so that every architecture variant compared in
the writeup series uses the **same yardstick**. Without this, no result
in any later post is meaningful.

---

## 1. Opponent gauntlet

Every model is evaluated against three fixed opponents:

| Opponent   | Source                              |
|------------|-------------------------------------|
| `random`   | `src/ai/` random agent              |
| `heuristic`| `src/ai/` heuristic agent           |
| `simple`   | `src/ai/` simple agent              |

These three are baked into `--opponents` defaults across the `eval`,
`elo`, and `analyze` subcommands and serve as the writeup's permanent
reference points. They are intentionally weak-to-medium opponents; the
goal is **comparable progression across variants**, not absolute strength.

## 2. Sample size

- **5000 games per opponent**, per evaluated checkpoint.
- Total per checkpoint: 3 opponents × 5000 = **15000 games**.

Rationale: 5000 games gives a 95% CI half-width of ≤ ±1.4pp at a 50%
win rate (worst case) — tight enough to distinguish variants that
differ by ≥ ~3pp with confidence, which covers the typical effect
sizes seen from architecture changes in deep RL.

## 3. Seat protocol

- Exactly **2500 games as first player + 2500 as second player** per
  opponent.
- First-player advantage is real in Star Realms; without alternation,
  win rate measures seat luck as much as skill.

## 4. Training seeds

- Every training run that produces a checkpoint must be repeated with
  **3 independent seeds**: `0`, `1`, `2`.
- Each seed initializes Python `random`, `numpy`, and `torch`.
- Reported numbers are **mean ± 95% CI across the 3 seeds** (Student-t,
  df=2).

- Implementation: pass `--seed N` to `python -m src train`. The flag
  seeds Python `random`, `numpy`, and `torch` (CPU + all CUDA devices)
  before any model construction or worker-seed derivation, which
  propagates into the per-worker seed lists in `batch_runner` /
  `mp_batch_runner` (they draw from `random.randint`).

## 5. Metrics

### Primary — per-opponent win rate
For each opponent ∈ {random, heuristic, simple}:
- Mean win rate across the 3 training seeds.
- 95% CI from the per-seed win rates (Student-t, df=2).

### Secondary — cross-play Elo
- Run the `elo` subcommand pairing all evaluated checkpoints against
  each other and against the three gauntlet opponents.
- 5000 games per pairing.
- Anchor: `random` opponent is fixed at **Elo 0** in every report.
- Use the existing tournament implementation in
  `src/ppo/elo_tournament.py`.

### Tertiary — sample efficiency
- Number of PPO updates required for a training run to first reach
  **≥ 70% win rate vs. `heuristic`** in its routine eval (`eval_every`
  cadence). Reported as mean across the 3 seeds, or "not reached" if
  no seed crossed the threshold within the training budget.

## 6. Reporting format

Every post in the series reports its results in **exactly this table
shape**, one table per architecture variant:

```markdown
### Variant: <name>

|                  | vs. random      | vs. heuristic   | vs. simple      |
|------------------|-----------------|-----------------|-----------------|
| Win rate (mean)  | XX.X% ± Y.Y     | XX.X% ± Y.Y     | XX.X% ± Y.Y     |
| Sample efficiency (updates → 70% vs. heuristic) | mean: NN | NN | NN |
| Cross-play Elo (vs. random=0)                    | EEE             |
| Per-seed checkpoints | seed0=…  seed1=…  seed2=…                    |
```

Plus one shared summary table per post that stacks all variants
side-by-side on the **vs. heuristic** column (the headline number).

## 7. Compute & determinism caveats

- All training and evaluation runs use the project's GPU defaults
  (`cuda`); CPU fallback per `DeviceConfig.resolve` is acceptable for
  spot-checks but not for reported numbers.
- ROCm + PyTorch is **not bitwise-deterministic** even with the same
  seed. Three-seed averaging is what makes results comparable, not
  per-seed bit-for-bit reproduction.
- Hardware is recorded in each results file (`results/*.md` includes
  GPU model, ROCm/CUDA version, torch version).

## 8. Storage layout

```
docs/eval_protocol.md            # this file
results/baseline_v3.md            # Step 2 output
results/<variant_name>.md         # one per future variant
analysis/elo_<tag>_<timestamp>/   # raw Elo tournament outputs
models/<variant>_seed{N}.pth      # per-seed checkpoints
logs/<variant>_seed{N}_<ts>.log   # per-run training logs
```

## 9. Stability clause

Changes to **opponents, sample size, seat protocol, seed count, or
metrics** require:

1. A new protocol version (e.g., `v0.2-eval`) with a changelog at the
   bottom of this file.
2. Re-running every prior baseline under the new protocol, **or**
   explicitly tagging old results in `results/*.md` with the protocol
   version that produced them.

Cosmetic edits (typos, clarification, expanded prose) do not require
a version bump.

## 10. Changelog
- `v0.1` — initial protocol, locked at git tag `v0.1-baseline`.
