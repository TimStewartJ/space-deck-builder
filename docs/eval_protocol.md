# Eval Protocol

**Status:** Locked as of the v0.1 baseline (git tag `baseline/2026-04-22`).
This document is a contract.
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
- Evaluation runners deterministically alternate the requested model's seat
  by global game index and record both seat counts. A locked 5000-game
  opponent result is invalid unless both counts equal 2500. Elo pairings
  alternate side A and side B in the same way.

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
- **1000 games per pairing** (`--games-per-pair 1000`).
- Anchor: `compute_mle_ratings` pins the **first participant** (index 0,
  i.e. the first `--checkpoints` entry) at **Elo 1000** to fix the scale.
  Every other player — including `random` — floats relative to it, so pass
  the reference checkpoint first and name it when reporting.
- Use the existing tournament implementation in
  `src/ppo/elo_tournament.py`.

> Elo is the more sensitive metric once the fixed gauntlet saturates.
> `random` and `simple` no longer separate strong checkpoints, so
> head-to-head play carries most of the discriminating signal.

### Tertiary — sample efficiency
- Number of PPO updates required for a training run to first reach
  **≥ 70% win rate vs. `heuristic`** in its routine eval (`eval_every`
  cadence). Reported as mean across the 3 seeds, or "not reached" if
  no seed crossed the threshold within the training budget.

### Inference temperature
- Checkpoints play at **temperature 1** (sampling their policy) unless a
  result names another temperature. Every published result before
  `v0.1.3` used temperature 1.
- Temperature is part of a variant's identity: `eval --temperature T` and
  `elo --checkpoint-temperatures ...` evaluate the same checkpoint as a
  different player. Temperature 0 selects the highest-probability legal
  action. Results at different temperatures are compared as paired
  variants of the same seeds, never pooled.
- Built-in opponents and training rollouts are unaffected.

### Held-out league (from `v0.2`)
- Measures a checkpoint against six scripted opponents that no agent
  trains on (`src/ai/held_out_agents.py`: balanced, aggro, economy,
  scrapper, fortress, blobrush). Keep them out of every training opponent
  registry, and do not tune them against checkpoints.
- `python -m src league --models <checkpoints> --output-json <file>`: the
  checkpoint plays at **temperature 0**, **4,000 games per opponent**,
  deals `game_seed(101, index)`, the checkpoint first on even indices, so
  exactly 2,000 games in each seat.
- Report wins with a Wilson 95% interval per opponent, and the pooled win
  rate over the 24,000 games, which is the **league score**.
- `--calibration` plays each held-out opponent against Heuristic (2,000
  games, deal seed 103) as a check on the opponents themselves.

### Purchase audit (from `v0.2`)
- Measures how much win probability a checkpoint's purchase decisions give
  away, scored by complete greedy rollouts rather than by the critic.
- `python -m src purchase-audit --model <checkpoint> --output-json <file>`:
  1,000 decisions, one per greedy self-play game sampled uniformly among its
  eligible purchase decisions, 256 rollouts per candidate, master seed
  20260930.
- Headline: the **confident-override gain** per decision in points of win
  probability, with its standard error. Also report the share of decisions
  with a clear mistake and the gain of the one-turn search's choice, overall
  and by turn counter.

The full protocol of both measures, including eligibility, sampling, and
formulas, is fixed in
[`docs/decisions/0013-strength-yardstick.md`](decisions/0013-strength-yardstick.md).

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
docs/eval_protocol.md             # this file
results/baseline_v3.md            # Step 2 output
results/<variant_name>.md         # one per future variant
results/<variant_name>/           # that variant's committed evidence: eval and
                                  #   Elo logs, metrics, manifests, summaries
                                  #   (*.log is gitignored; force-add it)
analysis/elo_<tag>_<timestamp>/   # raw Elo tournament outputs
models/ppo_agent_<ts>_upd<N>_wins<W>.pth   # checkpoints, as emitted by the trainer
logs/<variant>_seed<N>_<ts>.log   # per-run training logs
```

> **Preserve the artifacts a result depends on.** Text evidence a
> `results/*.md` file cites is committed beside it. Checkpoints and snapshot
> pools are published as GitHub release assets on the tag that identifies the
> result: `scripts/release_artifacts.py publish` bundles them with a
> `SHA256SUMS` manifest (cross-checked against `ARTIFACTS.sha256` when
> present), and `scripts/release_artifacts.py fetch` downloads, verifies, and
> restores them. `results/optimized_selfplay_baseline/` predates this policy,
> so its checkpoints are committed and also published in the
> `v0.3-selfplay-baseline` release. Width-probe checkpoints referenced by
> `results/capacity_probe.md` were lost because they existed only in a deleted
> worktree.

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
- `v0.1` — initial protocol, locked at git tag `baseline/2026-04-22`.
- `v0.1.1` — documentation corrections only; **no result is invalidated**.
  - Cross-play Elo corrected from "5000 games per pairing" to **1000**.
    Every result produced under this protocol
    (`baseline_v3`, `attention_ablation`, `capacity_probe`,
    `token_features_phase1`, `curriculum_v02`) used 1000, so this records
    actual practice rather than changing it.
  - Elo anchor description corrected. The old text claimed `random` was
    fixed at Elo 0; in fact `compute_mle_ratings` pins the **first
    participant** at Elo 1000 and `random` floats (it lands between −41 and
    +35 across existing reports).
  - Dangling `v0.1-baseline` tag reference replaced with the tag that
    actually exists (`baseline/2026-04-22`).
  - Storage layout updated to the checkpoint naming the trainer emits.
- `v0.1.2` — storage policy only; **no result is invalidated**.
  - Checkpoints and snapshot pools are published as verified GitHub release
    bundles on the result's tag instead of being committed.
- `v0.1.3` — adds inference temperature as an explicit variant attribute;
  **no result is invalidated**.
  - Temperature 1 remains the default and describes every prior result.
- `v0.2` — adds two metrics, the **held-out league** and the **purchase
  audit** (§5), defined by decision record 0013.
  - The gauntlet, seat protocol, seed count, sample sizes, and cross-play
    Elo are unchanged, so every earlier result stays valid for the metrics
    it reports. Writeups produced before `v0.2` are tagged `v0.1.x` in
    [`results/README.md`](../results/README.md); they do not report the new
    metrics.
  - The new metrics are measured for the published checkpoints in
    [`results/strength_yardstick.md`](../results/strength_yardstick.md).
