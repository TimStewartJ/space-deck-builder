# Strength yardstick: held-out league, purchase audit, exploiter versus control

**Status:** Measures complete (2026-10-27). Decision 0013's fallback, an
exploiter trained from random weights against a ladder of the champion's
earlier checkpoints ([decision 0014](../docs/decisions/0014-ladder-exploiter.md)),
is running; its result will be added here. **Pre-registered outcome so far:
not shown exploitable** (mean E - C +4.5 points against a 5-point threshold),
and **further training narrowly misses the "helps" bar** (mean C 54.8%
against 55%).
**Design:** [`docs/decisions/0013-strength-yardstick.md`](../docs/decisions/0013-strength-yardstick.md),
committed before any data was generated (`c2e1a89`). Its dated amendments
change only when things ran; each was made before the results it concerns
existed or had been looked at.
**Evidence:** [`strength_yardstick/`](strength_yardstick/), produced by
[`run_arms.ps1`](strength_yardstick/run_arms.ps1) and
[`run_measures.ps1`](strength_yardstick/run_measures.ps1), checked by
[`check_port.py`](strength_yardstick/check_port.py) and summarized by
[`summarize.py`](strength_yardstick/summarize.py)
([`summary.md`](strength_yardstick/summary.md)).
**Protocol:** [`docs/eval_protocol.md`](../docs/eval_protocol.md) v0.2.

## Results in brief

- **No exploit of the champion was found by this adversary.** 400 updates of
  training against the frozen champion beat it 59.3% of the time; 400
  updates of the ordinary recipe beat it 54.8%. The difference, +4.5 points
  (95% CI -1.0 to +9.9), misses the pre-registered 5-point threshold, so the
  champion is not shown exploitable and 0013's fallback adversary runs next.
- **Both arms are stronger than the champion, and the exploiters are the
  strongest checkpoints measured.** Every arm checkpoint beats the champion in
  its seed's tournament and outscores it against held-out opponents
  (exploiters 83.5%, controls 82.0%, champion 79.2%). The exploiter beats the
  control in every seed. So the published champion was not at its ceiling,
  and in these runs training against a strong fixed opponent improved it more
  than the reference recipe's opponent mix did.
- **The held-out league works.** It reproduces the prototype exactly and
  separates checkpoints the gauntlet tied (fixed-opponent seeds about 65.5%,
  self-play seeds 73.9% to 79.2%).
- **The published self-play checkpoints' purchases give away 0.27 to 0.58
  points of win probability per decision** by full-game rollouts, the most
  per decision early in the game for two of the three seeds. The controls
  give away about as much as the champion overall, and suggestively less in
  the opening.
- **Exploratory: held-out strength was still rising when training stopped.**
  Along the champion's own run the league rose 1.9 points from update 300 to
  400, while those two checkpoints tied head to head.
- **Side finding:** the engine gives both options of every OR ability and
  differs from the rulebook in two smaller ways; every agent here learned that
  game.

## Question

The 2026-09-30 audit found that Heuristic, the strongest opponent in the
locked gauntlet, no longer separates the published checkpoints, while scripted
opponents they never trained against do. At temperature 0, `selfplay_seed0`
beat Heuristic in 93.7% of 4,000 games and `fixed_seed0` in 93.2% of 2,000.
Against the same six held-out opponents they won 79.2% (24,000 games) and
66.1% (12,000 games). The prototype results are archived in
`D:\space-deck-builder-archive\strength-audit-20260930\heldout_full.json`.
Three measures are added to the yardstick for strength:

1. **Held-out league.** How often does a checkpoint beat six scripted opponents
   that no agent has trained against?
2. **Purchase audit.** How much win probability do a checkpoint's purchases
   give away, judged by complete rollouts?
3. **Exploiter versus control.** Can 400 updates of training against the
   frozen champion find a weakness that the same amount of ordinary training
   does not?

## Method

Everything ran through Task Scheduler on the RX 9070 with torch
2.11.0+rocm7.13.0a20260408, from worktrees pinned to the commits below.

| Part | Commit | Notes |
|---|---|---|
| Control arm | `c2e1a89` | preset `continue-selfplay-seed0`; this lane also scored seed 2 |
| Exploiter arm | `a792c96` | preset `exploit-selfplay-seed0`; this lane scored seeds 0 and 1 |
| League, port check and audits | `2784f95` | |

The tournament code is the same at both scoring commits. Of what it imports,
only `src/config.py` changed between them, gaining optional frozen-opponent
fields whose defaults leave every existing setting as it was.

- **Champion:** `selfplay_seed0` (SHA-256 `9d10dca5…a503`), from
  [`optimized_selfplay_baseline`](optimized_selfplay_baseline.md), at
  temperature 0.
- **Arms.** 400 updates of 2,000 episodes from the champion's weights with a
  fresh optimizer, a cosine learning rate from 3e-4 to 1e-5, and fresh
  processes every 25 updates; three seeds each.
  - The exploiter plays every training game against the frozen champion
    choosing its highest-scoring legal move.
  - The control is the reference self-play recipe unchanged: half its games
    against Random, Heuristic and Simple, half against its own snapshots.
- **Scoring.** For each seed, an `elo` tournament at temperature 0 with exact
  seat alternation and 4,000 games per pairing among the champion,
  `selfplay_seed1`, `selfplay_seed2`, `exploiter_seed<s>` and
  `control_seed<s>`.
- **League.** Each checkpoint plays each of the six held-out opponents
  (`src/ai/held_out_agents.py`) for 4,000 games at temperature 0. Deals are
  `game_seed(101, index)` and the checkpoint moves first on even indices.
  Calibration plays each opponent against Heuristic for 2,000 games. The
  checkpoint matchups ran with six worker processes; the calibration had
  finished in about a minute with twelve. The counts do not depend on the
  number of workers, since each game's deal and seats are fixed by its index;
  the port check matched a prototype that used eight. After training, an
  attempt with twelve workers finished no 4,000-game matchup in 30 minutes.
  On this machine, throughput flattens at about six workers: 1, 2, 4 and 6
  workers played 3,364, 5,455, 8,472 and 9,333 games a minute
  ([`scale/`](strength_yardstick/scale/)), and process CPU counters during
  the exploratory league showed six workers using about 2.9 cores each,
  about 19 of the 20 logical CPUs together with the parent. Why twelve
  stalled rather than merely flattened is not established.
- **Audit.** 1,000 purchase decisions per checkpoint, each sampled from its own
  greedy self-play game. Every candidate gets 256 complete greedy rollouts
  from the acting player's information set.

## Exploiter versus control

Each seed's tournament played 4,000 games per pairing at temperature 0 with
exact seat alternation ([`scores/`](strength_yardstick/scores/)).

| Seed | E: exploiter vs champion | C: control vs champion | E - C (points) | Exploiter vs control | O: exploiter vs seeds 1, 2 | O: control vs seeds 1, 2 | L: exploiter league | L: control league |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 59.3% | 55.5% | +3.9 | 52.2% | 61.6% | 57.2% | 84.0% | 81.8% |
| 1 | 60.3% | 53.4% | +6.9 | 53.9% | 61.7% | 57.6% | 83.7% | 81.6% |
| 2 | 58.2% | 55.5% | +2.6 | 53.2% | 59.6% | 57.5% | 82.9% | 82.7% |
| Mean (t95) | 59.3% (56.7 to 61.8) | 54.8% (51.7 to 57.8) | +4.5 (-1.0 to +9.9) | | 60.9% (58.0 to 63.9) | 57.4% (56.8 to 58.0) | 83.5% (82.1 to 84.9) | 82.0% (80.6 to 83.4) |

Exact wins of 4,000: E 2,373, 2,411 and 2,328; C 2,218, 2,135 and 2,222.

**Pre-registered verdict.**

- **Not shown exploitable.** The rule needs a mean E - C of at least 5 points
  with an interval that excludes zero; the mean is +4.5 points and the
  interval runs from -1.0 to +9.9. So "narrow or general" does not apply, and
  0013's fallback runs (decision 0014).
- **More training does not pass the bar, narrowly.** The rule needs a mean C
  of at least 55% with its interval above 50%; the mean is 54.8%, with its
  interval (51.7% to 57.8%) above 50%.

**What the data show beyond the verdict.** Both arms beat the champion in
every seed: each of the six win rates is at least 53.4% over 4,000 games, more
than four standard errors above 50%. The exploiter beat the control head to
head in every seed (52.2% to 53.9%). Its gain also reached opponents it never
trained against: it beat the two other published self-play checkpoints more
often than the control did (60.9% against 57.4%), and it scored higher in the
held-out league in each seed, by 2.2, 2.1 and 0.2 points (83.5% against 82.0%
on average; the champion scores 79.2%). In these runs, training only against
the frozen champion produced a generally stronger checkpoint than the
reference recipe, which spends half its games against Random, Heuristic and
Simple.

The exploiter's training games against the frozen champion (the learner
sampling, the champion at temperature 0) went from 45% wins at update 1 to
53% to 54% over the last 25 updates in every seed. They rose over the last
100 updates (the mean over updates 351 to 400 was 0.7 to 0.9 points above
the mean over 301 to 350) but show no further rise over the final 50.

**Diagnostics outside the rule.** The locked final gauntlet (sampled play,
15,000 games against Random, Heuristic and Simple) gave the controls 14,421,
14,332 and 14,409 wins and the exploiters 14,559, 14,503 and 14,501. Each
seed's tournament ratings, anchored at that seed's control = 1000:

| Seed | Exploiter | Control | `selfplay_seed0` (champion) | `selfplay_seed1` | `selfplay_seed2` |
|---:|---:|---:|---:|---:|---:|
| 0 | 1024 | 1000 | 961 | 932 | 959 |
| 1 | 1032 | 1000 | 967 | 935 | 963 |
| 2 | 1019 | 1000 | 963 | 933 | 964 |

**What changed in play (exploratory).** In 400 greedy games against the
champion per checkpoint ([`purchase_profiles/`](strength_yardstick/purchase_profiles/),
[`purchase_profile.py`](strength_yardstick/purchase_profile.py)), all six arm
checkpoints buy fewer Imperial Fighters than the champion (0.37 to 0.50 fewer
per game), fewer Blob Wheels, and more Federation Shuttles (0.17 to 0.30 more)
and Trade Bots (0.12 to 0.21 more), and they scrap more cards (0.31 to 0.70
more per game). Since the controls show the same shifts, they come from
further training rather than from training against the champion. Where the
arms differ, the exploiters buy fewer Blob Fighters than the controls (0.12
to 0.19 fewer than the champion, against 0.01 to 0.09) and drop fewer Blob
Wheels (0.06 to 0.11 fewer than the champion, against 0.23 to 0.29). With 400
games per profile, differences below about 0.1 per game are within noise.

## Held-out league

Every checkpoint played each held-out opponent 4,000 times at temperature 0
([`league/league.json`](strength_yardstick/league/league.json)). Pooled win
rates over the 24,000 games, with Wilson 95% intervals:

| Checkpoint | balanced | aggro | economy | scrapper | fortress | blobrush | Pooled (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `exploiter_seed0` | 81.8% | 86.5% | 82.1% | 82.0% | 83.7% | 87.9% | **84.0%** (83.5 to 84.4) |
| `exploiter_seed1` | 81.5% | 87.1% | 82.7% | 81.8% | 81.8% | 87.4% | **83.7%** (83.2 to 84.2) |
| `exploiter_seed2` | 80.9% | 85.5% | 82.5% | 80.2% | 82.2% | 86.1% | **82.9%** (82.4 to 83.4) |
| `control_seed2` | 80.4% | 85.7% | 81.2% | 81.1% | 80.7% | 86.8% | **82.7%** (82.2 to 83.1) |
| `control_seed0` | 79.8% | 84.4% | 80.3% | 80.0% | 80.7% | 85.4% | **81.8%** (81.3 to 82.3) |
| `control_seed1` | 79.5% | 84.5% | 79.4% | 79.8% | 80.6% | 85.9% | **81.6%** (81.1 to 82.1) |
| `distilled_seed0` | 78.4% | 82.7% | 77.8% | 77.8% | 78.0% | 83.4% | **79.7%** (79.2 to 80.2) |
| `selfplay_seed0` (champion) | 77.1% | 83.0% | 77.8% | 76.4% | 77.3% | 83.6% | **79.2%** (78.7 to 79.7) |
| `selfplay_seed2` | 76.4% | 82.3% | 76.4% | 77.2% | 75.4% | 83.0% | **78.5%** (77.9 to 79.0) |
| `selfplay_seed1` | 72.2% | 77.8% | 71.2% | 73.0% | 70.9% | 78.5% | **73.9%** (73.4 to 74.5) |
| `fixed_seed2` | 64.2% | 73.0% | 60.8% | 66.8% | 55.9% | 73.1% | **65.6%** (65.0 to 66.2) |
| `fixed_seed0` | 64.2% | 72.5% | 61.0% | 65.1% | 57.6% | 73.3% | **65.6%** (65.0 to 66.2) |
| `fixed_seed1` | 64.1% | 72.7% | 60.7% | 65.9% | 55.3% | 73.2% | **65.3%** (64.7 to 65.9) |
| *Opponent vs Heuristic (2,000 games)* | 84.2% | 76.6% | 79.5% | 83.2% | 83.5% | 74.0% | |

- **The port reproduces the prototype exactly.** All six of the champion's
  counts and all six calibration counts against Heuristic match the
  2026-09-30 prototype ([`league/port_check.json`](strength_yardstick/league/port_check.json)),
  so the prototype's numbers and these are the same measurement.
- **The league separates what the gauntlet could not.** The three
  fixed-opponent checkpoints score 65.3% to 65.6% and the three self-play
  checkpoints 73.9% to 79.2%, while on the locked gauntlet (sampled play) the
  self-play seeds scored 94.40% and the fixed-opponent seeds 94.71% overall
  ([`optimized_selfplay_baseline.md`](optimized_selfplay_baseline.md)).
  Every held-out opponent beats Heuristic 74% to 84% of the time, so all six
  are stronger than the gauntlet's best scripted opponent.
- **Both arms' gains transfer to opponents nobody trained on.** All six arm
  checkpoints score above the champion's 79.2%: the controls 81.6% to 82.7%,
  the exploiters 82.9% to 84.0%.
- **The distilled checkpoint's gain does not transfer.** It scores 79.7%
  against its source's 79.2%, inside the noise, as the prototype found.

## Purchase audit

Every audit sampled its 1,000 decisions from the first 1,000 games, with no
game skipped, no replay divergence and no search mismatch
([`audit/`](strength_yardstick/audit/)). One rollout of the 992,768 in
`control_seed0`'s audit was still running after 20,000 decisions and counted
as a loss for the acting player. It belongs to a decision where the greedy
purchase won 27 more of its 256 rollouts than any alternative, so it changes
nothing. Every other rollout finished.

| Checkpoint | Override gain, points (SE) | Decisions with an override | Clear mistakes | Their mean gain, points | One-turn search gain, points (SE) | Search differs from greedy |
|---|---:|---:|---:|---:|---:|---:|
| `selfplay_seed0` (champion) | +0.27 (0.06) | 6.1% | 19 (1.9%) | +14.4 | +0.08 (0.04) | 8.2% |
| `selfplay_seed1` | +0.58 (0.10) | 7.6% | 37 (3.7%) | +17.4 | +0.20 (0.07) | 8.6% |
| `selfplay_seed2` | +0.40 (0.10) | 6.7% | 22 (2.2%) | +18.9 | +0.25 (0.10) | 8.6% |
| `control_seed0` (optional) | +0.23 (0.06) | 5.9% | 16 (1.6%) | +14.9 | +0.20 (0.06) | 10.8% |
| `control_seed1` (optional) | +0.19 (0.06) | 5.3% | 13 (1.3%) | +15.8 | +0.10 (0.07) | 11.6% |
| `control_seed2` (optional) | +0.37 (0.12) | 4.6% | 20 (2.0%) | +19.7 | +0.15 (0.12) | 8.8% |

Override gain by turn counter (both players' turns, from 0), points (SE):

| Checkpoint | Turns 0-3 | 4-9 | 10-15 | 16+ |
|---|---:|---:|---:|---:|
| `selfplay_seed0` | +0.64 (0.23) | +0.24 (0.09) | +0.37 (0.14) | -0.02 (0.03) |
| `selfplay_seed1` | +1.30 (0.42) | +0.83 (0.22) | +0.33 (0.15) | +0.26 (0.13) |
| `selfplay_seed2` | +0.39 (0.16) | +0.37 (0.13) | +0.27 (0.14) | +0.55 (0.29) |
| `control_seed0` | +0.34 (0.13) | +0.29 (0.16) | +0.18 (0.08) | +0.16 (0.10) |
| `control_seed1` | +0.23 (0.15) | +0.22 (0.12) | +0.23 (0.13) | +0.11 (0.08) |
| `control_seed2` | +0.13 (0.12) | +0.22 (0.09) | +0.22 (0.09) | +0.77 (0.39) |

- **Purchases give away a little, measurably.** Across the three self-play
  seeds a confident full-game rollout override recovers 0.27 to 0.58 points
  of win probability per purchase decision, all clearly above zero. Between
  1.9% and 3.7% of purchases are clear mistakes, worth 14 to 19 points each
  on average.
- **Among the three self-play seeds, the champion gives away the least and is
  at least as strong as the others.** In the three tournaments here it beats
  `selfplay_seed1` 54.0% to 54.6% of the time and ties `selfplay_seed2`
  (50.0% to 50.7%), and it leads them in the league (79.2% against 78.5% and
  73.9%).
- **Further training did not measurably reduce the give-away.** The three
  controls, 400 updates past the champion and 2.8 points stronger than it in
  the league, give away +0.19 to +0.37 points per decision (mean +0.26)
  against the champion's +0.27, with 1.3% to 2.0% clear mistakes against its
  1.9%. Their give-away in turns 0 to 3 is smaller (+0.13 to +0.34 against
  +0.64), but with standard errors of 0.12 to 0.23 that difference is only
  suggestive.
- **Early purchases matter most for two of the three self-play seeds.** For
  `selfplay_seed0` and `selfplay_seed1` the gain is largest in turns 0 to 3;
  `selfplay_seed2` is flat. The 2026-09-30 prototype found the same early
  concentration on the champion.
- **The one-turn search recovers part of it.** Where it disagrees with the
  greedy move (about 8.5% of the self-play seeds' decisions) its choice gains
  1.0, 2.4 and 2.9 points on average, by the same rollouts.

## Exploratory measurements

None of these was pre-registered. They ran after the pre-registered
measures, with the same code (`2784f95`), and cannot change the verdict.

### One tournament of all nine checkpoints

The pre-registered tournaments compare each seed's arms only with each other
and the published self-play checkpoints. One more temperature-0 tournament of
all nine, 2,000 games per pairing with exact seat alternation, compares them
across seeds ([`explore/elo_all_T0.json`](strength_yardstick/explore/elo_all_T0.json)).
Ratings are anchored at `control_seed0` = 1000.

| Checkpoint | Elo | Wins of its 16,000 games |
|---|---:|---:|
| `exploiter_seed0` | 1029 | 56.1% |
| `exploiter_seed1` | 1024 | 55.2% |
| `exploiter_seed2` | 1015 | 53.8% |
| `control_seed2` | 1004 | 52.1% |
| `control_seed1` | 1000 | 51.5% |
| `control_seed0` | 1000 | 51.4% |
| `selfplay_seed2` | 960 | 45.0% |
| `selfplay_seed0` (champion) | 957 | 44.5% |
| `selfplay_seed1` | 931 | 40.5% |

- **The groups rank as in the league:** exploiters, then controls, then the
  published self-play checkpoints, with `selfplay_seed1` last.
- **Every exploiter beats every control but one.** Over the nine cross-seed
  pairings the exploiters won 53.0% of 18,000 games (Wilson 95% interval
  52.2% to 53.7%); the one exception is `exploiter_seed2` against
  `control_seed2` (49.7%).
- **The controls' margin over the champion is near the 55% bar.** On these
  deals the controls beat the champion 56.0% to 56.5% of the time, against
  53.4% to 55.5% in the pre-registered tournaments. So the pre-registered
  reading (mean C just short of the 55% "helps" bar) is a narrow miss that a
  different sample could have reversed; the verdict stands as pre-registered.

### The league along the training curve

The same league for the archived checkpoints of the champion's own training
run (the original seed-0 reference run) at updates 50, 100, 200 and 300, and
for `control_seed0` halfway through its 400 updates
([`explore/league_curve.json`](strength_yardstick/explore/league_curve.json),
identical settings and code; the update-400 rows come from the main league).

| Checkpoint | Pooled league win rate (95% CI) |
|---|---:|
| Reference run, update 50 | 59.0% (58.4 to 59.7) |
| Reference run, update 100 | 71.1% (70.5 to 71.6) |
| Reference run, update 200 | 75.8% (75.3 to 76.3) |
| Reference run, update 300 | 77.3% (76.8 to 77.9) |
| Reference run, update 400 (the champion) | 79.2% (78.7 to 79.7) |
| `control_seed0`, 200 updates past the champion | 80.2% (79.7 to 80.7) |
| `control_seed0`, 400 updates past the champion | 81.8% (81.3 to 82.3) |

- **Held-out strength had not levelled off when training stopped.** It rose
  1.5 points from update 200 to 300 and 1.9 points from 300 to 400. Update
  400 nevertheless tied update 300 head to head (49.6% in the 2026-09-30
  tournament of this run's checkpoints), so the ratings among one run's own
  checkpoints understated its late gains.
- **Continued training kept gaining at a slower rate,** 1.0 and then 1.6
  points per 200 updates. This is one seed; phase 2 of the roadmap tests
  longer schedules properly.

### Purchase audits of the exploiters

The same audit, with the same code and settings, for the three exploiters
([`explore/`](strength_yardstick/explore/)): 1,000 decisions each, with no
game skipped and no divergence or mismatch. One rollout of the 988,928 in
`exploiter_seed1`'s audit was still running after 20,000 decisions and
counted as a loss; as with `control_seed0`'s, counting it either way changes
no override or clear-mistake call.

| Checkpoint | Override gain, points (SE) | Decisions with an override | Clear mistakes | Their mean gain, points | One-turn search gain, points (SE) | Search differs from greedy |
|---|---:|---:|---:|---:|---:|---:|
| `exploiter_seed0` | +0.19 (0.06) | 4.7% | 14 (1.4%) | +16.5 | +0.17 (0.07) | 9.0% |
| `exploiter_seed1` | +0.12 (0.04) | 3.8% | 10 (1.0%) | +13.6 | +0.12 (0.05) | 7.2% |
| `exploiter_seed2` | +0.26 (0.07) | 5.3% | 20 (2.0%) | +16.1 | +0.19 (0.06) | 8.7% |

| Checkpoint | Turns 0-3 | 4-9 | 10-15 | 16+ |
|---|---:|---:|---:|---:|
| `exploiter_seed0` | +0.19 (0.14) | +0.10 (0.09) | +0.17 (0.09) | +0.30 (0.17) |
| `exploiter_seed1` | -0.00 (0.04) | +0.07 (0.08) | +0.08 (0.04) | +0.26 (0.12) |
| `exploiter_seed2` | +0.15 (0.11) | +0.21 (0.12) | +0.33 (0.16) | +0.34 (0.15) |

- **Across the nine audited checkpoints, the give-away falls as league
  strength rises.** Mean override gain: published self-play seeds +0.41,
  controls +0.26, exploiters +0.19 points per decision; the correlation with
  the pooled league win rate is -0.85 (Spearman -0.70). It rests mostly on
  the two weaker self-play seeds: without `selfplay_seed1` it is -0.60
  (Spearman -0.57), and over the champion and the six arm checkpoints only
  -0.31. Within groups the order differs: among the published self-play
  seeds the give-away falls as league strength rises, among the controls it
  rises, and among the exploiters it is mixed. The exploiters' +0.19 against
  the champion's +0.27 is about one standard error per checkpoint. The audit
  separates weak from strong checkpoints better than it separates close ones.
- **The early give-away shrinks most.** Correlations with league strength by
  turn counter: turns 0 to 3, -0.94 (Spearman -0.85); turns 4 to 9, -0.93
  (-0.90); turns 10 to 15, -0.63 (-0.73); turns 16 and later, +0.12
  (+0.30). In turns 0 to 3 the champion gives away +0.64 (SE 0.23), its
  controls +0.13 to +0.34 and its exploiters -0.00 to +0.19. All six arm
  checkpoints give away less there than the champion (mean +0.17; z about
  2.0 against the champion's single estimate, 1.7 for the controls alone).
  So further training may already recover much of the opening give-away that
  full-game rollouts recovered for the champion, the largest opportunity in
  the 2026-09-30 audit; this is suggestive, not established.

## Deviations from the plan

- **Run order.** Decision 0013 said each arm would run one seed at a time.
  Instead, the arms trained side by side in three lanes: the controls,
  exploiter seeds 0 and 1, and exploiter seed 2. A single training run left
  the GPU about 40% idle. Nothing in training depends on wall-clock time. Both
  amendments to 0013 were committed before any score, audit or checkpoint
  league result existed.
- **One chunk rerun.** The held-out league first ran beside the trainings.
  While it did, their chunk-end evaluations went from about 20 seconds to 5
  to 14 minutes, and one timed out (exploiter seed 0, update 125), ending its
  chunk. That chunk, updates 100 to 125, was rerun from its starting
  checkpoint, which restores the random-number and optimizer state. The
  league then moved after all training.
- **Audits started early and resumed.** The audit of `selfplay_seed0` ran
  beside training from 03:09 to 03:23 and was stopped, after 305 decisions
  were journaled, when committed memory passed 70 GB of 73. From 04:15 the
  audits ran one at a time beside the last training lane: `selfplay_seed1`,
  then `selfplay_seed0` resuming from its journal, then `selfplay_seed2`, with
  the same code and configuration throughout.
- **Audit population corrected before any audit ran.** The code review found
  that the first implementation (`a69b010`) left out decisions with a single
  legal purchase on which the one-turn search chooses something else. 0013
  counts those as eligible. `7d872d1` makes the code follow the record; on 40
  champion games it adds the 3 decisions the review found missing.

## Side finding: the engine's rules differ from the rulebook

While checking the Machine Base deviation for phase 2, three deviations from
the official rulebook (Star Realms Box Set 1.1E, pages 4 and 5) turned up.
[`rules_probe.py`](strength_yardstick/rules_probe.py) counts how often the
champion uses them in 400 greedy self-play games
([`rules_probe_selfplay_seed0.json`](strength_yardstick/rules_probe_selfplay_seed0.json)).

| Rulebook | Engine | Champion's use (400 games) |
|---|---|---|
| An OR ability gives one of its options. | It gives both: the card's effect action applies the first option and then, used again, the remaining one. 10 of the 80 trade-deck cards have an OR ability. | An OR ability was used in 377 games, and in every one of them both options took effect at least once; 2,998 of 3,117 first options (96.2%) were followed by a second option that changed something (3,108 applied it). |
| A ship's primary ability happens immediately as it is played. | Choices in it (for example Trade Bot's "may scrap") can be made at any time in the main phase. | 731 of 4,169 ship choices were made later in the turn. |
| A base's primary ability is used when the player chooses, and resolves in full. | On the turn Machine Base is played, it draws at once and its scrap is optional. | Its scrap was skipped on 9 of 100 plays. |

The OR deviation changes the game the most: Patrol Mech, for example, gives
3 Trade and 5 Combat instead of one of them. Because the agents and the
held-out opponents all play under these rules, the comparisons in this
write-up stand, but no result in this repository measures the published
game. The fix is planned for phase 2 of the strength roadmap; a faithful
OR choice needs a change to the action space.

## Limits

- **Rules.** Every number here uses the engine's current rules (the engine is
  unchanged on this branch), which differ from the official rulebook in three
  ways found on 2026-10-27. OR
  abilities give both options (10 of 80 trade-deck cards). Ship primary-ability
  choices can wait until later in the turn. Machine Base's scrap is optional on
  the turn it is played. All agents and opponents play under the same rules,
  so the comparisons stand; they are not measurements of the published game.
  The fixes are planned for phase 2.
- **Three seeds.** Intervals over three seeds use a t quantile of 4.3, so they
  are wide. Both pre-registered thresholds were missed narrowly (E - C +4.5
  against 5 points; C 54.8% against 55%); the verdicts are as the rule reads
  them, not evidence that the effects are absent.
- **An exploiter is a lower bound.** It started from the champion's own
  weights and trained for 400 updates; another adversary might find more.
  This outcome is the condition for 0013's fallback, an exploiter from random
  weights trained against a ladder of the champion's earlier checkpoints
  ([decision 0014](../docs/decisions/0014-ladder-exploiter.md)), which is
  running.
- **The held-out opponents are scripted and untuned,** and were designed by
  the agent that ran the 2026-09-30 audit. They are stronger than Heuristic
  but say nothing about strength against people.
- **The audit judges each purchase by the checkpoint's own continuation.**
  Rollouts play both seats greedily with the audited policy, so a purchase is
  scored by how that policy goes on to play, not by optimal play.
- **Exploratory parts.** The purchase profiles, the cross-seed tournament, the
  league along the training curve and the exploiters' audits were not
  pre-registered.
