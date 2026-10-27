# 0013 - Strength yardstick: held-out league, purchase audit, and exploiter versus control

**Status:** Accepted on `feat/strength-yardstick` (2026-10-27). Committed
before any data for the measures it defines was generated with this
repository's code. The prototype numbers it cites come from an audit on
2026-09-30, whose scripts and results are archived off-repo under
`D:\space-deck-builder-archive\strength-audit-20260930`.

## Context

Every published strength number compares a checkpoint with the three scripted
gauntlet opponents or with other checkpoints from the same recipe. The
2026-09-30 audit of the published checkpoints, all playing at temperature 0,
found that this cannot say how strong an agent is:

- Heuristic, the only gauntlet opponent that still separates checkpoints, uses
  every scrap ability as soon as it can: it scraps 81% of the Battle Stations
  and 78% of the Dreadnaughts it buys. Six untuned scripted opponents written
  for the audit beat it 74.0% to 84.2% over 2,000 games each.
- Those opponents separate checkpoints that the gauntlet ties. `selfplay_seed0`
  beats them 79.2% over 24,000 games and `fixed_seed0` 66.1% over 12,000, while
  both beat Heuristic about 93%.
- Seed 0's update-400 checkpoint beats its update-200 checkpoint 55.8% (+40
  Elo, 95% CI +25 to +56) and ties update 300. The learning rate had decayed
  to 5.3e-5 by update 300: training stopped because its schedule ended, not
  because the agent stopped improving.
- Full-game rollouts on 957 of `selfplay_seed0`'s purchase decisions found an
  alternative at least three standard errors better on 27 of them (2.8%),
  worth 15 to 18 points of win probability each. Overriding only when the
  rollouts were confident gained 0.37 points per decision (SE 0.09).
- The critic predicts +0.33 in turns 0 to 3 of search self-play games whose
  mean outcome there is -0.01 (correlation 0.24), so the one-turn search,
  which scores leaves with it, gains 0.02 points per purchase decision (SE
  0.14) in each player's first two turns.

None of this says how much an opponent that targets a checkpoint could gain.

## Decision

Three measures join the evaluation toolkit. They are reported alongside the
locked gauntlet and cross-play Elo of `docs/eval_protocol.md`, which do not
change. All checkpoints play at temperature 0 (their highest-scoring legal
move after the production END_TURN suppression) unless stated otherwise.

The **champion** is `results/optimized_selfplay_baseline/checkpoints/selfplay_seed0.pth`
(SHA-256 `9d10dca5110c2bbe9cc0f8693e11578dee347a8c60e7ed63002af7d82e80a503`).

### Held-out league

**Opponents.** Six scripted agents in `src/ai/held_out_agents.py`: balanced,
aggro, economy, scrapper, fortress and blobrush. Their behaviour is that of the
audit prototype (`probe_heldout.py`, SHA-256
`8adf097436d8a4e2447d99066ccc340628c3d486123e01ae48d16880b9f476bb`). They are
held out: they never enter a training opponent pool (`AGENT_REGISTRY`,
`_AGENT_FACTORIES`, the worker registry in `mp_sim_worker.py`, or
`OPPONENT_CODE`), and they are not tuned against any checkpoint.

**Match.** A checkpoint plays each opponent for 4,000 games. Game `index` uses
the deal seed `game_seed(101, index)` from `src/search/uplift.py`, and the
checkpoint moves first on even indices, so it plays exactly 2,000 games in each
seat. Games run through `BatchedGameRunner` with 256 games in flight per
process and policy inference on the GPU; each process plays one opponent's
4,000 games in index order.

**Report.** Per checkpoint: wins out of 4,000 against each opponent with a
Wilson 95% interval, and the pooled wins out of 24,000 with a Wilson interval.
The pooled rate is the league score.

**Port check.** Before any league number is reported, the port must reproduce
the prototype's counts for the champion: 3,084, 3,321, 3,111, 3,055, 3,092 and
3,345 wins against balanced, aggro, economy, scrapper, fortress and blobrush.
Each opponent must also reproduce its counts against Heuristic over 2,000 games
with deal seed 103, the opponent moving first on even indices: 1,683, 1,533,
1,589, 1,664, 1,670 and 1,479. A difference is explained before league numbers
are reported. If it comes from arithmetic rather than behaviour, such as two
nearly equal logits ordered differently, the explanation is written up and the
port's numbers stand.

### Purchase audit

**Population.** For a checkpoint, games of greedy self-play (both seats are the
checkpoint) on deals `game_seed(seed, index)`, with seat A moving first on even
indices. A decision is eligible when the acting player has no pending choice,
at least one purchase is legal, the checkpoint's greedy move is a purchase or
END_TURN, and there are at least two candidates. The **candidates** are every
legal purchase, END_TURN when it is legal after the production suppression,
and the one-turn search's choice (`TurnSearch` at budget 64, 8 candidates,
search seed 0) when it is not already among them.

**Sampling.** One decision per game, drawn uniformly among that game's eligible
decisions of both seats. A first pass plays each game to the end and counts its
eligible decisions; the target is drawn from a generator seeded by the master
seed and game index; a second pass replays the game and stops at that decision.
Games are taken in index order. A game with no eligible decision, or whose
replay ends or diverges before the target, is skipped and counted. An audit
uses the first 1,000 games that yield a decision.

**Rollouts.** Each candidate gets 256 rollouts. A rollout applies the
candidate, then plays the game to the end in the vector engine with both seats
choosing the checkpoint's greedy move. The root is the acting player's
information set (`SearchRootBuilder`); rollout `r` re-deals the opponent's
hidden hand and seeds the engine with rollout id `r` (`deal_world_hands`), and
every candidate of a decision uses the same rollout ids. Rollouts 0 to 127 are
half A and 128 to 255 half B.

**Measures**, in points of win probability:

- **Confident override** (headline). On one half, let `c` be the candidate
  with the most wins, ties going to the earlier candidate in the order above.
  If `c` is not the checkpoint's move `g` and its win rate leads by at least
  `2 * sqrt(p_c(1 - p_c)/128 + p_g(1 - p_g)/128)`, the gain is `c`'s win rate
  minus `g`'s on the other half; otherwise it is 0. The decision's gain is the
  mean over the two directions. The audit reports the mean over decisions and
  its standard error.
- **Clear mistakes.** With all 256 rollouts, an alternative is clearly better
  when its mean paired difference from `g` is at least three paired standard
  errors (standard deviation of the per-rollout difference over the square
  root of 256). Report the share of decisions with one and the mean gain of
  the best such alternative.
- **Search's choice.** The one-turn search's candidate's win rate minus `g`'s
  over all 256 rollouts.

Each is also reported by turn counter (both players' turns, counted from 0):
0 to 3, 4 to 9, 10 to 15, and 16 or more.

**Runs.** `selfplay_seed0`, `selfplay_seed1` and `selfplay_seed2`, with master
seed 20260930 and 1,000 decisions each. Others are optional.

### Exploiter versus control

**Question.** Can an agent trained only against the frozen champion beat it by
clearly more than the same amount of ordinary training does?

**Arms.** Both start from the champion's weights with a fresh optimizer and
train 400 updates of 2,000 episodes on the vector engine with the reference
recipe's PPO settings, cosine learning-rate schedule (3e-4 to 1e-5 over the
400 updates), 25-update fresh-process chunks and evaluation cadence, through
`scripts/train_reference.ps1`.

- **Exploiter** (preset `exploit-selfplay-seed0`): every training game is
  against the frozen champion choosing its highest-scoring legal move
  (`--frozen-opponents` with ratio 1 and temperature 0). It adds no self-play
  snapshots.
- **Control** (preset `continue-selfplay-seed0`): the `reference-selfplay`
  recipe unchanged: half the games against Random, Heuristic and Simple, half
  against its own snapshots by variance PFSP.

Each arm runs seeds 0, 1 and 2, one run at a time. The control arm runs first,
from the commit that adds this record and its preset. The exploiter arm runs
from a later commit that adds the frozen-opponent option; the option leaves
the control's code path unchanged when its flags are absent, which its tests
check. The write-up records both commits.

**Scoring.** For each training seed `s`, one `elo` tournament at temperature 0
with exact seat alternation and 4,000 games per pairing among the champion,
`selfplay_seed1`, `selfplay_seed2`, `exploiter_seed<s>` and `control_seed<s>`.

- `E_s`: the exploiter's win rate against the champion.
- `C_s`: the control's win rate against the champion.
- `O_s`, for each arm: the mean of its win rates against `selfplay_seed1` and
  `selfplay_seed2`.
- `L_s`, for each arm: its league score.

**Reading rule.** Every quantity is computed per seed. A mean and its interval
are the mean of the three per-seed values and its Student-t 95% interval with
two degrees of freedom (`mean +/- 4.303 * sd / sqrt(3)`). For `E - C` the
three values are the per-seed differences `E_s - C_s`.

- **Exploitable**: the mean of `E - C` is at least 5 points and its interval
  excludes zero.
  - **Narrow**, a weakness of this champion: the exploiter is not the better
    all-round player; its mean `L` and its mean `O` are both no higher than
    the control's.
  - **General**: otherwise. Training against one frozen opponent produced a
    better all-round agent; recorded as a finding about training, not as a
    weakness.
- **Not shown exploitable**: anything else. Only then does the fallback run.
- Separately, for the next phase: a mean `C` of at least 55% with its interval
  above 50% is evidence that more training alone strengthens the champion.

Five points is about 35 Elo near 50%. With 4,000 games per pairing the
standard error of one seed's `E_s - C_s` is about 1.1 points.

**Diagnostics, outside the rule:** the exploiter's rollout win rate against the
frozen champion per update (`rollout.per_opponent` in the training metrics),
each arm's final gauntlet, and each arm's full tournament ratings.

**Fallback**, only if the champion is not shown exploitable, and only after its
own decision record: an exploiter from random weights trained for 400 updates
against a ladder of frozen opponents (seed 0's archived checkpoints at updates
25, 50, 100, 200 and 300, and the champion) on seeds 0, 1 and 2, each scored
against the champion over 4,000 games at temperature 0.

**Limit.** An exploiter gives a lower bound. If it wins, the champion is
exploitable; if it fails, only this adversary found nothing.

## Rejected alternatives

**Training against the held-out opponents, or tuning them.** Either makes them
part of what the agents are fitted to and ends their use as a held-out measure.

**Exploiter from random weights first.** The champion won 341 of 342 training
games against Random, so an exploiter from random weights would win very few
early games and learn slowly. It remains the fallback.

**Scoring purchases with the critic or the one-turn search.** The critic is
uninformative early in the game, and the one-turn search scores its leaves
with it. Full-game rollouts measure outcomes.

**Replacing the locked gauntlet.** Earlier results stay comparable; the new
measures are added.

## Consequences

- `docs/eval_protocol.md` gains the league and the audit in a new protocol
  version when their results are published, as its stability clause requires.
- The league, the audit and the frozen-opponent option are new code on this
  branch; they are reviewed before merge.

## Amendments

**2026-10-27, run order.** Made before any score, league or audit result
existed. The two arms train side by side instead of one run at a time. A single
training run left the GPU about 40% idle; with both arms training, each kept
close to its speed alone (about 10 to 13 seconds per update, against about 10
alone). Training does not depend on wall-clock time, so this changes only when
results arrive. The control arm still started first, from `c2e1a89`, and the
exploiter arm runs from `a792c96`. The league and the audit run from the
commit that adds this amendment. Before it, the code review found that the
audit's first implementation left out decisions with a single legal purchase
on which the one-turn search chooses something else; the text above counts
them as eligible and is unchanged, and the code now follows it.

**2026-10-27, run order, continued.** Also made before any score, league or
audit result existed. Exploiter seed 2 trains in a third lane alongside the
other two. The league and the audits run after all training has finished.
While the league ran beside the trainings, their evaluation workers stalled
for minutes, and one interim evaluation timed out and ended its chunk
(exploiter seed 0, updates 100 to 125). That chunk was rerun from its starting
checkpoint, which restores the random-number and optimizer state.

**2026-10-27, audits, continued.** Made before any audit result was computed
or looked at. The audit of `selfplay_seed0` started beside training at 03:09
(from `2784f95`, at below-normal priority) and was stopped at 03:23, after 305
of its 1,000 decisions were rolled out and journaled, when its memory and one
ordinary training evaluation together brought committed memory past 70 GB of
the machine's 73 GB limit. The audits finish after all training, resuming
from that journal; the audit's code, configuration and journal identity are
unchanged.
