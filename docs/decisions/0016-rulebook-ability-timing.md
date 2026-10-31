# 0016 - Resolve abilities when the rulebook says: one OR option, ship choices on play, Machine Base in one activation

**Status:** Accepted on `feat/rules-fidelity` (2026-10-30). Supersedes the OR
behaviour of ADR 0004. Introduces game rules version 2.

## Context

The Star Realms Box Set rulebook 1.1E (pages 4 and 5) says when abilities
resolve:

- "When a ship is played, its Primary Ability happens immediately."
- "Also unlike Ships, you may choose when during your Main Phase to use your
  bases' Primary Abilities."
- "With the exception of a Ship's Primary Ability (which happens immediately
  as the Ship is played), players may use the various abilities of their
  in-play Ships and Bases at any time during their Main Phase. Each ability
  may be used just once each turn."
- "Other abilities allow you to choose from multiple effects (for example;
  Trading Post provides either 1 [Authority] or 1 [Trade])."

While building the strength yardstick, three deviations from these rules
turned up (`results/strength_yardstick.md`, side finding). Both engines had
them, because ADR 0004 made the vector engine copy the scalar engine:

1. **An OR ability gave both options.** Both options shared one action,
   `effect_nonscrap:<card>`: it applied the first option, and used again it
   applied the OR parent, which applied the remaining option. Ten trade-deck
   copies have an OR ability. In 400 greedy self-play games the champion
   followed 2,998 of 3,117 first options (96.2%) with a second option that
   changed something.
2. **Choices in a ship's primary ability could wait.** They were
   `effect_nonscrap:<ship>` actions usable at any time in the main phase.
   Fifteen trade-deck copies hold one; the champion made 731 of 4,169 of
   these choices later in the turn.
3. **Machine Base drew at once on the turn it was played**, and its scrap
   ("Draw a card, then scrap a card from your hand") became a separate
   optional action, skipped on 9 of 100 plays. On later turns it was already
   one activation that draws and then must scrap.

Two more deviations turned up while fixing these, Central Office's while
checking every card for the Machine Base pattern and Blob World's in code
review:

4. **Blob World counted only the Blob cards still in play.** Its option
   "Draw a card for each Blob card that you've played this turn" counted the
   cards in play, so a Blob card scrapped after it was played (Ram for its
   trade, or a Stealth Needle that copied a Blob ship) no longer counted.
5. **An established Central Office forced its top-deck.** On the turn it is
   played its 2 Trade resolve at once and "You may put the next ship you
   acquire this turn on top of your deck" stays a separate optional action,
   but on later turns one activation gave the trade and the top-deck
   together.

The action layout had ten card-indexed groups (494 actions for the 49 Core Set
cards), so no action could name the second option of an OR ability.

## Decision

**A second option action.** A new card-indexed group `effect_alt`
(`effect_alt:<card>`) is appended after `discard`, the last group, so every
existing action index keeps its position. The layout grows to 11 groups:
543 actions for the Core Set (49 x 11 + 4 global actions).
`get_action_space_layout` in `src/encoding/action_encoder.py` is the single
source of truth for it; nothing else assumes a group count. Option 1 of an OR
ability is `effect_nonscrap:<card>` and option 2 is `effect_alt:<card>`.
Choosing either uses the whole ability for the turn: the other option is no
longer offered. A base's OR ability is used whenever the player chooses
during the main phase, including the turn the base is played. No pending
state is needed: the action names the card. With several copies of a card in
play, an option action applies to one copy whose ability is unused, picked by
the order every card-scoped action already uses, identically in both engines.

**A ship's primary ability resolves completely when the ship is played.**
`PlayResolution` and `ability_play_resolution` in `src/rules/ir.py` say what
of a top-level ability resolves on play; both engines use them. A ship's
primary ability resolves whole: its choices become pending decisions that the
player makes before any other action, through the existing target actions
(`scrap_hand`, `scrap_discard`, `scrap_trade`, `destroy_base`). SKIP is legal
when the text says "you may". A choice with no legal target opens no
decision, as with every other pending choice. Patrol Mech's OR ability opens
a mandatory choice between its two option actions (no SKIP); the vector
engine represents it as a new pending mode, `PENDING_CHOOSE_OPTION`. A
Stealth Needle that copies one of these ships resolves the copied choice at
once in the same way. Ally and scrap abilities stay usable at any time during
the main phase.

**An automatic part followed by a mandatory choice is one activation.** A
base's primary ability still resolves its automatic parts on the turn the
base is played (in the Core Set only combat and trade, whose timing within
the main phase changes nothing), unless a mandatory choice must follow them
in the same ability. Then the whole ability stays one player-activated
action, as on later turns. Of the Core Set's cards only Machine Base has this
pattern.

**An established base resolves like a base just played.** On a later turn,
the first activation of a base's primary ability resolves the same automatic
parts as playing the base does, and its choices stay a separate activation,
as on the turn it is played (`_activation_resolution` in
`src/rules/scalar.py`; in the vector engine an activation group with ready
automatic bundles applies only those). Only Central Office has both: its 2
Trade, then its optional top-deck, on every turn. Card text with no operation
of its own is never an automatic part on its own, so Brain World, whose "Draw
a card for each card scrapped this way" is its scrap's completion draw, stays
one activation that opens its scrap.

**Blob World counts every Blob card played this turn.** Each player keeps a
per-turn count of the cards played by faction (`Player.faction_plays`,
`VectorState.faction_plays`): playing a card counts it for each of its
factions, a Stealth Needle's copy counts it again for each faction the copy
adds, scrapping never decrements it, and the end of the turn resets it. The
draw-per-faction opcode, which counts exactly one faction, reads it. Blob
World counts itself on the turn it is played.

**Rules version.** `src.rules.ir.RULES_VERSION` is 2. Checkpoint schema
version 5 records `rules_version`; a checkpoint without one was trained under
rules version 1. `load_checkpoint` rejects a checkpoint whose rules version
differs from the engines', naming both versions and this record, before any
tensor shape is compared. `RULE_IR_VERSION` is 2, with a new golden snapshot.

## Behaviour changes

| Card (trade-deck copies) | Ability | Rules version 1 | Rules version 2 |
|---|---|---|---|
| Patrol Mech (2) | {Gain 3 Trade} OR {Gain 5 Combat} | Effect action at any time: 3 Trade, then used again 5 Combat as well | Pending as it is played: `effect_nonscrap` (3 Trade) or `effect_alt` (5 Combat), no SKIP. Its ally scrap ability is unchanged |
| Trading Post (2) | {Gain 1 Authority} OR {Gain 1 Trade} | Both options, by two uses of one action | One option: `effect_nonscrap` (Authority) or `effect_alt` (Trade), at any time in the main phase |
| Barter World (2) | {Gain 2 Authority} OR {Gain 2 Trade} | Both options | One option: Authority or Trade |
| Defense Center (1) | {Gain 3 Authority} OR {Gain 2 Combat} | Both options | One option: Authority or Combat |
| Blob World (1) | {Gain 5 Combat} OR draw a card for each Blob card played this turn | Both options | One option: Combat or the draws |
| Recycling Station (2) | {Gain 1 Trade} OR discard up to two cards, then draw that many | Both options | One option: Trade, or the optional discard-then-draw decision |
| Trade Bot (3) | {Gain 1 Trade}; you may scrap a card in your hand or discard pile | Trade on play; the scrap was an action usable at any time in the main phase | Scrap choice pending as it is played, SKIP legal |
| Missile Bot (3) | {Gain 2 Combat}; you may scrap a card in your hand or discard pile | As Trade Bot | As Trade Bot |
| Supply Bot (3) | {Gain 2 Trade}; you may scrap a card in your hand or discard pile | As Trade Bot | As Trade Bot |
| Battle Mech (1) | {Gain 4 Combat}; you may scrap a card in your hand or discard pile | As Trade Bot | As Trade Bot |
| Battle Pod (2) | {Gain 4 Combat}; you may scrap a card in the trade row | As Trade Bot | Trade-row scrap pending as it is played, SKIP legal |
| Missile Mech (1) | {Gain 6 Combat}; you may destroy target base | Combat on play; the destroy-base was an action usable at any time | Destroy-base pending as it is played, SKIP legal |
| Machine Base (1) | Draw a card, then scrap a card from your hand | Turn played: drew at once, and the scrap was an optional action; later turns: one activation | Every turn one activation: draw, then a mandatory scrap from the hand (no SKIP); nothing when played |
| Blob World (1), draw option | Draw a card for each Blob card that you've played this turn | Counted the Blob cards in play | Counts every Blob card played this turn, also one scrapped since and a Stealth Needle that copied a Blob ship |
| Central Office (1) | {Gain 2 Trade}; you may put the next ship you acquire this turn on top of your deck | Turn played: trade at once, top-deck an optional action; later turns: one activation gave both | Every turn: the trade (as played, or by the first activation), then the top-deck as an optional activation |
| Stealth Needle (1) | Copy another ship you've played this turn | A copied choice was an action usable at any time | A copied choice is pending at once; a copied Patrol Mech opens its option choice |

Cards checked for the Machine Base pattern: Central Office (an automatic part
followed by an optional choice, which the established-base rule above
covers), Brain World, Junkyard and Recycling Station (choices without an
automatic part before them), and the scrap abilities of Battlecruiser and
Port of Call (draw, then an optional destroy-base, used whenever the player
chooses). None has an automatic part followed by a mandatory choice.

## Scripted agents

Each rule is deterministic and documented in the agent's code, and the vector
fixed opponents (`src/vector_engine/opponents.py`) replicate it. As ADR 0006
specifies, a choice between different cards that tie on the scalar agent's
score, and differ only in the scalar engine's list order, is broken by card
order in the vector replica.

- **Random**: uniform over the physical legal actions, as before. Each OR
  option is its own action, and a pending decision offers its choices plus
  SKIP when the choice is optional.
- **Simple**: the first legal action: option 1 of every OR ability and the
  first choice of every pending decision, never SKIP while a target exists.
  It uses abilities after purchases, so on both turns it buys what it can
  before it uses Central Office's top-deck.
- **Heuristic**: an OR option by `_choose_or_option`, in order: combat that
  reaches the opponent's authority while no outpost blocks a direct attack;
  trade that lets it afford a card it would buy this turn (its trade after
  playing its whole hand is short of the card's cost, and the option covers
  it); the draw-per-Blob option when at least two Blob cards were played this
  turn (scrapped or not); otherwise combat, then authority, then trade. It
  plays its whole hand before it uses a base, so Recycling Station's discard
  option would discard nothing and it takes the trade. It uses every other
  ability it can, so it takes Central Office's trade and then its top-deck on
  every turn. A ship's choice made as the ship is played follows its
  existing pending rules: destroy the first base offered; scrap Vipers first
  in the early game and Scouts first later, then Explorers, a copy in the
  discard pile before one in the hand, otherwise the most expensive
  trade-row card costing at least 5, otherwise SKIP.
- **Held-out bots** (`src/ai/held_out_agents.py`): an OR option by
  `_choose_or` per style: Recycling Station discards and draws while the hand
  holds junk, else trade; Blob World draws with at least two Blob cards
  played this turn (scrapped or not), else combat; Patrol Mech takes combat
  when it reaches the opponent's authority plus outpost defense, otherwise
  trade in the early phase (also the middle phase except for aggro) and
  combat later; Trading Post and Barter World take trade in the early and
  middle phases, then authority; Defense Center takes combat when no outpost
  is in the way or it is lethal (fortress: authority), otherwise authority.
  They consider an ability again when an activation left part of it unused,
  so they take Central Office's top-deck on every turn. A ship's choice made
  as the ship is played uses their existing pending rules (`_pending`),
  unchanged: destroy an outpost first, then the most expensive base; scrap
  their own junk, from the discard pile before the hand, otherwise SKIP (or
  their cheapest card when the scrap is mandatory); scrap a trade-row card
  only to deny an expensive one they cannot afford.

## Verification

- Per-card tests (`tests/test_rulebook_ability_timing.py`) step the scalar
  and vector engines together for every card above: the legal actions before
  and after each choice, that exactly one OR option applies, that ship
  choices are pending immediately and block every other action, SKIP
  availability, vacuous choices, and once-per-turn use; Blob World's count
  with a Blob card scrapped first (Blob World established or played this
  turn), with a Stealth Needle copy, and on the next turn; Central Office on
  the turn played and a later turn, with and without the top-deck; Brain
  World as one activation on both turns.
- The differential oracle compares masks and observations before every
  action and canonical state, including the per-faction play counts, after
  it. A new trace runs 16 games whose decks hold every card above plus Ram,
  capped at 300 turns because random play can scrap both decks to nothing
  with Machine Base; it uses both options of every OR ability, opens every
  ship's choice on play and through Stealth Needle copies, and uses Machine
  Base, Central Office, Ram's scrap, SKIP and discard-pile targets. The new
  coverage found two older divergences between the engines, now fixed: a
  vector discard decision whose targets ran out kept a state the scalar
  engine does not have, and the vector observation reported a legal
  purchase during Blob Carrier's free acquisition, which the scalar action
  context does not.
- The vector fixed opponents still match their scalar originals, including
  Heuristic's Blob World rule with a scrapped Blob card and Heuristic and
  Simple over an established Central Office's two activations.

## Rejected alternatives

**A pending two-way choice with two global actions.** Every OR ability would
open a pending decision answered by "first option" or "second option"
actions. The actions would not say which card's ability is being decided, so
the observation would need a new field for it, and a base's ability would
take two decisions to use. The appended `effect_alt` group needs neither.

**Inserting `effect_alt` beside `effect_nonscrap`.** It would shift every
later action index, and every tool or record that names indices, for no
benefit: old checkpoints cannot be used under the new rules either way.

**Checkpoint surgery** (widening a 494-action head to 543 and fine-tuning).
The old weights were trained under other rules; not implemented.

## Consequences

- Every earlier checkpoint and result belongs to rules version 1. Old
  checkpoints do not load. The published presets that start from or play
  against them (`continue-selfplay-seed0`, `exploit-selfplay-seed0`,
  `exploit-ladder-random`) stay in `src/config.py` as the record of
  published results, and their descriptions say so.
- `docs/eval_protocol.md` `v0.3`: measurements under rules version 2 are not
  comparable to earlier ones.
- The observation layout is unchanged (505 floats); the policy head grows
  from 494 to 543 actions. The vector state gains `faction_plays`, which the
  canonical snapshot and the information-set fingerprint include. The
  observation does not show cards scrapped this turn, so after a Blob card
  was scrapped a policy does not see the whole count Blob World's draw
  option uses; no rule requires that, so the layout stays as it is.

## Open questions

- **Stealth Needle copying a Stealth Needle** that copied nothing gains the
  copy ability again and must copy again, possibly the same Needle, without
  end. The Core Set has one Stealth Needle, so a real game cannot reach it.
