"""Backend-neutral effect opcodes and Effect-to-op compilation."""
from __future__ import annotations

from enum import IntEnum

from src.cards.effects import CardEffectType, Effect
from src.cards.factions import SINGLE_FACTIONS, Faction, parse_faction

RULE_IR_VERSION = 2

# Version of the game rules both engines implement. Checkpoints record it and
# a checkpoint trained under another version does not load, because the action
# layout and the meaning of actions differ between versions.
#   1: every result before decision 0016. An OR ability gave both options,
#      choices in a ship's primary ability could be used at any time,
#      Machine Base drew at once when played (494 actions for the Core Set),
#      and Blob World counted only the Blob cards still in play.
#   2: docs/decisions/0016-rulebook-ability-timing.md. An OR ability gives one
#      option (543 actions for the Core Set: options have their own action
#      group), a ship's primary ability resolves completely when the ship is
#      played, Machine Base draws and then must scrap in one activation, also
#      on the turn it is played, and Blob World counts every Blob card played
#      this turn.
RULES_VERSION = 2


class EffectOpcode(IntEnum):
    NOP = 0
    ADD_TRADE = 1
    ADD_COMBAT = 2
    ADD_HEALTH = 3
    DRAW = 4
    DRAW_PER_FACTION = 5
    DRAW_IF_BASES_GE_2 = 6
    QUEUE_SCRAP = 7
    ADD_DISCARD_DEBT = 8
    QUEUE_DESTROY_BASE = 9
    QUEUE_SCRAP_DRAW = 10
    QUEUE_DISCARD_DRAW = 11
    QUEUE_ACQUIRE_SHIP_TOP = 12
    ADD_NEXT_SHIP_TOP = 13
    QUEUE_COPY_SHIP = 14
    PASSIVE_SHIP_COMBAT = 15
    WILDCARD_ALLY = 16


class EffectTrigger(IntEnum):
    AUTO = 1
    ALLY = 2
    SCRAP = 3
    ACTIVATE = 4
    PASSIVE = 5
    # An option of the OR ability in a ship's primary ability. The ship's
    # player chooses one option in a pending decision as soon as it is played.
    ON_PLAY_OPTION = 6


class PlayResolution(IntEnum):
    """How much of a top-level ability resolves when its card is played.

    Rulebook 1.1E, pages 4 and 5: "When a ship is played, its Primary Ability
    happens immediately"; "you may choose when during your Main Phase to use
    your bases' Primary Abilities"; ally and scrap abilities may be used at any
    time during the Main Phase.
    """

    # Nothing: the player uses the ability later (an activated base ability,
    # an ally, scrap or passive ability, a base's OR ability, or a base
    # ability holding a mandatory choice, whose automatic parts and the
    # choice that must follow them are one activation).
    NONE = 0
    # The automatic parts (resources, draws, discard debt); any optional
    # choice in it stays a player-activated action. A base's primary ability
    # on the turn the base is played.
    AUTO_PARTS = 1
    # All of it: a ship's primary ability. Its choices become pending
    # decisions made before any other action, and an OR ability opens a
    # mandatory choice between its options.
    WHOLE = 2


AUTO_EFFECT_TYPES = frozenset(
    {
        CardEffectType.COMBAT,
        CardEffectType.TRADE,
        CardEffectType.HEAL,
        CardEffectType.DRAW,
        CardEffectType.TARGET_DISCARD,
        CardEffectType.PARENT,
        CardEffectType.COPY_SHIP,
        CardEffectType.COMPLEX,
        CardEffectType.PASSIVE_SHIP_COMBAT,
        CardEffectType.WILDCARD_ALLY,
    }
)

CHOICE_EFFECT_TYPES = frozenset(
    {
        CardEffectType.SCRAP,
        CardEffectType.DESTROY_BASE,
        CardEffectType.SCRAP_FROM_HAND_DISCARD,
        CardEffectType.DISCARD_DRAW,
        CardEffectType.NEXT_SHIP_TOP,
    }
)

PASSIVE_EFFECT_TYPES = frozenset(
    {
        CardEffectType.PASSIVE_SHIP_COMBAT,
        CardEffectType.WILDCARD_ALLY,
    }
)


def _holds_mandatory_choice(effect: Effect) -> bool:
    if effect.effect_type in CHOICE_EFFECT_TYPES and effect.is_mandatory:
        return True
    return any(
        _holds_mandatory_choice(child) for child in effect.child_effects or ()
    )


def ability_play_resolution(effect: Effect, *, ship: bool) -> PlayResolution:
    """What of the top-level ability *effect* resolves when its card is played.

    *ship* says whether the card is a ship rather than a base.
    """
    if (
        effect.effect_type in PASSIVE_EFFECT_TYPES
        or effect.is_scrap_effect
        or effect.faction_requirement
    ):
        return PlayResolution.NONE
    if ship:
        return PlayResolution.WHOLE
    if (
        effect.is_or_effect
        or effect.effect_type not in AUTO_EFFECT_TYPES
        or _holds_mandatory_choice(effect)
    ):
        return PlayResolution.NONE
    return PlayResolution.AUTO_PARTS


def compile_effect_trigger(
    effect: Effect,
    *,
    inherited_scrap: bool = False,
    inherited_faction: str | None = None,
    inherited_requirement: int = 0,
    resolution: PlayResolution = PlayResolution.NONE,
) -> tuple[EffectTrigger, int, int]:
    """Trigger of a non-OR effect under its top-level ability's *resolution*."""
    if effect.effect_type in PASSIVE_EFFECT_TYPES:
        return EffectTrigger.PASSIVE, int(Faction.NONE), 0
    faction_name = effect.faction_requirement or inherited_faction
    requirement = (
        effect.faction_requirement_count
        if effect.faction_requirement
        else inherited_requirement
    )
    if effect.is_scrap_effect or inherited_scrap:
        return EffectTrigger.SCRAP, int(Faction.NONE), 0
    if faction_name:
        return EffectTrigger.ALLY, int(parse_faction(faction_name)), requirement
    if resolution == PlayResolution.WHOLE:
        return EffectTrigger.AUTO, int(Faction.NONE), 0
    if effect.effect_type in CHOICE_EFFECT_TYPES:
        return EffectTrigger.ACTIVATE, int(Faction.NONE), 0
    if resolution == PlayResolution.AUTO_PARTS:
        return EffectTrigger.AUTO, int(Faction.NONE), 0
    return EffectTrigger.ACTIVATE, int(Faction.NONE), 0


TARGET_HAND = 1
TARGET_DISCARD = 2
TARGET_TRADE = 4
PENDING_MANDATORY = 1 << 8
PENDING_DRAW_ON_COMPLETE = 1 << 9


def compile_effect_op(effect: Effect) -> tuple[EffectOpcode, int, int]:
    """Compile one non-parent Effect node into one backend-neutral opcode."""
    kind = effect.effect_type
    if kind == CardEffectType.COMBAT:
        return EffectOpcode.ADD_COMBAT, effect.value, 0
    if kind == CardEffectType.TRADE:
        return EffectOpcode.ADD_TRADE, effect.value, 0
    if kind == CardEffectType.HEAL:
        return EffectOpcode.ADD_HEALTH, effect.value, 0
    if kind == CardEffectType.DRAW:
        return EffectOpcode.DRAW, effect.value, 0
    if kind == CardEffectType.DRAW_PER_FACTION:
        faction = (
            parse_faction(effect.faction_target)
            if effect.faction_target
            else Faction.NONE
        )
        if faction not in SINGLE_FACTIONS:
            raise ValueError(
                "a draw per faction counts the cards of exactly one faction"
            )
        return EffectOpcode.DRAW_PER_FACTION, 0, int(faction)
    if kind == CardEffectType.CONDITIONAL_DRAW:
        if effect.text != "bases_ge_2":
            raise ValueError(
                f"unsupported conditional draw condition {effect.text!r}"
            )
        return EffectOpcode.DRAW_IF_BASES_GE_2, effect.value, 2
    if kind == CardEffectType.SCRAP:
        sources = 0
        for target in effect.card_targets or ():
            if target == "hand":
                sources |= TARGET_HAND
            elif target == "discard":
                sources |= TARGET_DISCARD
            elif target == "trade":
                sources |= TARGET_TRADE
            else:
                raise ValueError(f"unsupported scrap target {target!r}")
        if effect.is_mandatory:
            sources |= PENDING_MANDATORY
        return EffectOpcode.QUEUE_SCRAP, effect.value, sources
    if kind == CardEffectType.TARGET_DISCARD:
        if effect.card_targets and "opponent" not in effect.card_targets:
            raise ValueError("target discard must target opponent")
        return EffectOpcode.ADD_DISCARD_DEBT, effect.value, 0
    if kind == CardEffectType.DESTROY_BASE:
        flags = PENDING_MANDATORY if effect.is_mandatory else 0
        return EffectOpcode.QUEUE_DESTROY_BASE, effect.value, flags
    if kind == CardEffectType.SCRAP_FROM_HAND_DISCARD:
        flags = TARGET_HAND | TARGET_DISCARD | PENDING_DRAW_ON_COMPLETE
        return EffectOpcode.QUEUE_SCRAP_DRAW, effect.value, flags
    if kind == CardEffectType.DISCARD_DRAW:
        flags = TARGET_HAND | PENDING_DRAW_ON_COMPLETE
        return EffectOpcode.QUEUE_DISCARD_DRAW, effect.value, flags
    if kind == CardEffectType.ACQUIRE_SHIP_TOP:
        return EffectOpcode.QUEUE_ACQUIRE_SHIP_TOP, effect.value, 0
    if kind == CardEffectType.NEXT_SHIP_TOP:
        return EffectOpcode.ADD_NEXT_SHIP_TOP, effect.value, 0
    if kind == CardEffectType.COPY_SHIP:
        return EffectOpcode.QUEUE_COPY_SHIP, effect.value, 0
    if kind == CardEffectType.PASSIVE_SHIP_COMBAT:
        return EffectOpcode.PASSIVE_SHIP_COMBAT, effect.value, 0
    if kind == CardEffectType.WILDCARD_ALLY:
        return EffectOpcode.WILDCARD_ALLY, 0, int(Faction.ALL)
    if kind == CardEffectType.COMPLEX:
        return EffectOpcode.NOP, 0, 0
    raise ValueError(f"unsupported effect type {kind.value}")
