"""Backend-neutral effect opcodes and Effect-to-op compilation."""
from __future__ import annotations

from enum import IntEnum

from src.cards.effects import CardEffectType, Effect
from src.cards.factions import Faction, parse_faction

RULE_IR_VERSION = 1


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


def effect_defaults_to_auto(effect: Effect) -> bool:
    return (
        effect.effect_type in AUTO_EFFECT_TYPES
        and not effect.is_or_effect
        and not effect.is_scrap_effect
        and not effect.faction_requirement
    )


def compile_effect_trigger(
    effect: Effect,
    *,
    inherited_scrap: bool = False,
    inherited_faction: str | None = None,
    inherited_requirement: int = 0,
    inherited_auto: bool = False,
) -> tuple[EffectTrigger, int, int]:
    if effect.effect_type in {
        CardEffectType.PASSIVE_SHIP_COMBAT,
        CardEffectType.WILDCARD_ALLY,
    }:
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
    if effect.effect_type in CHOICE_EFFECT_TYPES:
        return EffectTrigger.ACTIVATE, int(Faction.NONE), 0
    if inherited_auto:
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
            int(parse_faction(effect.faction_target))
            if effect.faction_target
            else int(Faction.NONE)
        )
        return EffectOpcode.DRAW_PER_FACTION, 0, faction
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
