"""Compile Card/Effect objects into fixed-width vector-engine rule tables.

The existing Effect tree is the source language. Gate 2 deliberately does not
create a second hand-authored card implementation: both the scalar IR
interpreter used by tests and the Numba engine consume these tables.

Unsupported target-selection effects are rejected per card rather than
silently approximated. ``COMPLEX`` effects compile to NOP because that is their
current scalar-engine behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from src.cards.card import Card, CardType
from src.cards.effects import CardEffectType, Effect
from src.cards.factions import Faction, parse_faction
from src.cards.registry import CardRegistry


MAX_BUNDLES = 8
MAX_OPS = 16
MAX_PENDING_PER_ACTIVATION = 4
PENDING_DOMAIN_BASE = 1 << 3


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


class EffectTrigger(IntEnum):
    AUTO = 1
    ALLY = 2
    SCRAP = 3
    ACTIVATE = 4


TARGET_HAND = 1
TARGET_DISCARD = 2
TARGET_TRADE = 4
PENDING_MANDATORY = 1 << 8
PENDING_DRAW_ON_COMPLETE = 1 << 9

_AUTO_EFFECT_TYPES = frozenset(
    {
        CardEffectType.COMBAT,
        CardEffectType.TRADE,
        CardEffectType.HEAL,
        CardEffectType.DRAW,
        CardEffectType.TARGET_DISCARD,
        CardEffectType.PARENT,
    }
)


@dataclass(frozen=True)
class CompiledRules:
    card_names: tuple[str, ...]
    supported: np.ndarray
    unsupported_reason: tuple[str | None, ...]

    cost: np.ndarray
    card_type: np.ndarray
    defense: np.ndarray
    faction: np.ndarray
    ally_factions: np.ndarray

    bundle_count: np.ndarray
    bundle_trigger: np.ndarray
    bundle_faction: np.ndarray
    bundle_requirement: np.ndarray
    bundle_source_effect: np.ndarray
    bundle_stage: np.ndarray
    bundle_next: np.ndarray
    bundle_op_start: np.ndarray
    bundle_op_count: np.ndarray

    op_count: np.ndarray
    op_code: np.ndarray
    op_value: np.ndarray
    op_arg: np.ndarray

    @property
    def num_cards(self) -> int:
        return len(self.card_names)

    @property
    def supported_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for index, name in enumerate(self.card_names)
            if bool(self.supported[index])
        )


@dataclass(frozen=True)
class _Leaf:
    opcode: EffectOpcode
    value: int
    arg: int
    trigger: EffectTrigger
    faction: int
    requirement: int
    stage: int


def _starter_templates(registry: CardRegistry) -> dict[str, Card]:
    scout = Card(
        "Scout",
        registry.card_index_map["Scout"],
        0,
        [Effect(CardEffectType.TRADE, 1)],
        CardType.SHIP,
    )
    viper = Card(
        "Viper",
        registry.card_index_map["Viper"],
        0,
        [Effect(CardEffectType.COMBAT, 1)],
        CardType.SHIP,
    )
    explorer = Card(
        "Explorer",
        registry.card_index_map["Explorer"],
        2,
        [
            Effect(CardEffectType.TRADE, 2),
            Effect(CardEffectType.COMBAT, 2, is_scrap_effect=True),
        ],
        CardType.SHIP,
    )
    return {card.name: card for card in (scout, viper, explorer)}


def _templates_by_name(
    cards: list[Card], registry: CardRegistry
) -> dict[str, Card]:
    templates: dict[str, Card] = {}
    for card in cards:
        templates.setdefault(card.name, card)
    for name, card in _starter_templates(registry).items():
        templates.setdefault(name, card)
    return templates


def _opcode(effect: Effect) -> tuple[EffectOpcode, int, int]:
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
    if kind == CardEffectType.COMPLEX:
        # The object engine intentionally treats COMPLEX as an applied no-op.
        return EffectOpcode.NOP, 0, 0
    raise ValueError(f"unsupported effect type {kind.value}")


def _effective_gate(
    effect: Effect,
    *,
    inherited_scrap: bool,
    inherited_faction: str | None,
    inherited_requirement: int,
    inherited_auto: bool,
) -> tuple[EffectTrigger, int, int]:
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
    if inherited_auto:
        return EffectTrigger.AUTO, int(Faction.NONE), 0
    return EffectTrigger.ACTIVATE, int(Faction.NONE), 0


def _flatten(
    effect: Effect,
    *,
    inherited_scrap: bool = False,
    inherited_faction: str | None = None,
    inherited_requirement: int = 0,
    inherited_auto: bool | None = None,
    stage: int = 0,
    inside_or: bool = False,
) -> list[_Leaf]:
    inherited_scrap = inherited_scrap or effect.is_scrap_effect
    if effect.is_or_effect and inside_or:
        raise ValueError("nested OR effects are unsupported")
    if effect.faction_requirement and inherited_faction:
        if parse_faction(effect.faction_requirement) != parse_faction(
            inherited_faction
        ):
            raise ValueError(
                "nested faction requirements form an unsupported conjunction"
            )
    inherited_faction = effect.faction_requirement or inherited_faction
    if effect.faction_requirement:
        inherited_requirement = effect.faction_requirement_count
    if inherited_auto is None:
        inherited_auto = (
            effect.effect_type in _AUTO_EFFECT_TYPES
            and not effect.is_or_effect
        )
    if inherited_scrap and inherited_faction:
        raise ValueError(
            "faction-gated scrap effects require independent eligibility "
            "and activation state"
        )
    if effect.is_or_effect and inherited_faction:
        raise ValueError(
            "faction-gated OR effects require atomic compound activation"
        )
    if effect.is_or_effect and inherited_scrap:
        raise ValueError("scrap-gated OR effects are unsupported")

    if effect.effect_type == CardEffectType.PARENT:
        if effect.is_or_effect:
            children = effect.child_effects or ()
            if len(children) != 2:
                raise ValueError(
                    "OR effects require exactly two child choices"
                )
            if any(
                child.is_or_effect or child.is_scrap_effect
                for child in children
            ):
                raise ValueError(
                    "nested or scrap OR children are unsupported"
                )
        parent_gate = _effective_gate(
            effect,
            inherited_scrap=inherited_scrap,
            inherited_faction=inherited_faction,
            inherited_requirement=inherited_requirement,
            inherited_auto=inherited_auto,
        )
        leaves: list[_Leaf] = []
        children = effect.child_effects or ()
        for child_index, child in enumerate(children):
            child_stage = stage
            if effect.is_or_effect:
                # ActionContext resolves the first child directly. On the
                # next action it resolves the parent, which applies every
                # remaining child. Thus OR is two activation stages.
                child_stage = 0 if child_index == 0 else 1
            leaves.extend(
                _flatten(
                    child,
                    inherited_scrap=inherited_scrap,
                    inherited_faction=inherited_faction,
                    inherited_requirement=inherited_requirement,
                    inherited_auto=inherited_auto,
                    stage=child_stage,
                    inside_or=inside_or or effect.is_or_effect,
                )
            )
        gates = {
            (leaf.trigger, leaf.faction, leaf.requirement)
            for leaf in leaves
        }
        if gates != {parent_gate}:
            raise ValueError(
                "mixed-gate parent effects cannot preserve atomic "
                "Effect.apply semantics"
            )
        return leaves

    opcode, value, arg = _opcode(effect)
    trigger, faction, requirement = _effective_gate(
        effect,
        inherited_scrap=inherited_scrap,
        inherited_faction=inherited_faction,
        inherited_requirement=inherited_requirement,
        inherited_auto=inherited_auto,
    )
    return [
        _Leaf(
            opcode=opcode,
            value=value,
            arg=arg,
            trigger=trigger,
            faction=faction,
            requirement=requirement,
            stage=stage,
        )
    ]


def _bundle_leaves(leaves: list[_Leaf]) -> list[list[_Leaf]]:
    """Group leaves applied by one encoded APPLY_EFFECT action."""
    bundles: list[list[_Leaf]] = []
    for leaf in leaves:
        key = (leaf.trigger, leaf.faction, leaf.requirement, leaf.stage)
        if bundles:
            previous = bundles[-1][0]
            if key == (
                previous.trigger,
                previous.faction,
                previous.requirement,
                previous.stage,
            ):
                bundles[-1].append(leaf)
                continue
        bundles.append([leaf])
    return bundles


def _pending_domain(leaf: _Leaf) -> int:
    if leaf.opcode == EffectOpcode.QUEUE_SCRAP:
        return leaf.arg & (TARGET_HAND | TARGET_DISCARD | TARGET_TRADE)
    if leaf.opcode == EffectOpcode.QUEUE_SCRAP_DRAW:
        return TARGET_HAND | TARGET_DISCARD
    if leaf.opcode == EffectOpcode.QUEUE_DISCARD_DRAW:
        return TARGET_HAND
    if leaf.opcode == EffectOpcode.QUEUE_DESTROY_BASE:
        return PENDING_DOMAIN_BASE
    return 0


def _validate_disjoint_pending_domains(leaves: list[_Leaf]) -> None:
    seen = 0
    for leaf in leaves:
        domain = _pending_domain(leaf)
        if domain and (seen & domain):
            raise ValueError(
                "one activation queues overlapping pending target domains"
            )
        seen |= domain


def compile_rules(
    cards: list[Card], registry: CardRegistry
) -> CompiledRules:
    """Compile all canonical card definitions.

    Cards containing target-selection or OR effects remain in the metadata
    tables but are marked unsupported with an explicit reason.
    """
    count = registry.num_cards
    names = tuple(registry.card_names)
    templates = _templates_by_name(cards, registry)

    supported = np.zeros(count, dtype=np.bool_)
    reasons: list[str | None] = [None] * count
    cost = np.zeros(count, dtype=np.int16)
    card_type = np.zeros(count, dtype=np.int8)
    defense = np.zeros(count, dtype=np.int16)
    faction = np.zeros(count, dtype=np.int8)
    ally_factions = np.zeros(count, dtype=np.int8)

    bundle_count = np.zeros(count, dtype=np.int8)
    bundle_trigger = np.zeros((count, MAX_BUNDLES), dtype=np.int8)
    bundle_faction = np.zeros((count, MAX_BUNDLES), dtype=np.int8)
    bundle_requirement = np.zeros((count, MAX_BUNDLES), dtype=np.int8)
    bundle_source_effect = np.full(
        (count, MAX_BUNDLES), -1, dtype=np.int8
    )
    bundle_stage = np.zeros((count, MAX_BUNDLES), dtype=np.int8)
    bundle_next = np.full((count, MAX_BUNDLES), -1, dtype=np.int8)
    bundle_op_start = np.zeros((count, MAX_BUNDLES), dtype=np.int8)
    bundle_op_count = np.zeros((count, MAX_BUNDLES), dtype=np.int8)

    op_count = np.zeros(count, dtype=np.int8)
    op_code = np.zeros((count, MAX_OPS), dtype=np.int8)
    op_value = np.zeros((count, MAX_OPS), dtype=np.int16)
    op_arg = np.zeros((count, MAX_OPS), dtype=np.int16)

    type_code = {
        CardType.SHIP: 0,
        CardType.BASE: 1,
        CardType.OUTPOST: 2,
    }

    for index, name in enumerate(names):
        card = templates.get(name)
        if card is None:
            reasons[index] = "missing canonical Card template"
            continue

        cost[index] = card.cost
        card_type[index] = type_code[card.card_type]
        defense[index] = card.defense or 0
        faction[index] = int(card.faction)
        ally_factions[index] = int(card.ally_factions)

        try:
            bundles_with_source: list[tuple[int, list[_Leaf]]] = []
            leaf_count = 0
            for source_effect, top_level in enumerate(card.effects):
                leaves = _flatten(top_level)
                leaf_count += len(leaves)
                bundles_with_source.extend(
                    (source_effect, bundle)
                    for bundle in _bundle_leaves(leaves)
                )
            if leaf_count > MAX_OPS:
                raise ValueError(
                    f"{leaf_count} leaf operations exceed MAX_OPS={MAX_OPS}"
                )
            if len(bundles_with_source) > MAX_BUNDLES:
                raise ValueError(
                    f"{len(bundles_with_source)} bundles exceed "
                    f"MAX_BUNDLES={MAX_BUNDLES}"
                )
            queue_opcodes = {
                EffectOpcode.QUEUE_SCRAP,
                EffectOpcode.QUEUE_DESTROY_BASE,
                EffectOpcode.QUEUE_SCRAP_DRAW,
                EffectOpcode.QUEUE_DISCARD_DRAW,
            }
            auto_queues = sum(
                leaf.opcode in queue_opcodes
                for _, bundle in bundles_with_source
                if bundle[0].trigger == EffectTrigger.AUTO
                for leaf in bundle
            )
            max_explicit_queues = max(
                (
                    sum(leaf.opcode in queue_opcodes for leaf in bundle)
                    for _, bundle in bundles_with_source
                    if bundle[0].trigger != EffectTrigger.AUTO
                ),
                default=0,
            )
            if max(auto_queues, max_explicit_queues) > MAX_PENDING_PER_ACTIVATION:
                raise ValueError(
                    "one activation can exceed the pending queue capacity "
                    f"({MAX_PENDING_PER_ACTIVATION})"
                )
            _validate_disjoint_pending_domains(
                [
                    leaf
                    for _, bundle in bundles_with_source
                    if bundle[0].trigger == EffectTrigger.AUTO
                    for leaf in bundle
                ]
            )
            for _, bundle in bundles_with_source:
                if bundle[0].trigger != EffectTrigger.AUTO:
                    _validate_disjoint_pending_domains(bundle)
        except ValueError as exc:
            reasons[index] = str(exc)
            continue

        operation = 0
        for bundle_index, (source_effect, bundle) in enumerate(
            bundles_with_source
        ):
            first = bundle[0]
            bundle_trigger[index, bundle_index] = int(first.trigger)
            bundle_faction[index, bundle_index] = first.faction
            bundle_requirement[index, bundle_index] = first.requirement
            bundle_source_effect[index, bundle_index] = source_effect
            bundle_stage[index, bundle_index] = first.stage
            bundle_op_start[index, bundle_index] = operation
            bundle_op_count[index, bundle_index] = len(bundle)
            for leaf in bundle:
                op_code[index, operation] = int(leaf.opcode)
                op_value[index, operation] = leaf.value
                op_arg[index, operation] = leaf.arg
                operation += 1

        # OR stage 0 transitions to the remaining-child stage 1 for the same
        # source effect. Non-OR bundles have no continuation.
        for bundle_index in range(len(bundles_with_source)):
            source_effect, bundle = bundles_with_source[bundle_index]
            if bundle[0].stage != 0:
                continue
            for candidate in range(bundle_index + 1, len(bundles_with_source)):
                candidate_source, candidate_bundle = bundles_with_source[
                    candidate
                ]
                if (
                    candidate_source == source_effect
                    and candidate_bundle[0].stage == 1
                ):
                    bundle_next[index, bundle_index] = candidate
                    break

        bundle_count[index] = len(bundles_with_source)
        op_count[index] = operation
        supported[index] = True

    return CompiledRules(
        card_names=names,
        supported=supported,
        unsupported_reason=tuple(reasons),
        cost=cost,
        card_type=card_type,
        defense=defense,
        faction=faction,
        ally_factions=ally_factions,
        bundle_count=bundle_count,
        bundle_trigger=bundle_trigger,
        bundle_faction=bundle_faction,
        bundle_requirement=bundle_requirement,
        bundle_source_effect=bundle_source_effect,
        bundle_stage=bundle_stage,
        bundle_next=bundle_next,
        bundle_op_start=bundle_op_start,
        bundle_op_count=bundle_op_count,
        op_count=op_count,
        op_code=op_code,
        op_value=op_value,
        op_arg=op_arg,
    )


def apply_bundle_to_totals(
    rules: CompiledRules,
    card_index: int,
    bundle_index: int,
    totals: np.ndarray,
    *,
    faction_counts: dict[int, int] | None = None,
    base_count: int = 0,
) -> None:
    """Small scalar IR interpreter used by compiler differential tests.

    ``totals`` layout: trade, combat, health, draws.
    """
    trigger = EffectTrigger(
        int(rules.bundle_trigger[card_index, bundle_index])
    )
    if trigger == EffectTrigger.ALLY:
        faction = int(rules.bundle_faction[card_index, bundle_index])
        requirement = int(
            rules.bundle_requirement[card_index, bundle_index]
        )
        if (faction_counts or {}).get(faction, 0) <= requirement:
            return

    start = int(rules.bundle_op_start[card_index, bundle_index])
    length = int(rules.bundle_op_count[card_index, bundle_index])
    for operation in range(start, start + length):
        opcode = EffectOpcode(int(rules.op_code[card_index, operation]))
        value = int(rules.op_value[card_index, operation])
        if opcode == EffectOpcode.ADD_TRADE:
            totals[0] += value
        elif opcode == EffectOpcode.ADD_COMBAT:
            totals[1] += value
        elif opcode == EffectOpcode.ADD_HEALTH:
            totals[2] += value
        elif opcode == EffectOpcode.DRAW:
            totals[3] += value
        elif opcode == EffectOpcode.DRAW_PER_FACTION:
            faction = int(rules.op_arg[card_index, operation])
            totals[3] += (faction_counts or {}).get(faction, 0)
        elif opcode == EffectOpcode.DRAW_IF_BASES_GE_2:
            if base_count >= int(rules.op_arg[card_index, operation]):
                totals[3] += value
        elif opcode != EffectOpcode.NOP:
            raise AssertionError(f"unhandled opcode {opcode}")
