"""Effect-object to vector-rule IR compiler tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.cards.card import Card, CardType
from src.cards.effects import CardEffectType, Effect
from src.cards.card import CardType
from src.config import DataConfig
from src.vector_engine.rules import (
    EffectOpcode,
    EffectTrigger,
    apply_bundle_to_totals,
    compile_rules,
)
from src.rules.ir import RULE_IR_VERSION


@pytest.fixture(scope="module")
def compiled():
    data = DataConfig()
    cards = data.load_cards()
    registry = data.build_registry(cards)
    return cards, registry, compile_rules(cards, registry)


def _bundles(rules, card_name):
    card = rules.card_names.index(card_name)
    return [
        {
            "trigger": EffectTrigger(int(rules.bundle_trigger[card, bundle])),
            "faction": int(rules.bundle_faction[card, bundle]),
            "requirement": int(rules.bundle_requirement[card, bundle]),
            "ops": [
                (
                    EffectOpcode(int(rules.op_code[card, operation])),
                    int(rules.op_value[card, operation]),
                    int(rules.op_arg[card, operation]),
                )
                for operation in range(
                    int(rules.bundle_op_start[card, bundle]),
                    int(rules.bundle_op_start[card, bundle])
                    + int(rules.bundle_op_count[card, bundle]),
                )
            ],
        }
        for bundle in range(int(rules.bundle_count[card]))
    ]


def _rule_ir_digest(rules):
    cards = []
    for card in range(rules.num_cards):
        bundles = []
        for bundle in range(int(rules.bundle_count[card])):
            start = int(rules.bundle_op_start[card, bundle])
            count = int(rules.bundle_op_count[card, bundle])
            bundles.append(
                {
                    "trigger": int(rules.bundle_trigger[card, bundle]),
                    "faction": int(rules.bundle_faction[card, bundle]),
                    "requirement": int(
                        rules.bundle_requirement[card, bundle]
                    ),
                    "source": int(
                        rules.bundle_source_effect[card, bundle]
                    ),
                    "stage": int(rules.bundle_stage[card, bundle]),
                    "next": int(rules.bundle_next[card, bundle]),
                    "ops": [
                        [
                            int(rules.op_code[card, operation]),
                            int(rules.op_value[card, operation]),
                            int(rules.op_arg[card, operation]),
                        ]
                        for operation in range(start, start + count)
                    ],
                }
            )
        cards.append(
            {
                "name": rules.card_names[card],
                "cost": int(rules.cost[card]),
                "type": int(rules.card_type[card]),
                "defense": int(rules.defense[card]),
                "faction": int(rules.faction[card]),
                "ally_factions": int(rules.ally_factions[card]),
                "bundles": bundles,
            }
        )
    payload = json.dumps(
        {"version": RULE_IR_VERSION, "cards": cards},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def test_core_set_rule_ir_matches_versioned_golden_snapshot(compiled):
    _, _, rules = compiled
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "core_set_rule_ir_v1.json"
        ).read_text()
    )
    assert fixture["version"] == RULE_IR_VERSION
    assert fixture["card_count"] == rules.num_cards
    assert fixture["sha256"] == _rule_ir_digest(rules)


def test_compiles_canonical_width_and_metadata(compiled):
    _, registry, rules = compiled
    assert rules.num_cards == 49
    assert rules.card_names == tuple(registry.card_names)
    assert rules.cost.shape == (49,)
    assert rules.op_code.shape[0] == 49
    assert rules.bundle_trigger.shape[0] == 49

    trading_post = registry.card_index_map["Trading Post"]
    assert rules.card_type[trading_post] == 2
    assert rules.defense[trading_post] == 4


def test_supported_set_is_explicit_and_stable(compiled):
    _, registry, rules = compiled
    assert rules.supported_names == tuple(registry.card_names)
    assert len(rules.supported_names) == 49
    assert all(reason is None for reason in rules.unsupported_reason)


def test_trade_pod_compiles_auto_and_ally_bundles(compiled):
    _, _, rules = compiled
    bundles = _bundles(rules, "Trade Pod")
    assert bundles == [
        {
            "trigger": EffectTrigger.AUTO,
            "faction": 0,
            "requirement": 0,
            "ops": [(EffectOpcode.ADD_TRADE, 3, 0)],
        },
        {
            "trigger": EffectTrigger.ALLY,
            "faction": 1,  # Blob
            "requirement": 1,
            "ops": [(EffectOpcode.ADD_COMBAT, 2, 0)],
        },
    ]


def test_parent_children_remain_one_auto_bundle(compiled):
    _, _, rules = compiled
    bundles = _bundles(rules, "Mothership")
    assert bundles[0]["trigger"] == EffectTrigger.AUTO
    assert bundles[0]["ops"] == [
        (EffectOpcode.ADD_COMBAT, 6, 0),
        (EffectOpcode.DRAW, 1, 0),
    ]
    assert bundles[1]["trigger"] == EffectTrigger.ALLY
    assert bundles[1]["ops"] == [(EffectOpcode.DRAW, 1, 0)]


def test_mech_world_compiles_as_passive_wildcard_ally(compiled):
    _, _, rules = compiled
    bundles = _bundles(rules, "Mech World")
    assert bundles == [
        {
            "trigger": EffectTrigger.PASSIVE,
            "faction": 0,
            "requirement": 0,
            "ops": [(EffectOpcode.WILDCARD_ALLY, 0, 15)],
        }
    ]


@pytest.mark.parametrize(
    ("card_name", "opcode", "trigger"),
    [
        (
            "Blob Carrier",
            EffectOpcode.QUEUE_ACQUIRE_SHIP_TOP,
            EffectTrigger.ALLY,
        ),
        (
            "Central Office",
            EffectOpcode.ADD_NEXT_SHIP_TOP,
            EffectTrigger.ACTIVATE,
        ),
        (
            "Freighter",
            EffectOpcode.ADD_NEXT_SHIP_TOP,
            EffectTrigger.ALLY,
        ),
        (
            "Stealth Needle",
            EffectOpcode.QUEUE_COPY_SHIP,
            EffectTrigger.AUTO,
        ),
        (
            "Fleet HQ",
            EffectOpcode.PASSIVE_SHIP_COMBAT,
            EffectTrigger.PASSIVE,
        ),
    ],
)
def test_physical_complex_effects_have_explicit_ir(
    compiled, card_name, opcode, trigger
):
    _, _, rules = compiled
    assert any(
        bundle["trigger"] == trigger
        and any(operation[0] == opcode for operation in bundle["ops"])
        for bundle in _bundles(rules, card_name)
    )


def test_explorer_scrap_bundle_is_separate(compiled):
    _, _, rules = compiled
    bundles = _bundles(rules, "Explorer")
    assert bundles[0]["trigger"] == EffectTrigger.AUTO
    assert bundles[0]["ops"] == [(EffectOpcode.ADD_TRADE, 2, 0)]
    assert bundles[1]["trigger"] == EffectTrigger.SCRAP
    assert bundles[1]["ops"] == [(EffectOpcode.ADD_COMBAT, 2, 0)]


@pytest.mark.parametrize(
    ("card_name", "opcode"),
    [
        ("Battle Pod", EffectOpcode.QUEUE_SCRAP),
        ("Imperial Fighter", EffectOpcode.ADD_DISCARD_DEBT),
        ("Missile Mech", EffectOpcode.QUEUE_DESTROY_BASE),
        ("Brain World", EffectOpcode.QUEUE_SCRAP_DRAW),
        ("Recycling Station", EffectOpcode.ADD_TRADE),
    ],
)
def test_choice_effects_compile_to_current_scalar_semantics(
    compiled, card_name, opcode
):
    _, _, rules = compiled
    bundles = _bundles(rules, card_name)
    assert any(
        operation[0] == opcode
        for bundle in bundles
        for operation in bundle["ops"]
    )


def test_or_effect_compiles_current_two_stage_resolver_semantics(compiled):
    _, _, rules = compiled
    bundles = _bundles(rules, "Blob World")
    operations = [
        operation[0] for bundle in bundles for operation in bundle["ops"]
    ]
    assert operations == [
        EffectOpcode.ADD_COMBAT,
        EffectOpcode.DRAW_PER_FACTION,
    ]
    assert bundles[0]["trigger"] == EffectTrigger.ACTIVATE
    card = rules.card_names.index("Blob World")
    assert rules.bundle_stage[card, 0] == 0
    assert rules.bundle_stage[card, 1] == 1
    assert rules.bundle_next[card, 0] == 1


def test_compiler_rejects_unrepresentable_or_shapes(compiled):
    cards, registry, _ = compiled
    original = cards[0]
    three_child = Card(
        original.name,
        original.index,
        original.cost,
        [
            Effect(
                CardEffectType.PARENT,
                is_or_effect=True,
                child_effects=[
                    Effect(CardEffectType.COMBAT, 1),
                    Effect(CardEffectType.TRADE, 1),
                    Effect(CardEffectType.HEAL, 1),
                ],
            )
        ],
        CardType.SHIP,
    )
    rules = compile_rules([three_child, *cards], registry)
    assert not rules.supported[original.index]
    assert "exactly two" in rules.unsupported_reason[original.index]


def test_compiler_rejects_scrap_gated_or(compiled):
    cards, registry, _ = compiled
    original = cards[0]
    synthetic = Card(
        original.name,
        original.index,
        original.cost,
        [
            Effect(
                CardEffectType.PARENT,
                is_or_effect=True,
                is_scrap_effect=True,
                child_effects=[
                    Effect(CardEffectType.COMBAT, 1),
                    Effect(CardEffectType.TRADE, 1),
                ],
            )
        ],
        CardType.SHIP,
    )
    rules = compile_rules([synthetic, *cards], registry)
    assert not rules.supported[original.index]
    assert "scrap-gated OR" in rules.unsupported_reason[original.index]


def test_compiler_rejects_indirectly_nested_or(compiled):
    cards, registry, _ = compiled
    original = cards[0]
    nested = Effect(
        CardEffectType.PARENT,
        is_or_effect=True,
        child_effects=[
            Effect(CardEffectType.COMBAT, 1),
            Effect(CardEffectType.TRADE, 1),
        ],
    )
    synthetic = Card(
        original.name,
        original.index,
        original.cost,
        [
            Effect(
                CardEffectType.PARENT,
                is_or_effect=True,
                child_effects=[
                    Effect(
                        CardEffectType.PARENT,
                        child_effects=[nested],
                    ),
                    Effect(CardEffectType.HEAL, 1),
                ],
            )
        ],
        CardType.SHIP,
    )
    rules = compile_rules([synthetic, *cards], registry)
    assert not rules.supported[original.index]
    assert "nested OR" in rules.unsupported_reason[original.index]


def test_scalar_ir_interpreter_enforces_ally_requirement(compiled):
    _, registry, rules = compiled
    card = registry.card_index_map["Trade Pod"]
    ally_bundle = 1

    totals = np.zeros(4, dtype=np.int64)
    apply_bundle_to_totals(
        rules,
        card,
        ally_bundle,
        totals,
        faction_counts={1: 1},
    )
    assert totals.tolist() == [0, 0, 0, 0]

    apply_bundle_to_totals(
        rules,
        card,
        ally_bundle,
        totals,
        faction_counts={1: 2},
    )
    assert totals.tolist() == [0, 2, 0, 0]


def test_conditional_draw_compiles_and_executes(compiled):
    _, registry, rules = compiled
    card = registry.card_index_map["Embassy Yacht"]
    bundles = _bundles(rules, "Embassy Yacht")
    conditional = next(
        bundle
        for bundle in bundles
        if any(op[0] == EffectOpcode.DRAW_IF_BASES_GE_2 for op in bundle["ops"])
    )
    bundle_index = bundles.index(conditional)

    totals = np.zeros(4, dtype=np.int64)
    apply_bundle_to_totals(
        rules, card, bundle_index, totals, base_count=1
    )
    assert totals[3] == 0
    apply_bundle_to_totals(
        rules, card, bundle_index, totals, base_count=2
    )
    assert totals[3] == 2


def test_every_supported_bundle_has_valid_operation_ranges(compiled):
    _, _, rules = compiled
    for card in range(rules.num_cards):
        if not rules.supported[card]:
            continue
        operation_count = int(rules.op_count[card])
        for bundle in range(int(rules.bundle_count[card])):
            start = int(rules.bundle_op_start[card, bundle])
            length = int(rules.bundle_op_count[card, bundle])
            assert 0 <= start <= operation_count
            assert 0 < length <= operation_count
            assert start + length <= operation_count


def test_no_single_activation_can_overflow_pending_queue(compiled):
    _, _, rules = compiled
    queue_ops = {
        EffectOpcode.QUEUE_SCRAP,
        EffectOpcode.QUEUE_DESTROY_BASE,
        EffectOpcode.QUEUE_SCRAP_DRAW,
        EffectOpcode.QUEUE_DISCARD_DRAW,
    }
    for card in range(rules.num_cards):
        for bundle in range(int(rules.bundle_count[card])):
            start = int(rules.bundle_op_start[card, bundle])
            length = int(rules.bundle_op_count[card, bundle])
            queued = sum(
                EffectOpcode(int(rules.op_code[card, operation]))
                in queue_ops
                for operation in range(start, start + length)
            )
            assert queued <= 4


def test_compiler_rejects_pending_queue_overflow(compiled):
    cards, registry, _ = compiled
    original = cards[0]
    synthetic = Card(
        original.name,
        original.index,
        original.cost,
        [
            Effect(
                CardEffectType.PARENT,
                child_effects=[
                    Effect(
                        CardEffectType.SCRAP,
                        1,
                        card_targets=["hand"],
                    )
                    for _ in range(5)
                ],
            )
        ],
        CardType.SHIP,
    )
    rules = compile_rules([synthetic, *cards], registry)
    assert not rules.supported[original.index]
    assert "pending queue capacity" in rules.unsupported_reason[original.index]


def test_compiler_rejects_overlapping_pending_domains(compiled):
    cards, registry, _ = compiled
    original = cards[0]
    synthetic = Card(
        original.name,
        original.index,
        original.cost,
        [
            Effect(
                CardEffectType.PARENT,
                child_effects=[
                    Effect(
                        CardEffectType.SCRAP,
                        1,
                        card_targets=["trade"],
                    ),
                    Effect(
                        CardEffectType.SCRAP,
                        1,
                        card_targets=["trade"],
                    ),
                ],
            )
        ],
        CardType.SHIP,
    )
    rules = compile_rules([synthetic, *cards], registry)
    assert not rules.supported[original.index]
    assert "overlapping pending" in rules.unsupported_reason[original.index]


def test_faction_gated_scrap_fails_closed(compiled):
    cards, registry, _ = compiled
    original = cards[0]
    synthetic = Card(
        original.name,
        original.index,
        original.cost,
        [
            Effect(
                CardEffectType.COMBAT,
                3,
                faction_requirement="Blob",
                is_scrap_effect=True,
            )
        ],
        CardType.SHIP,
        faction="Blob",
    )
    rules = compile_rules([synthetic, *cards], registry)
    assert not rules.supported[original.index]
    assert "faction-gated scrap" in rules.unsupported_reason[original.index]


def test_mixed_gate_parent_compiles_independent_trigger_bundles(compiled):
    cards, registry, _ = compiled
    original = cards[0]
    synthetic = Card(
        original.name,
        original.index,
        original.cost,
        [
            Effect(
                CardEffectType.PARENT,
                child_effects=[
                    Effect(CardEffectType.COMBAT, 1),
                    Effect(
                        CardEffectType.COMBAT,
                        4,
                        faction_requirement="Blob",
                    ),
                ],
            )
        ],
        CardType.SHIP,
        faction="Blob",
    )
    rules = compile_rules([synthetic, *cards], registry)
    assert rules.supported[original.index]
    bundles = _bundles(rules, original.name)
    assert [bundle["trigger"] for bundle in bundles] == [
        EffectTrigger.AUTO,
        EffectTrigger.ALLY,
    ]


def test_nested_different_faction_gates_fail_closed(compiled):
    cards, registry, _ = compiled
    original = cards[0]
    synthetic = Card(
        original.name,
        original.index,
        original.cost,
        [
            Effect(
                CardEffectType.PARENT,
                faction_requirement="Blob",
                child_effects=[
                    Effect(
                        CardEffectType.COMBAT,
                        4,
                        faction_requirement="Machine Cult",
                    )
                ],
            )
        ],
        CardType.SHIP,
        faction="Blob",
    )
    rules = compile_rules([synthetic, *cards], registry)
    assert not rules.supported[original.index]
    assert "unsupported conjunction" in rules.unsupported_reason[original.index]


def test_faction_gated_or_fails_closed(compiled):
    cards, registry, _ = compiled
    original = cards[0]
    synthetic = Card(
        original.name,
        original.index,
        original.cost,
        [
            Effect(
                CardEffectType.PARENT,
                faction_requirement="Blob",
                is_or_effect=True,
                child_effects=[
                    Effect(CardEffectType.COMBAT, 1),
                    Effect(CardEffectType.TRADE, 1),
                ],
            )
        ],
        CardType.SHIP,
        faction="Blob",
    )
    rules = compile_rules([synthetic, *cards], registry)
    assert not rules.supported[original.index]
    assert "faction-gated OR" in rules.unsupported_reason[original.index]
