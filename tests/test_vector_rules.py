"""Effect-object to vector-rule IR compiler tests."""
from __future__ import annotations

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
    _, _, rules = compiled
    expected = {
        "Battle Blob",
        "Blob Carrier",
        "Blob Fighter",
        "Blob Wheel",
        "Mothership",
        "Ram",
        "The Hive",
        "Trade Pod",
        "Battle Station",
        "Mech World",
        "Stealth Needle",
        "Corvette",
        "Dreadnaught",
        "Fleet HQ",
        "Space Station",
        "War World",
        "Central Office",
        "Cutter",
        "Embassy Yacht",
        "Federation Shuttle",
        "Flagship",
        "Freighter",
        "Trade Escort",
        "Scout",
        "Viper",
        "Explorer",
    }
    assert set(rules.supported_names) == expected
    assert len(rules.supported_names) == 26


@pytest.mark.parametrize(
    ("card", "reason_fragment"),
    [
        ("Battle Pod", "target selection"),
        ("Blob Destroyer", "target selection"),
        ("Blob World", "OR effects"),
        ("Brain World", "target selection"),
        ("Imperial Fighter", "target selection"),
        ("Recycling Station", "OR effects"),
    ],
)
def test_unsupported_cards_fail_closed(compiled, card, reason_fragment):
    _, registry, rules = compiled
    index = registry.card_index_map[card]
    assert not rules.supported[index]
    assert reason_fragment in rules.unsupported_reason[index]


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


def test_complex_effects_compile_to_current_noop_semantics(compiled):
    _, _, rules = compiled
    bundles = _bundles(rules, "Mech World")
    assert bundles == [
        {
            "trigger": EffectTrigger.ACTIVATE,
            "faction": 0,
            "requirement": 0,
            "ops": [(EffectOpcode.NOP, 0, 0)],
        }
    ]


def test_explorer_scrap_bundle_is_separate(compiled):
    _, _, rules = compiled
    bundles = _bundles(rules, "Explorer")
    assert bundles[0]["trigger"] == EffectTrigger.AUTO
    assert bundles[0]["ops"] == [(EffectOpcode.ADD_TRADE, 2, 0)]
    assert bundles[1]["trigger"] == EffectTrigger.SCRAP
    assert bundles[1]["ops"] == [(EffectOpcode.ADD_COMBAT, 2, 0)]


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


def test_mixed_gate_parent_fails_closed(compiled):
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
    assert not rules.supported[original.index]
    assert "mixed-gate parent" in rules.unsupported_reason[original.index]


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
