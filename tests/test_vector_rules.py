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
from src.rules.ir import (
    PENDING_MANDATORY,
    RULE_IR_VERSION,
    TARGET_HAND,
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
                    "option": int(rules.bundle_option[card, bundle]),
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
            / f"core_set_rule_ir_v{RULE_IR_VERSION}.json"
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


@pytest.mark.parametrize(
    ("card_name", "first", "second"),
    [
        ("Patrol Mech", (EffectOpcode.ADD_TRADE, 3), (EffectOpcode.ADD_COMBAT, 5)),
        ("Trading Post", (EffectOpcode.ADD_HEALTH, 1), (EffectOpcode.ADD_TRADE, 1)),
        ("Barter World", (EffectOpcode.ADD_HEALTH, 2), (EffectOpcode.ADD_TRADE, 2)),
        ("Defense Center", (EffectOpcode.ADD_HEALTH, 3), (EffectOpcode.ADD_COMBAT, 2)),
        ("Blob World", (EffectOpcode.ADD_COMBAT, 5), (EffectOpcode.DRAW_PER_FACTION, 0)),
        (
            "Recycling Station",
            (EffectOpcode.ADD_TRADE, 1),
            (EffectOpcode.QUEUE_DISCARD_DRAW, 2),
        ),
    ],
)
def test_or_options_compile_to_one_activation_bundle_each(
    compiled, card_name, first, second
):
    """Option 1 is the effect_nonscrap action and option 2 effect_alt.

    A ship's options are chosen in a pending decision as it is played; a
    base's are activated in the main phase.
    """
    _, _, rules = compiled
    card = rules.card_names.index(card_name)
    trigger = (
        EffectTrigger.ON_PLAY_OPTION
        if rules.card_type[card] == 0
        else EffectTrigger.ACTIVATE
    )
    options = {
        int(rules.bundle_option[card, bundle]): bundle
        for bundle in range(int(rules.bundle_count[card]))
        if rules.bundle_option[card, bundle]
    }
    assert sorted(options) == [1, 2]
    bundles = _bundles(rules, card_name)
    for option, expected in ((1, first), (2, second)):
        bundle = options[option]
        assert bundles[bundle]["trigger"] == trigger
        assert [op[:2] for op in bundles[bundle]["ops"]] == [expected]
    assert (
        rules.bundle_source_effect[card, options[1]]
        == rules.bundle_source_effect[card, options[2]]
    )
    other = [
        bundle
        for bundle in range(int(rules.bundle_count[card]))
        if bundle not in options.values()
    ]
    assert all(rules.bundle_option[card, bundle] == 0 for bundle in other)


def _or_ability():
    return Effect(
        CardEffectType.PARENT,
        is_or_effect=True,
        child_effects=[
            Effect(CardEffectType.COMBAT, 1),
            Effect(CardEffectType.TRADE, 1),
        ],
    )


@pytest.mark.parametrize(
    ("card_name", "ops"),
    [
        ("Trade Bot", [(EffectOpcode.ADD_TRADE, 1), (EffectOpcode.QUEUE_SCRAP, 1)]),
        ("Missile Bot", [(EffectOpcode.ADD_COMBAT, 2), (EffectOpcode.QUEUE_SCRAP, 1)]),
        ("Supply Bot", [(EffectOpcode.ADD_TRADE, 2), (EffectOpcode.QUEUE_SCRAP, 1)]),
        ("Battle Mech", [(EffectOpcode.ADD_COMBAT, 4), (EffectOpcode.QUEUE_SCRAP, 1)]),
        ("Battle Pod", [(EffectOpcode.ADD_COMBAT, 4), (EffectOpcode.QUEUE_SCRAP, 1)]),
        (
            "Missile Mech",
            [(EffectOpcode.ADD_COMBAT, 6), (EffectOpcode.QUEUE_DESTROY_BASE, 1)],
        ),
    ],
)
def test_ship_primary_choices_resolve_when_played(compiled, card_name, ops):
    """A ship's whole primary ability is one bundle applied on play."""
    _, _, rules = compiled
    bundles = _bundles(rules, card_name)
    assert bundles[0]["trigger"] == EffectTrigger.AUTO
    assert [operation[:2] for operation in bundles[0]["ops"]] == ops
    assert all(
        bundle["trigger"] in (EffectTrigger.ALLY, EffectTrigger.SCRAP)
        for bundle in bundles[1:]
    )


def test_machine_base_draws_and_scraps_in_one_activation(compiled):
    """Its draw and the scrap that must follow are one activated bundle."""
    _, _, rules = compiled
    assert _bundles(rules, "Machine Base") == [
        {
            "trigger": EffectTrigger.ACTIVATE,
            "faction": 0,
            "requirement": 0,
            "ops": [
                (EffectOpcode.DRAW, 1, 0),
                (EffectOpcode.QUEUE_SCRAP, 1, TARGET_HAND | PENDING_MANDATORY),
            ],
        }
    ]


def test_compiler_rejects_an_or_ability_below_the_top_level(compiled):
    cards, registry, _ = compiled
    original = cards[0]
    synthetic = Card(
        original.name,
        original.index,
        original.cost,
        [
            Effect(
                CardEffectType.PARENT,
                child_effects=[Effect(CardEffectType.HEAL, 1), _or_ability()],
            )
        ],
        CardType.SHIP,
    )
    rules = compile_rules([synthetic, *cards], registry)
    assert not rules.supported[original.index]
    assert "top-level" in rules.unsupported_reason[original.index]


def test_compiler_rejects_two_or_abilities_on_one_card(compiled):
    cards, registry, _ = compiled
    original = cards[0]
    synthetic = Card(
        original.name,
        original.index,
        original.cost,
        [_or_ability(), _or_ability()],
        CardType.BASE,
        defense=3,
    )
    rules = compile_rules([synthetic, *cards], registry)
    assert not rules.supported[original.index]
    assert "at most one OR" in rules.unsupported_reason[original.index]


def test_compiler_rejects_a_faction_gated_or_option(compiled):
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
                child_effects=[
                    Effect(CardEffectType.COMBAT, 1),
                    Effect(CardEffectType.TRADE, 1, faction_requirement="Blob"),
                ],
            )
        ],
        CardType.BASE,
        defense=3,
        faction="Blob",
    )
    rules = compile_rules([synthetic, *cards], registry)
    assert not rules.supported[original.index]
    assert "faction-gated OR options" in rules.unsupported_reason[original.index]


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
