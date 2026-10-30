"""Shared declarative rule IR consumed by scalar and vector backends."""

from src.rules.ir import (
    EffectOpcode,
    EffectTrigger,
    RULE_IR_VERSION,
    RULES_VERSION,
    PENDING_DRAW_ON_COMPLETE,
    PENDING_MANDATORY,
    TARGET_DISCARD,
    TARGET_HAND,
    TARGET_TRADE,
    compile_effect_op,
    compile_effect_trigger,
    effect_defaults_to_auto,
)

__all__ = [
    "EffectOpcode",
    "EffectTrigger",
    "RULE_IR_VERSION",
    "RULES_VERSION",
    "PENDING_DRAW_ON_COMPLETE",
    "PENDING_MANDATORY",
    "TARGET_DISCARD",
    "TARGET_HAND",
    "TARGET_TRADE",
    "compile_effect_op",
    "compile_effect_trigger",
    "effect_defaults_to_auto",
]
