"""Shared declarative rule IR consumed by scalar and vector backends."""

from src.rules.ir import (
    EffectOpcode,
    EffectTrigger,
    PlayResolution,
    RULE_IR_VERSION,
    RULES_VERSION,
    PENDING_DRAW_ON_COMPLETE,
    PENDING_MANDATORY,
    TARGET_DISCARD,
    TARGET_HAND,
    TARGET_TRADE,
    ability_play_resolution,
    compile_effect_op,
    compile_effect_trigger,
)

__all__ = [
    "EffectOpcode",
    "EffectTrigger",
    "PlayResolution",
    "RULE_IR_VERSION",
    "RULES_VERSION",
    "PENDING_DRAW_ON_COMPLETE",
    "PENDING_MANDATORY",
    "TARGET_DISCARD",
    "TARGET_HAND",
    "TARGET_TRADE",
    "ability_play_resolution",
    "compile_effect_op",
    "compile_effect_trigger",
]
