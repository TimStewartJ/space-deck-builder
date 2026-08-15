"""Compiler-backed struct-of-arrays engine.

Gate 2 starts with a rules compiler and a rules-correct Numba SPMD vertical
slice. The existing object engine remains the external oracle until every card
and transition is differential-tested.
"""

from src.vector_engine.rules import (
    CompiledRules,
    EffectOpcode,
    EffectTrigger,
    compile_rules,
)

__all__ = [
    "CompiledRules",
    "EffectOpcode",
    "EffectTrigger",
    "compile_rules",
]
