"""Gate 2 rules-engine, observation, and integrated-policy benchmark.

Run detached at Normal priority. The engine-only gate includes real compiled
rules, legal masks, 505-float observations, and state transitions. The
integrated gate additionally calls the actual PPO model and copies sampled
actions back to the CPU engine.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from src.config import DataConfig, ModelConfig
from src.encoding.action_encoder import get_action_space_size
from src.encoding.state_encoder import get_state_size
from src.ppo.ppo_actor_critic import PPOActorCritic
from src.vector_engine.kernels import (
    ACTION_DIM,
    ATTACK_BASE_OFFSET,
    ATTACK_PLAYER_INDEX,
    BUY_OFFSET,
    EFFECT_NONSCRAP_OFFSET,
    EFFECT_SCRAP_OFFSET,
    END_TURN_INDEX,
    PLAY_OFFSET,
    VectorEngine,
    configure_numba_threads,
)
from src.vector_engine.policy import VectorPolicyEvaluator
from src.vector_engine.rules import compile_rules
from src.vector_engine.state import (
    INSTANCE_ACTIVE,
    INSTANCE_PLAYED_THIS_TURN,
    VectorState,
    ZONE_BASES,
    ZONE_DECK,
    ZONE_DISCARD,
    ZONE_HAND,
    ZONE_PLAYED,
)
from src.vector_engine.rules import EffectTrigger


def make_state(games, rules, registry):
    state = VectorState.empty(games, rules.num_cards)
    state.health[:] = 300  # keep the benchmark population alive
    state.trade[:] = 3
    state.combat[:] = 10
    state.training_player[:] = 0
    state.current_player[:] = 0
    state.first_player[:] = 0
    state.explorer_count[:] = 10

    scout = registry.card_index_map["Scout"]
    viper = registry.card_index_map["Viper"]
    trade_pod = registry.card_index_map["Trade Pod"]
    blob_fighter = registry.card_index_map["Blob Fighter"]
    blob_wheel = registry.card_index_map["Blob Wheel"]
    mothership = registry.card_index_map["Mothership"]
    state.zones[:, :, ZONE_HAND, scout] = 2
    state.zones[:, :, ZONE_HAND, viper] = 1
    state.zones[:, :, ZONE_HAND, trade_pod] = 1
    state.zones[:, :, ZONE_HAND, blob_fighter] = 1
    state.zones[:, :, ZONE_DECK, scout] = 5
    state.zones[:, :, ZONE_DECK, viper] = 2
    state.zones[:, :, ZONE_DECK, mothership] = 1
    state.zones[:, :, ZONE_DISCARD, scout] = 3

    # Seed the explicit-effect and attack branches. They need only remain
    # nonzero; the benchmark's correctness gates, not this synthetic mix,
    # establish semantic fidelity.
    explorer = registry.card_index_map["Explorer"]
    state.zones[:, 0, ZONE_PLAYED, trade_pod] = 1
    state.zones[:, 0, ZONE_PLAYED, blob_fighter] = 1
    state.zones[:, 0, ZONE_PLAYED, explorer] = 1
    # Established non-outpost bases expose ATTACK_BASE while still allowing
    # ATTACK_PLAYER. Copies keep those branches live over a long synthetic run.
    state.zones[:, 0, ZONE_BASES, blob_wheel] = 4
    state.zones[:, 1, ZONE_BASES, blob_wheel] = 4
    for bundle in range(int(rules.bundle_count[trade_pod])):
        if rules.bundle_trigger[trade_pod, bundle] == int(EffectTrigger.ALLY):
            state.effect_state[:, 0, trade_pod, 0] |= (
                np.uint16(1 << bundle)
            )
    state.effect_state[:, 0, trade_pod, 0] |= (
        INSTANCE_ACTIVE | INSTANCE_PLAYED_THIS_TURN
    )
    for bundle in range(int(rules.bundle_count[explorer])):
        if rules.bundle_trigger[explorer, bundle] == int(EffectTrigger.SCRAP):
            state.effect_state[:, 0, explorer, 0] |= np.uint16(1 << bundle)
    state.effect_state[:, 0, explorer, 0] |= (
        INSTANCE_ACTIVE | INSTANCE_PLAYED_THIS_TURN
    )
    ready_bits = int(INSTANCE_ACTIVE)
    for bundle in range(int(rules.bundle_count[blob_wheel])):
        ready_bits |= 1 << bundle
    state.effect_state[:, 0, blob_wheel, :4] = np.uint16(ready_bits)
    state.effect_state[:, 1, blob_wheel, :4] = np.uint16(ready_bits)

    supported = np.flatnonzero(rules.supported)
    game = np.arange(games)[:, None]
    slot = np.arange(5)[None, :]
    state.trade_row[:] = supported[
        (game * 7 + slot * 11) % len(supported)
    ]
    state.trade_deck[:, supported] = 3
    return state


def choose_actions(mask, step, num_cards):
    actions = np.full(mask.shape[0], END_TURN_INDEX, dtype=np.int32)
    game = np.arange(mask.shape[0])
    phase = (game + step) % 7
    ranges = (
        (PLAY_OFFSET, PLAY_OFFSET + num_cards),
        (EFFECT_NONSCRAP_OFFSET, EFFECT_NONSCRAP_OFFSET + num_cards),
        (BUY_OFFSET, BUY_OFFSET + num_cards),
        (ATTACK_BASE_OFFSET, ATTACK_BASE_OFFSET + num_cards),
        (ATTACK_PLAYER_INDEX, ATTACK_PLAYER_INDEX + 1),
        (EFFECT_SCRAP_OFFSET, EFFECT_SCRAP_OFFSET + num_cards),
    )
    chosen = phase == 6
    for target_phase, (start, end) in enumerate(ranges):
        candidates = mask[:, start:end]
        selected = (phase == target_phase) & candidates.any(axis=1)
        first = candidates.argmax(axis=1)
        actions[selected] = start + first[selected]
        chosen |= selected

    # Do not turn a missing target branch into END_TURN: use the first legal
    # non-end branch, preserving real work throughout the 200-step gate.
    for start, end in ranges:
        candidates = mask[:, start:end]
        selected = ~chosen & candidates.any(axis=1)
        first = candidates.argmax(axis=1)
        actions[selected] = start + first[selected]
        chosen |= selected
    return actions


def build_schedule(initial, rules, steps):
    engine = VectorEngine(initial.copy(), rules)
    actions = np.empty((steps, initial.games), dtype=np.int32)
    counts = {}
    for step in range(steps):
        mask = engine.build_masks()
        action = choose_actions(mask, step, rules.num_cards)
        actions[step] = action
        for name, start, end in (
            ("play", PLAY_OFFSET, PLAY_OFFSET + rules.num_cards),
            ("effect", EFFECT_NONSCRAP_OFFSET, EFFECT_NONSCRAP_OFFSET + rules.num_cards),
            ("buy", BUY_OFFSET, BUY_OFFSET + rules.num_cards),
            ("attack_base", ATTACK_BASE_OFFSET, ATTACK_BASE_OFFSET + rules.num_cards),
            ("attack_player", ATTACK_PLAYER_INDEX, ATTACK_PLAYER_INDEX + 1),
            ("scrap", EFFECT_SCRAP_OFFSET, EFFECT_SCRAP_OFFSET + rules.num_cards),
            ("end", END_TURN_INDEX, END_TURN_INDEX + 1),
        ):
            counts[name] = counts.get(name, 0) + int(
                ((action >= start) & (action < end)).sum()
            )
        engine.step(action)
    decisions = initial.games * steps
    minimum_rates = {
        "play": 0.08,
        "effect": 0.01,
        "buy": 0.02,
        "attack_base": 0.01,
        "attack_player": 0.01,
        "scrap": 0.002,
        "end": 0.08,
    }
    failures = [
        f"{name}={counts.get(name, 0) / decisions:.2%} < {minimum:.2%}"
        for name, minimum in minimum_rates.items()
        if counts.get(name, 0) / decisions < minimum
    ]
    if failures:
        raise RuntimeError(
            "Gate 2 workload degenerated: " + "; ".join(failures)
        )
    return actions, counts


def benchmark_engine(initial, rules, actions, repetitions):
    samples = []
    checksums = []
    for _ in range(repetitions):
        engine = VectorEngine(initial.copy(), rules)
        started = time.perf_counter()
        for action in actions:
            engine.build_masks()
            engine.materialize_observations()
            engine.step(action)
        elapsed = time.perf_counter() - started
        samples.append(initial.games * len(actions) / elapsed)
        checksums.append(
            int(engine.state.zones.sum())
            + int(engine.observations.sum())
            + int(engine.masks.sum())
        )
    if len(set(checksums)) != 1:
        raise RuntimeError(f"engine benchmark diverged: {checksums}")
    return samples


def benchmark_integrated(initial, rules, registry, steps, repetitions):
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm/CUDA unavailable")
    samples = []
    action_profile = {
        "play": 0,
        "effect": 0,
        "buy": 0,
        "attack_base": 0,
        "attack_player": 0,
        "scrap": 0,
        "end": 0,
    }
    for _ in range(repetitions):
        engine = VectorEngine(initial.copy(), rules)
        model = PPOActorCritic(
            state_dim=get_state_size(registry.card_names),
            action_dim=get_action_space_size(registry.card_names),
            num_cards=rules.num_cards,
            model_config=ModelConfig(),
        )
        policy = VectorPolicyEvaluator(model, torch.device("cuda"))
        # Warm allocations and first kernels.
        mask = engine.build_masks()
        obs = engine.materialize_observations()
        policy.decide(obs, mask)

        started = time.perf_counter()
        for _ in range(steps):
            mask = engine.build_masks()
            obs = engine.materialize_observations()
            decision = policy.decide(obs, mask)
            action = decision.actions
            action_profile["play"] += int(
                ((action >= PLAY_OFFSET) & (action < PLAY_OFFSET + rules.num_cards)).sum()
            )
            action_profile["effect"] += int(
                (
                    (action >= EFFECT_NONSCRAP_OFFSET)
                    & (action < EFFECT_NONSCRAP_OFFSET + rules.num_cards)
                ).sum()
            )
            action_profile["buy"] += int(
                ((action >= BUY_OFFSET) & (action < BUY_OFFSET + rules.num_cards)).sum()
            )
            action_profile["attack_base"] += int(
                (
                    (action >= ATTACK_BASE_OFFSET)
                    & (action < ATTACK_BASE_OFFSET + rules.num_cards)
                ).sum()
            )
            action_profile["attack_player"] += int(
                (action == ATTACK_PLAYER_INDEX).sum()
            )
            action_profile["scrap"] += int(
                (
                    (action >= EFFECT_SCRAP_OFFSET)
                    & (action < EFFECT_SCRAP_OFFSET + rules.num_cards)
                ).sum()
            )
            action_profile["end"] += int(
                (action == END_TURN_INDEX).sum()
            )
            engine.step(decision.actions)
        elapsed = time.perf_counter() - started
        samples.append(initial.games * steps / elapsed)
    return samples, action_profile


def summarize(values):
    return {
        "median_decisions_per_sec": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "repetitions": values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=16000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--integrated-steps", type=int, default=40)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    configure_numba_threads(16)
    data = DataConfig()
    cards = data.load_cards()
    registry = data.build_registry(cards)
    rules = compile_rules(cards, registry)
    initial = make_state(args.games, rules, registry)
    actions, profile = build_schedule(initial, rules, args.steps)

    # Warm all Numba kernels before timing.
    warm = VectorEngine(make_state(64, rules, registry), rules)
    warm.build_masks()
    warm.materialize_observations()
    warm.step(np.full(64, END_TURN_INDEX, dtype=np.int32))

    engine = benchmark_engine(initial, rules, actions, args.reps)
    integrated, integrated_profile = benchmark_integrated(
        initial,
        rules,
        registry,
        args.integrated_steps,
        args.reps,
    )
    payload = {
        "games": args.games,
        "steps": args.steps,
        "integrated_steps": args.integrated_steps,
        "threads": 16,
        "supported_cards": list(rules.supported_names),
        "supported_count": len(rules.supported_names),
        "action_profile": profile,
        "engine_mask_observation": summarize(engine),
        "integrated_actual_policy": summarize(integrated),
        "integrated_action_profile": integrated_profile,
    }
    print(json.dumps(payload, indent=2), flush=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
