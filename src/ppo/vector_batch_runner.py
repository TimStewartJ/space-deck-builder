"""End-to-end PPO rollout collection on the full vector engine.

The guarded Gate 3 path currently supports the random fixed opponent. Other
opponents fail explicitly rather than silently changing their semantics.
"""
from __future__ import annotations

import random
import time
import logging

import numba
import numpy as np
import torch
from numba import njit, prange

from src.config import DataConfig, PPOConfig
from src.encoding.action_encoder import get_action_space_size
from src.encoding.state_encoder import get_state_size
from src.ppo.ppo_actor_critic import PPOActorCritic
from src.vector_engine.kernels import (
    ATTACK_PLAYER_INDEX,
    END_TURN_INDEX,
    NUM_CARDS,
    PLAY_OFFSET,
    VectorEngine,
    _advance_rng,
    configure_numba_threads,
)
from src.vector_engine.policy import (
    VectorPolicyEvaluator,
    suppress_redundant_end_turn,
)
from src.vector_engine.rules import CompiledRules, compile_rules
from src.vector_engine.state import (
    NO_CARD,
    VectorState,
    ZONE_DECK,
    ZONE_HAND,
)

_log = logging.getLogger("training")
@njit(parallel=True, cache=True)
def _select_random_legal(
    weights, rng_state, selected_rows, actions, action_variants
):
    for position in prange(selected_rows.shape[0]):
        game = selected_rows[position]
        play_count = 0
        for action in range(PLAY_OFFSET, PLAY_OFFSET + NUM_CARDS):
            play_count += weights[game, action]
        if play_count > 0:
            next_rng = _advance_rng(rng_state[game])
            rng_state[game] = next_rng
            wanted = next_rng % play_count
            seen = 0
            for action in range(PLAY_OFFSET, PLAY_OFFSET + NUM_CARDS):
                weight = weights[game, action]
                if weight > 0:
                    if seen + weight > wanted:
                        actions[game] = action
                        action_variants[game] = wanted - seen
                        break
                    seen += weight
            continue
        if weights[game, ATTACK_PLAYER_INDEX] > 0:
            actions[game] = ATTACK_PLAYER_INDEX
            action_variants[game] = 0
            continue
        count = 0
        for action in range(weights.shape[1]):
            count += weights[game, action]
        if count == 0:
            actions[game] = END_TURN_INDEX
            continue
        next_rng = _advance_rng(rng_state[game])
        rng_state[game] = next_rng
        wanted = next_rng % count
        seen = 0
        for action in range(weights.shape[1]):
            weight = weights[game, action]
            if weight > 0:
                if seen + weight > wanted:
                    actions[game] = action
                    action_variants[game] = wanted - seen
                    break
                seen += weight


@njit(parallel=True, cache=True)
def _compute_linked_gae(
    last_transition,
    previous,
    rewards,
    dones,
    values,
    gamma,
    lam,
    returns,
    advantages,
):
    for game in prange(last_transition.shape[0]):
        index = last_transition[game]
        gae = 0.0
        next_value = 0.0
        while index >= 0:
            not_done = 0.0 if dones[index] else 1.0
            delta = (
                rewards[index]
                + gamma * next_value * not_done
                - values[index]
            )
            gae = delta + gamma * lam * not_done * gae
            advantages[index] = gae
            returns[index] = gae + values[index]
            next_value = values[index]
            index = previous[index]


def _sample_without_replacement(rng, counts, draws):
    result = np.zeros_like(counts)
    work = counts.copy()
    for _ in range(draws):
        total = int(work.sum())
        if total <= 0:
            break
        ticket = int(rng.integers(total))
        cumulative = 0
        for card, count in enumerate(work):
            cumulative += int(count)
            if cumulative > ticket:
                work[card] -= 1
                result[card] += 1
                break
    return result


def initialize_training_state(
    episodes: int,
    cards,
    registry,
    *,
    seed: int,
) -> tuple[VectorState, CompiledRules]:
    """Create physical-deck-faithful independent vector games."""
    rules = compile_rules(cards, registry)
    state = VectorState.empty(episodes, rules.num_cards)
    rng = np.random.default_rng(seed)
    scout = registry.card_index_map["Scout"]
    viper = registry.card_index_map["Viper"]

    physical_trade = np.zeros(rules.num_cards, dtype=np.int16)
    for card in cards:
        physical_trade[card.index] += 1

    for game in range(episodes):
        first = int(rng.integers(2))
        training = int(rng.integers(2))
        state.first_player[game] = first
        state.current_player[game] = first
        state.training_player[game] = training
        state.health[game] = 50
        state.explorer_count[game] = 10

        for player in range(2):
            starter = np.zeros(rules.num_cards, dtype=np.int16)
            starter[scout] = 8
            starter[viper] = 2
            hand_size = 3 if player == first else 5
            hand = _sample_without_replacement(rng, starter, hand_size)
            state.zones[game, player, ZONE_HAND] = hand
            state.zones[game, player, ZONE_DECK] = starter - hand

        market = _sample_without_replacement(rng, physical_trade, 5)
        remaining = physical_trade - market
        state.trade_deck[game] = remaining
        expanded = np.repeat(np.arange(rules.num_cards), market)
        rng.shuffle(expanded)
        state.trade_row[game, : len(expanded)] = expanded
        state.rng_state[game] = int(rng.integers(1, 0x7FFF_FFFF))

    return state, rules


class VectorBatchRunner:
    """PPO-compatible rollout runner using one Numba SoA population."""

    def __init__(
        self,
        model: PPOActorCritic,
        cards,
        registry,
        device: torch.device,
        ppo_config: PPOConfig,
        opponent_spec: str,
        seed: int | None = None,
        num_threads: int = 4,
        num_concurrent: int | None = None,
    ):
        normalized = opponent_spec.strip().lower()
        if normalized not in {"random", "random:1", "random:1.0"}:
            raise ValueError(
                "vector engine currently supports only --opponents random; "
                f"got {opponent_spec!r}"
            )
        self.model = model
        self.cards = cards
        self.registry = registry
        self.device = device
        self.ppo_config = ppo_config
        self.seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        self.num_threads = min(
            num_threads, numba.config.NUMBA_NUM_THREADS
        )
        self.num_concurrent = num_concurrent
        self.opponent_results: dict[str, list[int]] = {}

    def run_episodes(self, num_episodes: int) -> tuple:
        if (
            self.num_concurrent is not None
            and self.num_concurrent != num_episodes
        ):
            raise ValueError(
                "vector engine currently runs one resident population; "
                "--num-concurrent must be omitted or equal --episodes"
            )
        # The rollout alternates many short CPU kernels with GPU inference.
        # More OpenMP workers make standalone kernels faster but spin while
        # Python launches GPU work; four preserves near-linear CPU scaling
        # without starving the policy thread.
        configure_numba_threads(self.num_threads)
        state, rules = initialize_training_state(
            num_episodes,
            self.cards,
            self.registry,
            seed=self.seed,
        )
        engine = VectorEngine(state, rules)
        policy = VectorPolicyEvaluator(self.model, self.device)

        state_chunks = []
        action_chunks = []
        log_prob_chunks = []
        value_chunks = []
        mask_chunks = []
        previous_chunks = []
        last_transition = np.full(num_episodes, -1, dtype=np.int64)
        transition_offset = 0
        safety_limit = int(state.turn_cap.max()) * 200
        timing = {
            "mask_obs": 0.0,
            "policy": 0.0,
            "store": 0.0,
            "opponent": 0.0,
            "step": 0.0,
            "iterations": 0,
            "policy_calls": 0,
            "policy_rows": 0,
        }
        loop_started = time.perf_counter()

        for _ in range(safety_limit):
            timing["iterations"] += 1
            if state.done.all():
                break
            phase = time.perf_counter()
            masks = engine.build_masks()
            observations = engine.materialize_observations()
            timing["mask_obs"] += time.perf_counter() - phase
            active = ~state.done
            training_rows = np.flatnonzero(
                active & (state.current_player == state.training_player)
            )
            opponent_rows = np.flatnonzero(
                active & (state.current_player != state.training_player)
            )
            actions = np.full(
                num_episodes, END_TURN_INDEX, dtype=np.int32
            )
            action_variants = np.zeros(num_episodes, dtype=np.int16)

            if len(training_rows):
                timing["policy_calls"] += 1
                timing["policy_rows"] += len(training_rows)
                phase = time.perf_counter()
                train_obs = observations[training_rows]
                train_masks = masks[training_rows]
                effective_masks = suppress_redundant_end_turn(train_masks)
                decision = policy.decide(train_obs, train_masks)
                actions[training_rows] = decision.actions
                timing["policy"] += time.perf_counter() - phase

                phase = time.perf_counter()
                count = len(training_rows)
                indices = transition_offset + np.arange(count, dtype=np.int64)
                previous_chunks.append(last_transition[training_rows].copy())
                last_transition[training_rows] = indices
                transition_offset += count
                state_chunks.append(train_obs.copy())
                action_chunks.append(decision.actions.copy())
                log_prob_chunks.append(decision.log_probs.copy())
                value_chunks.append(decision.values.copy())
                mask_chunks.append(effective_masks.copy())
                timing["store"] += time.perf_counter() - phase

            if len(opponent_rows):
                phase = time.perf_counter()
                action_weights = engine.build_action_weights()
                _select_random_legal(
                    action_weights,
                    state.rng_state,
                    opponent_rows.astype(np.int64),
                    actions,
                    action_variants,
                )
                timing["opponent"] += time.perf_counter() - phase

            phase = time.perf_counter()
            engine.step(actions, action_variants=action_variants)
            timing["step"] += time.perf_counter() - phase
        else:
            raise RuntimeError("vector rollout exceeded safety limit")

        if not state.done.all():
            raise RuntimeError("vector rollout ended with active games")
        if not state_chunks:
            raise RuntimeError("vector rollout collected no policy decisions")

        loop_elapsed = time.perf_counter() - loop_started
        post_started = time.perf_counter()
        states = np.concatenate(state_chunks)
        actions_np = np.concatenate(action_chunks)
        old_lp = np.concatenate(log_prob_chunks)
        values = np.concatenate(value_chunks)
        masks_np = np.concatenate(mask_chunks)
        previous = np.concatenate(previous_chunks)
        concatenate_elapsed = time.perf_counter() - post_started
        rewards = np.zeros(len(states), dtype=np.float32)
        dones = np.zeros(len(states), dtype=np.bool_)
        for game in range(num_episodes):
            last = last_transition[game]
            if last < 0:
                continue
            rewards[last] = (
                1.0 if state.winner[game] == state.training_player[game] else -1.0
            )
            dones[last] = True

        returns = np.empty(len(states), dtype=np.float32)
        advantages = np.empty(len(states), dtype=np.float32)
        gae_started = time.perf_counter()
        _compute_linked_gae(
            last_transition,
            previous,
            rewards,
            dones,
            values,
            self.ppo_config.gamma,
            self.ppo_config.lam,
            returns,
            advantages,
        )
        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std() + 1e-8)
        gae_elapsed = time.perf_counter() - gae_started

        wins = int((state.winner == state.training_player).sum())
        self.opponent_results = {"Random": [wins, num_episodes]}
        transfer_started = time.perf_counter()
        result = (
            torch.from_numpy(states).to(self.device),
            torch.from_numpy(actions_np.astype(np.int64)).to(self.device),
            torch.from_numpy(old_lp).to(self.device),
            torch.from_numpy(returns).to(self.device),
            torch.from_numpy(advantages).to(self.device),
            torch.from_numpy(masks_np).to(self.device),
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        transfer_elapsed = time.perf_counter() - transfer_started
        _log.info(
            "[VectorRunner] loop=%.2fs mask_obs=%.2fs policy=%.2fs "
            "store=%.2fs opponent=%.2fs step=%.2fs concat=%.2fs "
            "gae=%.2fs transfer=%.2fs iterations=%d calls=%d avg_batch=%.0f",
            loop_elapsed,
            timing["mask_obs"],
            timing["policy"],
            timing["store"],
            timing["opponent"],
            timing["step"],
            concatenate_elapsed,
            gae_elapsed,
            transfer_elapsed,
            timing["iterations"],
            timing["policy_calls"],
            timing["policy_rows"] / max(1, timing["policy_calls"]),
        )
        return result
