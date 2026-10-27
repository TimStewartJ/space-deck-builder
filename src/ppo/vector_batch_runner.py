"""End-to-end PPO rollout collection on the full vector engine."""
from __future__ import annotations

import json
import random
import time
import logging

import numba
import numpy as np
import torch
from numba import njit, prange

from src.config import ModelConfig, PPOConfig
from src.encoding.action_encoder import get_action_space_size
from src.encoding.state_encoder import get_state_size
from src.ppo.ppo_actor_critic import PPOActorCritic
from src.vector_engine.kernels import (
    END_TURN_INDEX,
    VectorEngine,
    configure_numba_threads,
)
from src.vector_engine.opponents import (
    OPPONENT_CODE,
    OPPONENT_LABEL,
    SNAPSHOT_OFFSET,
    _select_random_legal,
    compile_opponent_policy_data,
    sample_mixed_opponent_kinds,
    select_fixed_actions,
)
from src.ppo.opponent_pool import OpponentPool
from src.vector_engine.policy import (
    CurrentSnapshotEvaluator,
    StackedSnapshotEvaluator,
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
        snapshot_state_dicts: (
            list[tuple[str, dict, ModelConfig | None]] | None
        ) = None,
        self_play_ratio: float = 0.0,
        pfsp_weights: list[float] | None = None,
        max_active_snapshots: int = 3,
        snapshot_temperature: float = 1.0,
    ):
        self.opponent_entries = OpponentPool(
            opponent_spec=opponent_spec
        ).entries
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
        self.snapshot_state_dicts = list(snapshot_state_dicts or ())
        self.self_play_ratio = self_play_ratio
        self.pfsp_weights = (
            list(pfsp_weights) if pfsp_weights is not None else None
        )
        self.max_active_snapshots = max_active_snapshots
        # Inference temperature of every snapshot opponent; 0 plays each
        # snapshot's highest-scoring legal action.
        self.snapshot_temperature = snapshot_temperature
        self.opponent_results: dict[str, list[int]] = {}
        self._batch_index = 0
        self._last_opponent_kinds = np.empty(0, dtype=np.int16)
        self._stacked_snapshot_evaluators: dict[
            tuple[int, ...], StackedSnapshotEvaluator
        ] = {}
        self._joint_snapshot_evaluators: dict[
            int, CurrentSnapshotEvaluator
        ] = {}

    @staticmethod
    def _snapshot_config_key(model_config: ModelConfig | None) -> str:
        return json.dumps(
            (model_config or ModelConfig()).to_dict(),
            sort_keys=True,
        )

    def _snapshot_groups(
        self, active_indices: np.ndarray
    ) -> list[tuple[tuple[int, ...], StackedSnapshotEvaluator]]:
        by_config: dict[str, list[int]] = {}
        for raw_index in active_indices:
            snapshot_index = int(raw_index)
            config = self.snapshot_state_dicts[snapshot_index][2]
            by_config.setdefault(
                self._snapshot_config_key(config), []
            ).append(snapshot_index)
        return [
            (
                tuple(indices),
                self._stacked_snapshot_evaluator(tuple(indices)),
            )
            for indices in by_config.values()
        ]

    def _stacked_snapshot_evaluator(
        self,
        snapshot_indices: tuple[int, ...],
    ) -> StackedSnapshotEvaluator:
        cached = self._stacked_snapshot_evaluators.get(snapshot_indices)
        if cached is not None:
            return cached
        evaluator = StackedSnapshotEvaluator(
            [
                self._load_snapshot_model(snapshot_index)
                for snapshot_index in snapshot_indices
            ],
            self.device,
            temperature=self.snapshot_temperature,
        )
        self._stacked_snapshot_evaluators[snapshot_indices] = evaluator
        return evaluator

    def _load_snapshot_model(self, snapshot_index: int) -> PPOActorCritic:
        state_dim = get_state_size(self.registry.card_names)
        action_dim = get_action_space_size(self.registry.card_names)
        _, state_dict, model_config = self.snapshot_state_dicts[
            snapshot_index
        ]
        model = PPOActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            num_cards=self.registry.num_cards,
            model_config=model_config or ModelConfig(),
            card_registry=self.registry,
        )
        model.load_state_dict(state_dict)
        return model

    def _joint_snapshot_evaluator(
        self, snapshot_index: int
    ) -> CurrentSnapshotEvaluator:
        cached = self._joint_snapshot_evaluators.get(snapshot_index)
        if cached is not None:
            return cached
        evaluator = CurrentSnapshotEvaluator(
            self.model,
            self._load_snapshot_model(snapshot_index),
            self.device,
            snapshot_temperature=self.snapshot_temperature,
        )
        self._joint_snapshot_evaluators[snapshot_index] = evaluator
        return evaluator

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
        batch_seed = self.seed + self._batch_index
        self._batch_index += 1
        state, rules = initialize_training_state(
            num_episodes,
            self.cards,
            self.registry,
            seed=batch_seed,
        )
        engine = VectorEngine(state, rules)
        policy = VectorPolicyEvaluator(self.model, self.device)
        policy_data = compile_opponent_policy_data(
            self.cards, self.registry, rules
        )
        opponent_kinds = sample_mixed_opponent_kinds(
            self.opponent_entries,
            len(self.snapshot_state_dicts),
            self.self_play_ratio,
            self.pfsp_weights,
            num_episodes,
            batch_seed,
            self.max_active_snapshots,
        )
        self._last_opponent_kinds = opponent_kinds.copy()
        active_snapshot_indices = np.unique(
            opponent_kinds[opponent_kinds >= SNAPSHOT_OFFSET]
            - SNAPSHOT_OFFSET
        )
        joint_snapshot = (
            (
                int(active_snapshot_indices[0]),
                self._joint_snapshot_evaluator(
                    int(active_snapshot_indices[0])
                ),
            )
            if len(active_snapshot_indices) == 1
            else None
        )
        snapshot_groups = (
            []
            if joint_snapshot is not None
            else self._snapshot_groups(active_snapshot_indices)
        )

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
            "joint_policy": 0.0,
            "store": 0.0,
            "opponent": 0.0,
            "step": 0.0,
            "iterations": 0,
            "policy_calls": 0,
            "policy_rows": 0,
            "snapshot_policy_calls": 0,
            "snapshot_policy_rows": 0,
            "snapshot_model_batches": 0,
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

            train_obs = observations[training_rows]
            train_masks = masks[training_rows]
            effective_masks = (
                suppress_redundant_end_turn(train_masks)
                if len(training_rows)
                else train_masks
            )
            training_actions = None
            training_log_probs = None
            training_values = None

            fixed_rows = opponent_rows[
                opponent_kinds[opponent_rows] < SNAPSHOT_OFFSET
            ]
            if len(fixed_rows):
                phase = time.perf_counter()
                select_fixed_actions(
                    engine,
                    policy_data,
                    opponent_kinds,
                    fixed_rows,
                    actions,
                    action_variants,
                )
                timing["opponent"] += time.perf_counter() - phase

            joint_rows = np.empty(0, dtype=np.int64)
            if joint_snapshot is not None:
                snapshot_index, joint_evaluator = joint_snapshot
                joint_rows = opponent_rows[
                    opponent_kinds[opponent_rows]
                    == SNAPSHOT_OFFSET + snapshot_index
                ]
            if len(joint_rows):
                if len(training_rows):
                    timing["policy_calls"] += 1
                    timing["policy_rows"] += len(training_rows)
                timing["snapshot_policy_calls"] += 1
                timing["snapshot_model_batches"] += 1
                timing["snapshot_policy_rows"] += len(joint_rows)
                phase = time.perf_counter()
                snapshot_observations = (
                    engine.materialize_current_player_observations(
                        joint_rows
                    )
                )
                decision = joint_evaluator.decide(
                    train_obs,
                    train_masks,
                    snapshot_observations,
                    masks[joint_rows],
                )
                if len(training_rows):
                    training_actions = decision.actions
                    training_log_probs = decision.log_probs
                    training_values = decision.values
                    actions[training_rows] = training_actions
                actions[joint_rows] = decision.snapshot_actions
                timing["joint_policy"] += time.perf_counter() - phase
            elif len(training_rows):
                timing["policy_calls"] += 1
                timing["policy_rows"] += len(training_rows)
                phase = time.perf_counter()
                decision = policy.decide(train_obs, train_masks)
                training_actions = decision.actions
                training_log_probs = decision.log_probs
                training_values = decision.values
                actions[training_rows] = training_actions
                timing["policy"] += time.perf_counter() - phase

            if len(training_rows):
                phase = time.perf_counter()
                count = len(training_rows)
                indices = transition_offset + np.arange(count, dtype=np.int64)
                previous_chunks.append(last_transition[training_rows].copy())
                last_transition[training_rows] = indices
                transition_offset += count
                state_chunks.append(train_obs.copy())
                action_chunks.append(training_actions.copy())
                log_prob_chunks.append(training_log_probs.copy())
                value_chunks.append(training_values.copy())
                mask_chunks.append(effective_masks.copy())
                timing["store"] += time.perf_counter() - phase

            if len(opponent_rows) and snapshot_groups:
                phase = time.perf_counter()
                snapshot_observations = None
                snapshot_rows = opponent_rows[
                    opponent_kinds[opponent_rows] >= SNAPSHOT_OFFSET
                ]
                for snapshot_indices, evaluator in snapshot_groups:
                    row_groups = [
                        opponent_rows[
                            opponent_kinds[opponent_rows]
                            == SNAPSHOT_OFFSET + snapshot_index
                        ]
                        for snapshot_index in snapshot_indices
                    ]
                    nonempty = sum(bool(len(rows)) for rows in row_groups)
                    if not nonempty:
                        continue
                    if snapshot_observations is None:
                        snapshot_observations = (
                            engine.materialize_current_player_observations(
                                snapshot_rows
                            )
                        )
                    timing["snapshot_policy_calls"] += 1
                    timing["snapshot_model_batches"] += nonempty
                    timing["snapshot_policy_rows"] += sum(
                        len(rows) for rows in row_groups
                    )
                    decision = evaluator.decide(
                        [
                            snapshot_observations[
                                np.searchsorted(snapshot_rows, rows)
                            ]
                            for rows in row_groups
                        ],
                        [masks[rows] for rows in row_groups],
                    )
                    for rows, selected_actions in zip(
                        row_groups, decision.actions
                    ):
                        if len(rows):
                            actions[rows] = selected_actions
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

        self.opponent_results = {}
        training_won = state.winner == state.training_player
        for name, _ in self.opponent_entries:
            code = OPPONENT_CODE[name]
            selected = opponent_kinds == code
            self.opponent_results[OPPONENT_LABEL[code]] = [
                int((training_won & selected).sum()),
                int(selected.sum()),
            ]
        for snapshot_index, (
            name,
            _,
            _,
        ) in enumerate(self.snapshot_state_dicts):
            selected = opponent_kinds == SNAPSHOT_OFFSET + snapshot_index
            self.opponent_results[name] = [
                int((training_won & selected).sum()),
                int(selected.sum()),
            ]
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
        snapshot_evaluators = (
            list(self._stacked_snapshot_evaluators.values())
            + list(self._joint_snapshot_evaluators.values())
        )
        snapshot_stats = {
            key: sum(
                evaluator.stats[key]
                for evaluator in snapshot_evaluators
            )
            for key in ("stage_s", "launch_s", "sync_s", "gpu_s")
        }
        _log.info(
            "[VectorRunner] loop=%.2fs mask_obs=%.2fs policy=%.2fs "
            "joint=%.2fs "
            "store=%.2fs opponent=%.2fs step=%.2fs concat=%.2fs "
            "gae=%.2fs transfer=%.2fs iterations=%d calls=%d avg_batch=%.0f "
            "snapshot_calls=%d snapshot_avg_batch=%.0f "
            "snapshot_model_batches=%d snapshot_stage=%.2fs "
            "snapshot_gpu=%.2fs snapshot_sync=%.2fs",
            loop_elapsed,
            timing["mask_obs"],
            timing["policy"],
            timing["joint_policy"],
            timing["store"],
            timing["opponent"],
            timing["step"],
            concatenate_elapsed,
            gae_elapsed,
            transfer_elapsed,
            timing["iterations"],
            timing["policy_calls"],
            timing["policy_rows"] / max(1, timing["policy_calls"]),
            timing["snapshot_policy_calls"],
            timing["snapshot_policy_rows"]
            / max(1, timing["snapshot_policy_calls"]),
            timing["snapshot_model_batches"],
            snapshot_stats["stage_s"],
            snapshot_stats["gpu_s"],
            snapshot_stats["sync_s"],
        )
        return result
