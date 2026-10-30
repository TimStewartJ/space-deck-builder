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
from src.vector_engine.graphed_policy import (
    GraphedPolicyBank,
    architecture_key,
    graphable,
    staged_rows,
)
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

# Learner decisions per episode that the graphed runner's device storage is
# first sized for (reference-recipe games average about 146); storage grows
# when a rollout needs more.
_INITIAL_DECISIONS_PER_EPISODE = 160


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
    rules: CompiledRules | None = None,
) -> tuple[VectorState, CompiledRules]:
    """Create physical-deck-faithful independent vector games.

    ``rules`` reuses an earlier :func:`compile_rules` result for the same
    cards; compiled rules are read-only.
    """
    if rules is None:
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
        graph_inference: bool = True,
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
        # Evaluate the learner and snapshots by CUDA/HIP graph replay when
        # the device and every model's architecture allow it; otherwise use
        # the eager evaluators below.
        self.graph_inference = graph_inference
        self.opponent_results: dict[str, list[int]] = {}
        self._batch_index = 0
        self._last_opponent_kinds = np.empty(0, dtype=np.int16)
        self._stacked_snapshot_evaluators: dict[
            tuple[int, ...], StackedSnapshotEvaluator
        ] = {}
        self._joint_snapshot_evaluators: dict[
            int, CurrentSnapshotEvaluator
        ] = {}
        # Compiled once and reused by every rollout of this runner.
        self._rules: CompiledRules | None = None
        self._policy_data = None
        # Graph banks persist across rollouts so captured graphs are reused.
        self._learner_bank: GraphedPolicyBank | None = None
        self._snapshot_bank: GraphedPolicyBank | None = None

    def reconfigure(
        self,
        *,
        snapshot_state_dicts: (
            list[tuple[str, dict, ModelConfig | None]] | None
        ),
        self_play_ratio: float,
        pfsp_weights: list[float] | None,
    ) -> None:
        """Set the next rollout's opponents as a freshly built runner would.

        The trainer calls this once per update instead of building a new
        runner, which keeps the compiled rules and captured graphs. Like a
        new runner without an explicit seed, it draws the next rollout seed
        from Python's ``random`` module.
        """
        self.snapshot_state_dicts = list(snapshot_state_dicts or ())
        self.self_play_ratio = self_play_ratio
        self.pfsp_weights = (
            list(pfsp_weights) if pfsp_weights is not None else None
        )
        self.seed = random.randint(0, 2**31 - 1)
        self._batch_index = 0
        self._stacked_snapshot_evaluators.clear()
        self._joint_snapshot_evaluators.clear()

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
            rules=self._rules,
        )
        self._rules = rules
        engine = VectorEngine(state, rules)
        if self._policy_data is None:
            self._policy_data = compile_opponent_policy_data(
                self.cards, self.registry, rules
            )
        policy_data = self._policy_data
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
        snapshot_models = self._graphable_snapshots(active_snapshot_indices)
        if snapshot_models is not None:
            return self._run_graphed(
                num_episodes,
                state,
                engine,
                policy_data,
                opponent_kinds,
                active_snapshot_indices,
                snapshot_models,
            )
        policy = VectorPolicyEvaluator(self.model, self.device)
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
        returns, advantages, gae_elapsed = self._returns_and_advantages(
            state, last_transition, previous, values
        )
        self._record_opponent_results(state, opponent_kinds)
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

    def _returns_and_advantages(
        self,
        state: VectorState,
        last_transition: np.ndarray,
        previous: np.ndarray,
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Terminal +/-1 rewards, linked GAE returns, normalized advantages."""
        count = len(values)
        rewards = np.zeros(count, dtype=np.float32)
        dones = np.zeros(count, dtype=np.bool_)
        games = np.flatnonzero(last_transition >= 0)
        last = last_transition[games]
        rewards[last] = np.where(
            state.winner[games] == state.training_player[games], 1.0, -1.0
        )
        dones[last] = True

        returns = np.empty(count, dtype=np.float32)
        advantages = np.empty(count, dtype=np.float32)
        started = time.perf_counter()
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
        return returns, advantages, time.perf_counter() - started

    def _record_opponent_results(
        self, state: VectorState, opponent_kinds: np.ndarray
    ) -> None:
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

    def _graphable_snapshots(
        self, active_snapshot_indices: np.ndarray
    ) -> list[PPOActorCritic] | None:
        """Active snapshot models for graphed inference, or ``None`` to use
        the eager evaluators (non-CUDA device, graphs disabled, or an
        architecture the graphs do not evaluate)."""
        if (
            not self.graph_inference
            or self.device.type != "cuda"
            or not graphable(self.model)
        ):
            return None
        key = architecture_key(self.model)
        models = []
        for raw_index in active_snapshot_indices:
            model = self._load_snapshot_model(int(raw_index))
            if not graphable(model) or architecture_key(model) != key:
                return None
            models.append(model)
        return models

    def _graph_banks(
        self, num_episodes: int, snapshot_slots: int
    ) -> tuple[GraphedPolicyBank, GraphedPolicyBank | None]:
        key = architecture_key(self.model)
        learner = self._learner_bank
        if (
            learner is None
            or learner.capacity != num_episodes
            or learner.weights.key != key
        ):
            learner = GraphedPolicyBank(
                self.model,
                1,
                num_episodes,
                self.device,
                temperature=1.0,
                with_values=True,
                storage_rows=num_episodes * _INITIAL_DECISIONS_PER_EPISODE,
            )
            self._learner_bank = learner
        snapshots = self._snapshot_bank
        if snapshot_slots and (
            snapshots is None
            or snapshots.capacity != num_episodes
            or snapshots.slots < snapshot_slots
            or snapshots.weights.key != key
            or snapshots.temperature != float(self.snapshot_temperature)
        ):
            snapshots = GraphedPolicyBank(
                self.model,
                max(snapshot_slots, self.max_active_snapshots),
                num_episodes,
                self.device,
                temperature=self.snapshot_temperature,
                with_values=False,
            )
            self._snapshot_bank = snapshots
        return learner, snapshots

    def _run_graphed(
        self,
        num_episodes: int,
        state: VectorState,
        engine: VectorEngine,
        policy_data,
        opponent_kinds: np.ndarray,
        active_snapshot_indices: np.ndarray,
        snapshot_models: list[PPOActorCritic],
    ) -> tuple:
        """Collect a rollout with graph-replayed learner and snapshot policies.

        The learner's decisions, with their effective masks, log-probabilities
        and values, accumulate in the learner bank's device storage, which is
        returned as views: they stay valid until this runner's next rollout.
        Fixed opponents choose on the CPU while the GPU evaluates the policies.
        """
        learner, snapshot_bank = self._graph_banks(
            num_episodes, len(snapshot_models)
        )
        learner.weights.load(0, self.model)
        slot_of_snapshot = np.full(
            max(1, len(self.snapshot_state_dicts)), -1, dtype=np.int64
        )
        for slot, (raw_index, model) in enumerate(
            zip(active_snapshot_indices, snapshot_models)
        ):
            snapshot_bank.weights.load(slot, model)
            slot_of_snapshot[int(raw_index)] = slot
        stream = torch.cuda.current_stream(self.device)

        previous_chunks = []
        last_transition = np.full(num_episodes, -1, dtype=np.int64)
        offset = 0
        safety_limit = int(state.turn_cap.max()) * 200
        actions = np.empty(num_episodes, dtype=np.int32)
        action_variants = np.zeros(num_episodes, dtype=np.int16)
        timing = {
            "stage": 0.0,
            "launch": 0.0,
            "opponent": 0.0,
            "wait": 0.0,
            "step": 0.0,
            "iterations": 0,
            "policy_calls": 0,
            "policy_rows": 0,
            "snapshot_calls": 0,
            "snapshot_rows": 0,
        }
        loop_started = time.perf_counter()

        for _ in range(safety_limit):
            timing["iterations"] += 1
            if state.done.all():
                break
            phase = time.perf_counter()
            masks = engine.build_masks()
            active = ~state.done
            to_move = state.current_player == state.training_player
            training_rows = np.flatnonzero(active & to_move)
            opponent_rows = np.flatnonzero(active & ~to_move)
            actions.fill(END_TURN_INDEX)
            action_variants.fill(0)
            kinds = opponent_kinds[opponent_rows]
            fixed_rows = opponent_rows[kinds < SNAPSHOT_OFFSET]
            snapshot_rows = opponent_rows[kinds >= SNAPSHOT_OFFSET]

            count = len(training_rows)
            if count:
                if offset + count > learner.storage_rows:
                    learner.allocate_storage(int((offset + count) * 1.5))
                engine.materialize_current_player_observations(
                    training_rows,
                    out=learner.host_observations[:count].numpy(),
                )
                np.take(
                    masks,
                    training_rows,
                    axis=0,
                    out=learner.host_masks[:count].numpy(),
                )
            slot_rows: list[tuple[int, np.ndarray]] = []
            snapshot_bucket = 0
            if len(snapshot_rows):
                slots = slot_of_snapshot[
                    opponent_kinds[snapshot_rows] - SNAPSHOT_OFFSET
                ]
                counts = np.bincount(slots, minlength=snapshot_bank.slots)
                snapshot_bucket = snapshot_bank.bucket(int(counts.max()))
                order = np.argsort(slots, kind="stable")
                start = 0
                for slot, slot_count in enumerate(counts.tolist()):
                    if not slot_count:
                        continue
                    rows = snapshot_rows[order[start : start + slot_count]]
                    start += slot_count
                    engine.materialize_current_player_observations(
                        rows,
                        out=staged_rows(
                            snapshot_bank.host_observations,
                            slot,
                            snapshot_bucket,
                            slot_count,
                        ),
                    )
                    np.take(
                        masks,
                        rows,
                        axis=0,
                        out=staged_rows(
                            snapshot_bank.host_masks,
                            slot,
                            snapshot_bucket,
                            slot_count,
                        ),
                    )
                    slot_rows.append((slot, rows))
            timing["stage"] += time.perf_counter() - phase

            phase = time.perf_counter()
            if count:
                learner.submit(learner.bucket(count), offset)
                timing["policy_calls"] += 1
                timing["policy_rows"] += count
            if slot_rows:
                snapshot_bank.submit(snapshot_bucket)
                timing["snapshot_calls"] += 1
                timing["snapshot_rows"] += len(snapshot_rows)
            timing["launch"] += time.perf_counter() - phase

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

            if count or slot_rows:
                phase = time.perf_counter()
                stream.synchronize()
                timing["wait"] += time.perf_counter() - phase
            if count:
                actions[training_rows] = learner.host_actions[:count].numpy()
                previous_chunks.append(last_transition[training_rows])
                last_transition[training_rows] = offset + np.arange(
                    count, dtype=np.int64
                )
                offset += count
            for slot, rows in slot_rows:
                actions[rows] = staged_rows(
                    snapshot_bank.host_actions,
                    slot,
                    snapshot_bucket,
                    len(rows),
                )

            phase = time.perf_counter()
            engine.step(actions, action_variants=action_variants)
            timing["step"] += time.perf_counter() - phase
        else:
            raise RuntimeError("vector rollout exceeded safety limit")

        if not state.done.all():
            raise RuntimeError("vector rollout ended with active games")
        if not offset:
            raise RuntimeError("vector rollout collected no policy decisions")

        loop_elapsed = time.perf_counter() - loop_started
        previous = np.concatenate(previous_chunks)
        values = learner.values[:offset].cpu().numpy()
        returns, advantages, gae_elapsed = self._returns_and_advantages(
            state, last_transition, previous, values
        )
        self._record_opponent_results(state, opponent_kinds)
        result = (
            learner.states[:offset],
            learner.actions[:offset],
            learner.log_probs[:offset],
            torch.from_numpy(returns).to(self.device),
            torch.from_numpy(advantages).to(self.device),
            learner.masks[:offset],
        )
        captured = len(learner.captured_buckets) + (
            len(snapshot_bank.captured_buckets) if snapshot_bank else 0
        )
        _log.info(
            "[VectorRunner] graphed loop=%.2fs stage=%.2fs launch=%.2fs "
            "opponent=%.2fs wait=%.2fs step=%.2fs gae=%.2fs iterations=%d "
            "calls=%d avg_batch=%.0f snapshot_calls=%d "
            "snapshot_avg_rows=%.0f graphs=%d",
            loop_elapsed,
            timing["stage"],
            timing["launch"],
            timing["opponent"],
            timing["wait"],
            timing["step"],
            gae_elapsed,
            timing["iterations"],
            timing["policy_calls"],
            timing["policy_rows"] / max(1, timing["policy_calls"]),
            timing["snapshot_calls"],
            timing["snapshot_rows"] / max(1, timing["snapshot_calls"]),
            captured,
        )
        return result
