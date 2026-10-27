"""Actual PPO policy integration for the CPU Numba vector engine."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, stack_module_state, vmap

from src.encoding.action_encoder import END_TURN_INDEX
from src.ppo.action_selection import validate_temperature
from src.ppo.mp_inference_server import _forward_and_sample
from src.ppo.ppo_actor_critic import PPOActorCritic


def _snapshot_actions(
    masked_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Snapshot moves along the last axis of logits whose illegal entries are ``-inf``.

    Temperature 1 samples the policy, other positive temperatures sample
    ``softmax(logits / temperature)``, and 0 selects the highest-scoring legal
    action (the lowest index on ties).
    """
    if temperature == 0.0:
        return masked_logits.argmax(dim=-1)
    logits = masked_logits if temperature == 1.0 else masked_logits / temperature
    return torch.distributions.Categorical(
        logits=logits,
        validate_args=False,
    ).sample()


@dataclass(frozen=True)
class PolicyDecision:
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    device_actions: torch.Tensor
    device_log_probs: torch.Tensor
    device_values: torch.Tensor


@dataclass(frozen=True)
class JointPolicyDecision:
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    snapshot_actions: np.ndarray


@dataclass(frozen=True)
class SnapshotPolicyDecision:
    actions: tuple[np.ndarray, ...]
    rows: int


def suppress_redundant_end_turn(masks: np.ndarray) -> np.ndarray:
    """Match production rollout's action-mask policy.

    The rules engine exposes END_TURN exactly like the scalar action context.
    Rollout collection then suppresses it while PLAY_CARD or direct
    ATTACK_PLAYER is legal, preventing premature pass actions.
    """
    if masks.ndim != 2:
        raise ValueError("masks must be rank-2")
    num_cards = (masks.shape[1] - 4) // 10
    if 4 + 10 * num_cards != masks.shape[1]:
        raise ValueError(
            f"action dimension {masks.shape[1]} is not canonical"
        )
    attack_player = 3 + 2 * num_cards
    effective = masks.copy()
    meaningful = (
        effective[:, 3 : 3 + num_cards].any(axis=1)
        | effective[:, attack_player]
    )
    effective[meaningful, END_TURN_INDEX] = False
    return effective


class VectorPolicyEvaluator:
    """Pinned CPU staging plus one resident-GPU policy model."""

    def __init__(self, model: PPOActorCritic, device: torch.device):
        self.device = device
        self.model = model.to(device).eval()
        self._capacity = 0
        self._state_size = 0
        self._action_dim = 0

    def _ensure_buffers(
        self, games: int, state_size: int, action_dim: int
    ) -> None:
        if (
            state_size == self._state_size
            and action_dim == self._action_dim
            and games <= self._capacity
        ):
            return
        capacity = max(games, self._capacity * 2)
        pin = self.device.type == "cuda"
        self.host_observations = torch.empty(
            (capacity, state_size), dtype=torch.float32, pin_memory=pin
        )
        self.host_masks = torch.empty(
            (capacity, action_dim), dtype=torch.uint8, pin_memory=pin
        )
        self.device_observations = torch.empty(
            (capacity, state_size), dtype=torch.float32, device=self.device
        )
        self.device_masks = torch.empty(
            (capacity, action_dim), dtype=torch.uint8, device=self.device
        )
        self.host_actions = torch.empty(
            capacity, dtype=torch.int64, pin_memory=pin
        )
        self.host_log_probs = torch.empty(
            capacity, dtype=torch.float32, pin_memory=pin
        )
        self.host_values = torch.empty(
            capacity, dtype=torch.float32, pin_memory=pin
        )
        self._capacity = capacity
        self._state_size = state_size
        self._action_dim = action_dim

    @torch.no_grad()
    def decide(
        self,
        observations: np.ndarray,
        masks: np.ndarray,
    ) -> PolicyDecision:
        if observations.ndim != 2 or masks.ndim != 2:
            raise ValueError("observations and masks must be rank-2")
        if observations.shape[0] != masks.shape[0]:
            raise ValueError("observations and masks batch sizes differ")
        games, state_size = observations.shape
        action_dim = masks.shape[1]
        self._ensure_buffers(games, state_size, action_dim)

        host_observations = self.host_observations[:games]
        host_masks = self.host_masks[:games]
        device_observations = self.device_observations[:games]
        device_masks = self.device_masks[:games]
        host_actions = self.host_actions[:games]
        host_log_probs = self.host_log_probs[:games]
        host_values = self.host_values[:games]

        np.copyto(host_observations.numpy(), observations)
        effective_masks = suppress_redundant_end_turn(masks)
        np.copyto(
            host_masks.numpy(), effective_masks, casting="unsafe"
        )
        async_copy = self.device.type == "cuda"
        device_observations.copy_(host_observations, non_blocking=async_copy)
        device_masks.copy_(host_masks, non_blocking=async_copy)

        actions, log_probs, values, _ = _forward_and_sample(
            self.model,
            device_observations,
            device_masks,
        )
        host_actions.copy_(actions, non_blocking=async_copy)
        host_log_probs.copy_(log_probs, non_blocking=async_copy)
        host_values.copy_(values, non_blocking=async_copy)
        if async_copy:
            torch.cuda.synchronize(self.device)

        return PolicyDecision(
            actions=host_actions.numpy().copy(),
            log_probs=host_log_probs.numpy().copy(),
            values=host_values.numpy().copy(),
            device_actions=actions,
            device_log_probs=log_probs,
            device_values=values,
        )


class CurrentSnapshotEvaluator:
    """Evaluate the learning policy and one snapshot with one synchronization.

    The snapshot plays at ``snapshot_temperature``: 1 samples its policy, other
    positive values sample ``softmax(logits / temperature)``, and 0 plays its
    highest-scoring legal action.
    """

    def __init__(
        self,
        current_model: PPOActorCritic,
        snapshot_model: PPOActorCritic,
        device: torch.device,
        snapshot_temperature: float = 1.0,
    ):
        self.device = device
        self.current_model = current_model.to(device).eval()
        self.snapshot_model = snapshot_model.to(device).eval()
        self.snapshot_temperature = validate_temperature(snapshot_temperature)
        self._current_capacity = 0
        self._snapshot_capacity = 0
        self._state_size = 0
        self._action_dim = 0
        self.stats = {
            "calls": 0,
            "rows": 0,
            "stage_s": 0.0,
            "launch_s": 0.0,
            "sync_s": 0.0,
            "gpu_s": 0.0,
        }
        if device.type == "cuda":
            self._gpu_start = torch.cuda.Event(enable_timing=True)
            self._gpu_end = torch.cuda.Event(enable_timing=True)

    def _ensure_buffers(
        self,
        current_count: int,
        snapshot_count: int,
        state_size: int,
        action_dim: int,
    ) -> None:
        if (
            state_size == self._state_size
            and action_dim == self._action_dim
            and current_count <= self._current_capacity
            and snapshot_count <= self._snapshot_capacity
        ):
            return
        current_capacity = max(
            1, current_count, self._current_capacity * 2
        )
        snapshot_capacity = max(
            1, snapshot_count, self._snapshot_capacity * 2
        )
        pin = self.device.type == "cuda"
        self.host_current_observations = torch.empty(
            (current_capacity, state_size),
            dtype=torch.float32,
            pin_memory=pin,
        )
        self.host_current_masks = torch.empty(
            (current_capacity, action_dim),
            dtype=torch.uint8,
            pin_memory=pin,
        )
        self.host_current_actions = torch.empty(
            current_capacity, dtype=torch.int64, pin_memory=pin
        )
        self.host_current_log_probs = torch.empty(
            current_capacity, dtype=torch.float32, pin_memory=pin
        )
        self.host_current_values = torch.empty(
            current_capacity, dtype=torch.float32, pin_memory=pin
        )
        self.host_snapshot_observations = torch.empty(
            (snapshot_capacity, state_size),
            dtype=torch.float32,
            pin_memory=pin,
        )
        self.host_snapshot_masks = torch.empty(
            (snapshot_capacity, action_dim),
            dtype=torch.uint8,
            pin_memory=pin,
        )
        self.host_snapshot_actions = torch.empty(
            snapshot_capacity, dtype=torch.int64, pin_memory=pin
        )
        self.device_current_observations = torch.empty(
            (current_capacity, state_size),
            dtype=torch.float32,
            device=self.device,
        )
        self.device_current_masks = torch.empty(
            (current_capacity, action_dim),
            dtype=torch.uint8,
            device=self.device,
        )
        self.device_snapshot_observations = torch.empty(
            (snapshot_capacity, state_size),
            dtype=torch.float32,
            device=self.device,
        )
        self.device_snapshot_masks = torch.empty(
            (snapshot_capacity, action_dim),
            dtype=torch.uint8,
            device=self.device,
        )
        self._current_capacity = current_capacity
        self._snapshot_capacity = snapshot_capacity
        self._state_size = state_size
        self._action_dim = action_dim

    @torch.no_grad()
    def decide(
        self,
        current_observations: np.ndarray,
        current_masks: np.ndarray,
        snapshot_observations: np.ndarray,
        snapshot_masks: np.ndarray,
    ) -> JointPolicyDecision:
        groups = (
            (current_observations, current_masks),
            (snapshot_observations, snapshot_masks),
        )
        for observations, masks in groups:
            if observations.ndim != 2 or masks.ndim != 2:
                raise ValueError("observations and masks must be rank-2")
            if observations.shape[0] != masks.shape[0]:
                raise ValueError(
                    "observation and mask batch sizes differ"
                )
        current_count = len(current_observations)
        snapshot_count = len(snapshot_observations)
        if snapshot_count == 0:
            raise ValueError("joint evaluation requires snapshot rows")
        state_size = snapshot_observations.shape[1]
        action_dim = snapshot_masks.shape[1]
        if current_count and (
            current_observations.shape[1] != state_size
            or current_masks.shape[1] != action_dim
        ):
            raise ValueError("current and snapshot dimensions differ")
        self._ensure_buffers(
            current_count,
            snapshot_count,
            state_size,
            action_dim,
        )

        started = time.perf_counter()
        if current_count:
            np.copyto(
                self.host_current_observations[
                    :current_count
                ].numpy(),
                current_observations,
            )
            np.copyto(
                self.host_current_masks[:current_count].numpy(),
                suppress_redundant_end_turn(current_masks),
                casting="unsafe",
            )
        np.copyto(
            self.host_snapshot_observations[:snapshot_count].numpy(),
            snapshot_observations,
        )
        np.copyto(
            self.host_snapshot_masks[:snapshot_count].numpy(),
            suppress_redundant_end_turn(snapshot_masks),
            casting="unsafe",
        )
        self.stats["stage_s"] += time.perf_counter() - started

        async_copy = self.device.type == "cuda"
        started = time.perf_counter()
        if async_copy:
            self._gpu_start.record()
        if current_count:
            current_observations_device = (
                self.device_current_observations[:current_count]
            )
            current_masks_device = self.device_current_masks[:current_count]
            current_observations_device.copy_(
                self.host_current_observations[:current_count],
                non_blocking=async_copy,
            )
            current_masks_device.copy_(
                self.host_current_masks[:current_count],
                non_blocking=async_copy,
            )
            current_actions, current_log_probs, current_values, _ = (
                _forward_and_sample(
                    self.current_model,
                    current_observations_device,
                    current_masks_device,
                )
            )
            self.host_current_actions[:current_count].copy_(
                current_actions, non_blocking=async_copy
            )
            self.host_current_log_probs[:current_count].copy_(
                current_log_probs, non_blocking=async_copy
            )
            self.host_current_values[:current_count].copy_(
                current_values, non_blocking=async_copy
            )

        snapshot_observations_device = (
            self.device_snapshot_observations[:snapshot_count]
        )
        snapshot_masks_device = self.device_snapshot_masks[:snapshot_count]
        snapshot_observations_device.copy_(
            self.host_snapshot_observations[:snapshot_count],
            non_blocking=async_copy,
        )
        snapshot_masks_device.copy_(
            self.host_snapshot_masks[:snapshot_count],
            non_blocking=async_copy,
        )
        snapshot_logits = self.snapshot_model.policy_logits(
            snapshot_observations_device
        ).masked_fill(
            snapshot_masks_device == 0,
            float("-inf"),
        )
        snapshot_actions = _snapshot_actions(
            snapshot_logits, self.snapshot_temperature
        )
        self.host_snapshot_actions[:snapshot_count].copy_(
            snapshot_actions, non_blocking=async_copy
        )
        if async_copy:
            self._gpu_end.record()
        self.stats["launch_s"] += time.perf_counter() - started

        started = time.perf_counter()
        if async_copy:
            torch.cuda.synchronize(self.device)
            self.stats["gpu_s"] += (
                self._gpu_start.elapsed_time(self._gpu_end) / 1000.0
            )
        self.stats["sync_s"] += time.perf_counter() - started
        self.stats["calls"] += 1
        self.stats["rows"] += current_count + snapshot_count

        return JointPolicyDecision(
            actions=self.host_current_actions[
                :current_count
            ].numpy().copy(),
            log_probs=self.host_current_log_probs[
                :current_count
            ].numpy().copy(),
            values=self.host_current_values[
                :current_count
            ].numpy().copy(),
            snapshot_actions=self.host_snapshot_actions[
                :snapshot_count
            ].numpy().copy(),
        )


class _ActorOnlyPolicy(nn.Module):
    """Functional-call wrapper that excludes the critic head."""

    def __init__(self, model: PPOActorCritic):
        super().__init__()
        self.model = model

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.model.policy_logits(observations)


class StackedSnapshotEvaluator:
    """Evaluate architecture-compatible snapshot actors in one GPU call.

    Every snapshot plays at ``temperature``, as in
    :class:`CurrentSnapshotEvaluator`.
    """

    def __init__(
        self,
        models: list[PPOActorCritic],
        device: torch.device,
        temperature: float = 1.0,
    ):
        if not models:
            raise ValueError("at least one snapshot model is required")
        self.device = device
        self.temperature = validate_temperature(temperature)
        wrappers = [
            _ActorOnlyPolicy(model.to(device).eval())
            for model in models
        ]
        self.model_count = len(models)
        if self.model_count == 1:
            self.single_model = wrappers[0]
            self.params = None
            self.buffers = None
            self.base_model = None
            self._vmapped_forward = None
        else:
            self.single_model = None
            self.params, self.buffers = stack_module_state(wrappers)
            self.base_model = copy.deepcopy(wrappers[0]).to("meta")
            self._vmapped_forward = vmap(
                self._functional_forward,
                in_dims=(0, 0, 0),
            )
        self._capacity = 0
        self._state_size = 0
        self._action_dim = 0
        self.stats = {
            "calls": 0,
            "rows": 0,
            "stage_s": 0.0,
            "launch_s": 0.0,
            "sync_s": 0.0,
            "gpu_s": 0.0,
        }
        if device.type == "cuda":
            self._gpu_start = torch.cuda.Event(enable_timing=True)
            self._gpu_end = torch.cuda.Event(enable_timing=True)

    def _functional_forward(self, params, buffers, observations):
        return functional_call(
            self.base_model,
            (params, buffers),
            (observations,),
            strict=True,
        )

    def _ensure_buffers(
        self,
        capacity: int,
        state_size: int,
        action_dim: int,
    ) -> None:
        if (
            state_size == self._state_size
            and action_dim == self._action_dim
            and capacity <= self._capacity
        ):
            return
        capacity = max(capacity, self._capacity * 2)
        pin = self.device.type == "cuda"
        shape = (self.model_count, capacity)
        self.host_observations = torch.empty(
            (*shape, state_size), dtype=torch.float32, pin_memory=pin
        )
        self.host_masks = torch.empty(
            (*shape, action_dim), dtype=torch.uint8, pin_memory=pin
        )
        self.host_actions = torch.empty(
            shape, dtype=torch.int64, pin_memory=pin
        )
        self.device_observations = torch.empty(
            (*shape, state_size),
            dtype=torch.float32,
            device=self.device,
        )
        self.device_masks = torch.empty(
            (*shape, action_dim),
            dtype=torch.uint8,
            device=self.device,
        )
        self._capacity = capacity
        self._state_size = state_size
        self._action_dim = action_dim

    @torch.no_grad()
    def decide(
        self,
        observations: list[np.ndarray],
        masks: list[np.ndarray],
    ) -> SnapshotPolicyDecision:
        if len(observations) != self.model_count or len(masks) != self.model_count:
            raise ValueError("one observation and mask group is required per model")
        counts = [len(group) for group in observations]
        if not any(counts):
            return SnapshotPolicyDecision(
                actions=tuple(
                    np.empty(0, dtype=np.int64) for _ in counts
                ),
                rows=0,
            )
        state_size = next(
            group.shape[1] for group in observations if len(group)
        )
        action_dim = next(group.shape[1] for group in masks if len(group))
        capacity = max(1, max(counts))
        self._ensure_buffers(capacity, state_size, action_dim)

        started = time.perf_counter()
        host_observations = self.host_observations[:, :capacity]
        host_masks = self.host_masks[:, :capacity]
        host_observations.zero_()
        host_masks.zero_()
        for model_index, count in enumerate(counts):
            if count:
                np.copyto(
                    host_observations[model_index, :count].numpy(),
                    observations[model_index],
                )
                effective = suppress_redundant_end_turn(masks[model_index])
                np.copyto(
                    host_masks[model_index, :count].numpy(),
                    effective,
                    casting="unsafe",
                )
            if count < capacity:
                host_masks[
                    model_index, count:capacity, END_TURN_INDEX
                ] = 1
        self.stats["stage_s"] += time.perf_counter() - started

        async_copy = self.device.type == "cuda"
        started = time.perf_counter()
        if async_copy:
            self._gpu_start.record()
        self.device_observations[:, :capacity].copy_(
            host_observations,
            non_blocking=async_copy,
        )
        self.device_masks[:, :capacity].copy_(
            host_masks,
            non_blocking=async_copy,
        )
        if self.single_model is not None:
            logits = self.single_model(
                self.device_observations[0, :capacity]
            ).unsqueeze(0)
        else:
            logits = self._vmapped_forward(
                self.params,
                self.buffers,
                self.device_observations[:, :capacity],
            )
        logits = logits.masked_fill(
            self.device_masks[:, :capacity] == 0,
            float("-inf"),
        )
        actions = _snapshot_actions(logits, self.temperature)
        self.host_actions[:, :capacity].copy_(
            actions,
            non_blocking=async_copy,
        )
        if async_copy:
            self._gpu_end.record()
        self.stats["launch_s"] += time.perf_counter() - started

        started = time.perf_counter()
        if async_copy:
            torch.cuda.synchronize(self.device)
            self.stats["gpu_s"] += (
                self._gpu_start.elapsed_time(self._gpu_end) / 1000.0
            )
        self.stats["sync_s"] += time.perf_counter() - started
        self.stats["calls"] += 1
        self.stats["rows"] += sum(counts)
        return SnapshotPolicyDecision(
            actions=tuple(
                self.host_actions[
                    model_index, :count
                ].numpy().copy()
                for model_index, count in enumerate(counts)
            ),
            rows=sum(counts),
        )
