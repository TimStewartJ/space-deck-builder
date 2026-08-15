"""Actual PPO policy integration for the CPU Numba vector engine."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.encoding.action_encoder import END_TURN_INDEX
from src.ppo.mp_inference_server import _forward_and_sample
from src.ppo.ppo_actor_critic import PPOActorCritic


@dataclass(frozen=True)
class PolicyDecision:
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    device_actions: torch.Tensor
    device_log_probs: torch.Tensor
    device_values: torch.Tensor


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
