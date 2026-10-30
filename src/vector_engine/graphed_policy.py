"""Graph-replayed policy inference for vector-engine rollouts.

A vector rollout asks the policy for one decision per active game hundreds of
times per update, each time for a few hundred rows. At that size the GPU time is
almost entirely per-kernel launch overhead, so this module evaluates the
policies from static device buffers with CUDA/HIP graphs, one graph per
batch-size bucket, captured the first time a bucket is needed and reused for
the life of the process.

The graphs evaluate the sum-pooled MLP architecture through
``PPOActorCritic.folded_input_weight`` with weights copied into static stacked
tensors, so loading a different snapshot never invalidates a graph. Selection
matches :func:`src.ppo.action_selection.select_actions` on logits masked as in
:func:`src.ppo.mp_inference_server._forward_and_sample` after
:func:`src.vector_engine.policy.suppress_redundant_end_turn`: the same
distribution, drawn from the graph-safe device generator.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.encoding.action_encoder import END_TURN_INDEX, get_action_space_layout
from src.ppo.action_selection import validate_temperature
from src.ppo.ppo_actor_critic import PPOActorCritic


def batch_buckets(capacity: int) -> list[int]:
    """Graph batch sizes: four per doubling from 16, ending at ``capacity``."""
    if capacity < 1:
        raise ValueError("capacity must be positive")
    sizes = []
    base = 16
    while base < capacity:
        for step in (4, 5, 6, 7):
            size = base * step // 4
            if size >= capacity:
                break
            sizes.append(size)
        base *= 2
    sizes.append(capacity)
    return sorted(set(size for size in sizes if size <= capacity))


def _linear_layers(sequence: nn.Sequential) -> list[nn.Linear]:
    return [module for module in sequence if isinstance(module, nn.Linear)]


def _relu_pattern_ok(sequence: nn.Sequential, trailing_relu: bool) -> bool:
    modules = list(sequence)
    expected: list[type] = []
    linears = len(_linear_layers(sequence))
    for index in range(linears):
        expected.append(nn.Linear)
        if trailing_relu or index < linears - 1:
            expected.append(nn.ReLU)
    return [type(module) for module in modules] == expected


def graphable(model: PPOActorCritic) -> bool:
    """Whether ``model`` has the architecture graphed inference evaluates."""
    return (
        isinstance(model, PPOActorCritic)
        and model._foldable
        and model.actor_type == "mlp"
        and _relu_pattern_ok(model.trunk, trailing_relu=True)
        and _relu_pattern_ok(model.actor_head, trailing_relu=False)
        and _relu_pattern_ok(model.critic_head, trailing_relu=False)
    )


def architecture_key(model: PPOActorCritic) -> tuple:
    """Shapes of every layer graphed inference stacks across slots."""
    return (
        model.state_dim,
        model.action_dim,
        tuple(
            tuple(layer.weight.shape)
            for sequence in (model.trunk, model.actor_head, model.critic_head)
            for layer in _linear_layers(sequence)
        ),
    )


class StackedPolicyWeights:
    """Static per-slot copies of an MLP actor (and optionally its critic)."""

    def __init__(
        self,
        template: PPOActorCritic,
        slots: int,
        device: torch.device,
        *,
        with_critic: bool,
    ):
        if not graphable(template):
            raise ValueError("model architecture is not graphable")
        self.slots = slots
        self.with_critic = with_critic
        self.key = architecture_key(template)

        def stacked(layer: nn.Linear, in_features: int | None = None):
            out_features = layer.weight.shape[0]
            width = in_features or layer.weight.shape[1]
            return (
                torch.zeros(slots, out_features, width, device=device),
                torch.zeros(slots, out_features, device=device),
            )

        trunk = _linear_layers(template.trunk)
        self.trunk = [stacked(trunk[0], template.state_dim)] + [
            stacked(layer) for layer in trunk[1:]
        ]
        self.actor = [stacked(layer) for layer in _linear_layers(template.actor_head)]
        self.critic = (
            [stacked(layer) for layer in _linear_layers(template.critic_head)]
            if with_critic
            else []
        )

    @torch.no_grad()
    def load(self, slot: int, model: PPOActorCritic) -> None:
        """Copy ``model``'s weights into ``slot``, folding its first layer."""
        if architecture_key(model) != self.key or not graphable(model):
            raise ValueError("model architecture differs from the stacked slots")
        trunk = _linear_layers(model.trunk)
        weight, bias = self.trunk[0]
        weight[slot].copy_(model.folded_input_weight())
        bias[slot].copy_(trunk[0].bias)
        for (weight, bias), layer in zip(self.trunk[1:], trunk[1:]):
            weight[slot].copy_(layer.weight)
            bias[slot].copy_(layer.bias)
        for (weight, bias), layer in zip(
            self.actor, _linear_layers(model.actor_head)
        ):
            weight[slot].copy_(layer.weight)
            bias[slot].copy_(layer.bias)
        for (weight, bias), layer in zip(
            self.critic, _linear_layers(model.critic_head)
        ):
            weight[slot].copy_(layer.weight)
            bias[slot].copy_(layer.bias)

    @staticmethod
    def _apply(layers, x: torch.Tensor, final_relu: bool) -> torch.Tensor:
        for index, (weight, bias) in enumerate(layers):
            x = torch.baddbmm(bias.unsqueeze(1), x, weight.transpose(1, 2))
            if final_relu or index < len(layers) - 1:
                x = x.relu_()
        return x

    def forward(self, observations: torch.Tensor):
        """``observations`` ``[slots, rows, state]`` to logits and values."""
        hidden = self._apply(self.trunk, observations, final_relu=True)
        logits = self._apply(self.actor, hidden, final_relu=False)
        values = (
            self._apply(self.critic, hidden, final_relu=False).squeeze(-1)
            if self.with_critic
            else None
        )
        return logits, values


class GraphedPolicyBank:
    """Graph-replayed decisions for ``slots`` stacked policies.

    Callers stage each slot's rows in the pinned ``host_observations`` and
    ``host_masks`` at slot stride ``bucket`` (slot ``s`` starts at row
    ``s * bucket``), call :meth:`submit`, synchronize the stream, and read
    ``host_actions``. A bank built with ``storage_rows`` also appends every
    decision of its single slot, with its effective mask, log-probability and
    value, to device rollout storage at a caller-supplied row offset.
    """

    def __init__(
        self,
        template: PPOActorCritic,
        slots: int,
        capacity: int,
        device: torch.device,
        *,
        temperature: float,
        with_values: bool,
        storage_rows: int = 0,
    ):
        if device.type != "cuda":
            raise ValueError("graphed inference requires a CUDA/HIP device")
        if storage_rows and slots != 1:
            raise ValueError("rollout storage records a single slot")
        self.device = device
        self.slots = slots
        self.capacity = capacity
        self.temperature = validate_temperature(temperature)
        self.with_values = with_values
        self.weights = StackedPolicyWeights(
            template, slots, device, with_critic=with_values
        )
        self.state_dim = template.state_dim
        self.action_dim = template.action_dim
        layout = get_action_space_layout(template.num_cards)
        play = next(
            (offset, count)
            for name, offset, count in layout["card_groups"]
            if name == "play_card"
        )
        self._play = slice(play[0], play[0] + play[1])
        self._attack_player = layout["global_indices"]["attack_player"]
        self.buckets = batch_buckets(capacity)

        rows = slots * capacity
        self.host_observations = torch.empty(
            (rows, self.state_dim), dtype=torch.float32, pin_memory=True
        )
        self.host_masks = torch.zeros(
            (rows, self.action_dim), dtype=torch.bool, pin_memory=True
        )
        self.host_actions = torch.zeros(rows, dtype=torch.int64, pin_memory=True)
        self.host_offset = torch.zeros(1, dtype=torch.int64, pin_memory=True)
        self._observations = torch.zeros(
            (rows, self.state_dim), device=device
        )
        self._masks = torch.zeros(
            (rows, self.action_dim), dtype=torch.bool, device=device
        )
        self._actions = torch.zeros(rows, dtype=torch.int64, device=device)
        self._offset = torch.zeros(1, dtype=torch.int64, device=device)
        self._row_index = torch.arange(capacity, device=device)
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._pool = None
        self.storage_rows = 0
        if storage_rows:
            self.allocate_storage(storage_rows)

    # ------------------------------------------------------------------
    # Rollout storage
    # ------------------------------------------------------------------
    def allocate_storage(self, rows: int) -> None:
        """(Re)allocate rollout storage for ``rows`` decisions.

        The captured graphs write into the storage tensors, so replacing them
        discards every graph; they are recaptured on next use. Rows already
        recorded are kept.
        """
        rows = int(rows) + self.capacity
        old = (
            (self.states, self.masks, self.actions, self.log_probs, self.values)
            if self.storage_rows
            else None
        )
        self.states = torch.empty((rows, self.state_dim), device=self.device)
        self.masks = torch.empty(
            (rows, self.action_dim), dtype=torch.bool, device=self.device
        )
        self.actions = torch.empty(rows, dtype=torch.int64, device=self.device)
        self.log_probs = torch.empty(rows, device=self.device)
        self.values = torch.empty(rows, device=self.device)
        if old is not None:
            kept = min(self.storage_rows, rows)
            for new, previous in zip(
                (self.states, self.masks, self.actions, self.log_probs, self.values),
                old,
            ):
                new[:kept].copy_(previous[:kept])
        self.storage_rows = rows - self.capacity
        self._graphs.clear()
        self._pool = None

    # ------------------------------------------------------------------
    # Graph capture and replay
    # ------------------------------------------------------------------
    def bucket(self, rows: int) -> int:
        """Smallest graph batch size holding ``rows`` rows per slot."""
        if rows > self.capacity:
            raise ValueError(f"{rows} rows exceed capacity {self.capacity}")
        for size in self.buckets:
            if size >= rows:
                return size
        return self.capacity

    def _select(self, logits: torch.Tensor):
        if self.temperature == 0.0:
            actions = logits.argmax(dim=-1)
            return actions, torch.zeros_like(logits[..., 0])
        scaled = logits if self.temperature == 1.0 else logits / self.temperature
        log_probs = torch.log_softmax(scaled, dim=-1)
        flat = log_probs.reshape(-1, log_probs.shape[-1])
        actions = torch.multinomial(flat.exp(), 1, True).reshape(
            log_probs.shape[:-1]
        )
        chosen = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        return actions, chosen

    def _decide(self, bucket: int) -> None:
        rows = self.slots * bucket
        observations = self._observations[:rows].view(
            self.slots, bucket, self.state_dim
        )
        masks = self._masks[:rows].view(self.slots, bucket, self.action_dim)
        meaningful = masks[..., self._play].any(dim=-1) | masks[
            ..., self._attack_player
        ]
        masks[..., END_TURN_INDEX] &= ~meaningful
        logits, values = self.weights.forward(observations)
        logits = logits.masked_fill(~masks, float("-inf"))
        no_valid = ~masks.any(dim=-1)
        logits[..., END_TURN_INDEX] = torch.where(
            no_valid,
            torch.zeros_like(logits[..., END_TURN_INDEX]),
            logits[..., END_TURN_INDEX],
        )
        actions, log_probs = self._select(logits)
        self._actions[:rows].copy_(actions.reshape(-1))
        if self.storage_rows:
            index = self._offset + self._row_index[:bucket]
            self.states.index_copy_(0, index, observations[0])
            self.masks.index_copy_(0, index, masks[0])
            self.actions.index_copy_(0, index, actions[0])
            self.log_probs.index_copy_(0, index, log_probs[0])
            if values is not None:
                self.values.index_copy_(0, index, values[0])

    def _graph(self, bucket: int) -> torch.cuda.CUDAGraph:
        graph = self._graphs.get(bucket)
        if graph is not None:
            return graph
        if self.storage_rows:
            # Warm-up writes must stay inside storage: aim them at the rows
            # past the recorded decisions, which the next replays overwrite.
            self._offset.fill_(self.storage_rows)
        side = torch.cuda.Stream(self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(2):
                self._decide(bucket)
        torch.cuda.current_stream(self.device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        with torch.cuda.graph(graph, pool=self._pool), torch.no_grad():
            self._decide(bucket)
        self._graphs[bucket] = graph
        return graph

    def submit(self, bucket: int, offset: int | None = None) -> None:
        """Copy staged rows in, replay ``bucket``'s graph, copy actions out.

        Asynchronous: the caller synchronizes the stream before reading
        ``host_actions`` or restaging rows.
        """
        rows = self.slots * bucket
        graph = self._graph(bucket)
        if self.storage_rows:
            if offset is None or offset + bucket > self.storage_rows + self.capacity:
                raise ValueError("rollout storage offset out of range")
            self.host_offset[0] = offset
            self._offset.copy_(self.host_offset, non_blocking=True)
        self._observations[:rows].copy_(
            self.host_observations[:rows], non_blocking=True
        )
        self._masks[:rows].copy_(self.host_masks[:rows], non_blocking=True)
        graph.replay()
        self.host_actions[:rows].copy_(self._actions[:rows], non_blocking=True)

    @property
    def captured_buckets(self) -> list[int]:
        return sorted(self._graphs)


def staged_rows(host: torch.Tensor, slot: int, bucket: int, count: int) -> np.ndarray:
    """NumPy view of slot ``slot``'s first ``count`` staging rows."""
    start = slot * bucket
    return host[start : start + count].numpy()
