"""Graph-replayed policy inference for vector-engine rollouts.

A vector rollout asks the policy for one decision per active game hundreds of
times per update, each time for a few hundred rows. At that size the GPU time is
almost entirely launch overhead, so this module evaluates the policies from
static device buffers with CUDA/HIP graphs, one graph per batch-size bucket,
captured the first time a bucket is needed and reused for the life of the
process. Replaying a graph on ROCm still costs host time for every node it
holds (about 40 microseconds per GEMM on the project's machine), so each
decision step replays a single graph in which the learner and every opponent
policy share the same stacked kernels, and the CPU prepares masks while
staging rows.

The graphs evaluate the sum-pooled MLP architecture through
``PPOActorCritic.folded_input_weight`` with weights copied into static stacked
tensors, so loading a different snapshot never invalidates a graph. Masks are
staged as :func:`src.vector_engine.policy.suppress_redundant_end_turn` makes
them, and selection matches :func:`src.ppo.action_selection.select_actions`:
the same distribution, drawn from the graph-safe device generator.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from numba import njit

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


@njit(cache=True)
def _suppress_redundant_end_turn(out, play_start, play_end, attack_player, end_turn):
    actions = out.shape[1]
    for index in range(out.shape[0]):
        meaningful = out[index, attack_player]
        if not meaningful:
            for action in range(play_start, play_end):
                if out[index, action]:
                    meaningful = True
                    break
        if meaningful:
            out[index, end_turn] = False
        elif not out[index, end_turn]:
            any_legal = False
            for action in range(actions):
                if out[index, action]:
                    any_legal = True
                    break
            if not any_legal:
                out[index, end_turn] = True


def stage_effective_masks(
    masks, rows, out, play_start, play_end, attack_player, end_turn
):
    """Copy ``masks[rows]`` into ``out`` as the policy sees them.

    END_TURN is removed while a card can be played or the opponent attacked
    directly (``suppress_redundant_end_turn``). A row with no legal action at
    all keeps END_TURN, which is what the policy plays there.
    """
    np.take(masks, rows, axis=0, out=out)
    _suppress_redundant_end_turn(
        out, play_start, play_end, attack_player, end_turn
    )


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


def _stacked(slots: int, out_features: int, in_features: int, device):
    return (
        torch.zeros(slots, out_features, in_features, device=device),
        torch.zeros(slots, out_features, device=device),
    )


class StackedPolicyWeights:
    """Static per-slot copies of an MLP actor (and optionally its critic).

    With the critic, actor and critic heads of equal depth are evaluated as
    one stacked head: the first layer concatenates both heads' outputs over
    the shared trunk features and later layers are block-diagonal, so the
    heads stay independent while taking half the kernels.
    """

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
        trunk = _linear_layers(template.trunk)
        actor = _linear_layers(template.actor_head)
        critic = _linear_layers(template.critic_head)
        self.action_dim = actor[-1].weight.shape[0]
        self.trunk = [_stacked(slots, trunk[0].weight.shape[0], template.state_dim, device)] + [
            _stacked(slots, *layer.weight.shape, device) for layer in trunk[1:]
        ]
        self.combined = with_critic and len(actor) == len(critic)
        self.heads: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.actor: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.critic: list[tuple[torch.Tensor, torch.Tensor]] = []
        if self.combined:
            for index, (a, c) in enumerate(zip(actor, critic)):
                width = a.weight.shape[1] + (c.weight.shape[1] if index else 0)
                self.heads.append(
                    _stacked(
                        slots, a.weight.shape[0] + c.weight.shape[0], width, device
                    )
                )
        else:
            self.actor = [_stacked(slots, *layer.weight.shape, device) for layer in actor]
            if with_critic:
                self.critic = [
                    _stacked(slots, *layer.weight.shape, device) for layer in critic
                ]

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
        actor = _linear_layers(model.actor_head)
        critic = _linear_layers(model.critic_head)
        if self.combined:
            # Off-diagonal blocks are never written and stay zero.
            for index, ((weight, bias), a, c) in enumerate(
                zip(self.heads, actor, critic)
            ):
                rows = a.weight.shape[0]
                columns = a.weight.shape[1]
                if index == 0:
                    weight[slot, :rows].copy_(a.weight)
                    weight[slot, rows:].copy_(c.weight)
                else:
                    weight[slot, :rows, :columns].copy_(a.weight)
                    weight[slot, rows:, columns:].copy_(c.weight)
                bias[slot, :rows].copy_(a.bias)
                bias[slot, rows:].copy_(c.bias)
            return
        for (weight, bias), layer in zip(self.actor, actor):
            weight[slot].copy_(layer.weight)
            bias[slot].copy_(layer.bias)
        for (weight, bias), layer in zip(self.critic, critic):
            weight[slot].copy_(layer.weight)
            bias[slot].copy_(layer.bias)

    @staticmethod
    def _apply(layers, x: torch.Tensor, slots: int, final_relu: bool) -> torch.Tensor:
        for index, (weight, bias) in enumerate(layers):
            x = torch.baddbmm(
                bias[:slots].unsqueeze(1), x, weight[:slots].transpose(1, 2)
            )
            if final_relu or index < len(layers) - 1:
                x = x.relu_()
        return x

    def forward(self, observations: torch.Tensor):
        """``observations`` ``[slots, rows, state]`` for the first ``slots``
        slots to logits ``[slots, rows, actions]`` and values (or ``None``)."""
        slots = observations.shape[0]
        hidden = self._apply(self.trunk, observations, slots, final_relu=True)
        if self.combined:
            out = self._apply(self.heads, hidden, slots, final_relu=False)
            return out[..., : self.action_dim], out[..., self.action_dim]
        logits = self._apply(self.actor, hidden, slots, final_relu=False)
        values = (
            self._apply(self.critic, hidden, slots, final_relu=False).squeeze(-1)
            if self.with_critic
            else None
        )
        return logits, values


class GraphedRolloutPolicy:
    """Graph-replayed decisions for a learner and stacked opponent policies.

    Slot 0 holds the learner; slots ``1..opponent_slots`` hold opponents
    (snapshots or frozen checkpoints). Each decision step replays one graph
    that evaluates every slot with the same stacked kernels, so a step costs
    one replay however many policies act in it. Steps without opponent rows
    replay a learner-only graph.

    Callers stage each slot's rows in the pinned ``host_observations`` (for
    example with ``VectorEngine.materialize_current_player_observations``) and
    their masks with :meth:`stage_masks`, at slot stride ``bucket`` (slot
    ``s`` starts at row ``s * bucket``); call :meth:`submit`; synchronize the
    stream; and read ``host_actions`` at the same layout.

    With ``storage_rows``, every learner decision is recorded with its staged
    observation and mask, log-probability and value in device rollout storage
    at a caller-chosen row offset. Each replay writes a whole bucket there, so
    the rows after a step's decisions hold padding that the next step
    overwrites.
    """

    def __init__(
        self,
        template: PPOActorCritic,
        opponent_slots: int,
        capacity: int,
        device: torch.device,
        *,
        learner_temperature: float = 1.0,
        opponent_temperature: float = 1.0,
        storage_rows: int = 0,
    ):
        if device.type != "cuda":
            raise ValueError("graphed inference requires a CUDA/HIP device")
        self.device = device
        self.slots = 1 + opponent_slots
        self.opponent_slots = opponent_slots
        self.capacity = capacity
        self.learner_temperature = validate_temperature(learner_temperature)
        self.opponent_temperature = validate_temperature(opponent_temperature)
        self.weights = StackedPolicyWeights(
            template, self.slots, device, with_critic=True
        )
        self.state_dim = template.state_dim
        self.action_dim = template.action_dim
        layout = get_action_space_layout(template.num_cards)
        play = next(
            (offset, count)
            for name, offset, count in layout["card_groups"]
            if name == "play_card"
        )
        self.play_start, self.play_end = play[0], play[0] + play[1]
        self.attack_player = layout["global_indices"]["attack_player"]
        self.buckets = batch_buckets(capacity)

        # Every staged row, including rows padding a bucket, holds finite
        # observations and at least one legal action, so selection never sees
        # an all-masked row.
        rows = self.slots * capacity
        self.host_observations = torch.zeros(
            (rows, self.state_dim), dtype=torch.float32, pin_memory=True
        )
        self.host_masks = torch.zeros(
            (rows, self.action_dim), dtype=torch.bool, pin_memory=True
        )
        self.host_masks[:, END_TURN_INDEX] = True
        self.host_actions = torch.zeros(rows, dtype=torch.int64, pin_memory=True)
        self.host_offset = torch.zeros(1, dtype=torch.int64, pin_memory=True)
        self._observations = torch.zeros((rows, self.state_dim), device=device)
        self._masks = torch.zeros(
            (rows, self.action_dim), dtype=torch.bool, device=device
        )
        self._masks[:, END_TURN_INDEX] = True
        self._actions = torch.zeros(rows, dtype=torch.int64, device=device)
        self._offset = torch.zeros(1, dtype=torch.int64, device=device)
        self._row_index = torch.arange(capacity, device=device)
        self._graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self._pool = None
        self.storage_rows = 0
        if storage_rows:
            self.allocate_storage(storage_rows)

    # ------------------------------------------------------------------
    # Weights and rollout storage
    # ------------------------------------------------------------------
    def load_learner(self, model: PPOActorCritic) -> None:
        self.weights.load(0, model)

    def load_opponent(self, slot: int, model: PPOActorCritic) -> None:
        """Load an opponent policy into ``slot`` (``1..opponent_slots``)."""
        if not 1 <= slot <= self.opponent_slots:
            raise ValueError(f"opponent slot {slot} is out of range")
        self.weights.load(slot, model)

    def allocate_storage(self, rows: int) -> None:
        """(Re)allocate rollout storage for ``rows`` decisions, keeping the
        rows already recorded.

        The captured graphs write into the storage tensors, so replacing them
        discards every graph; they are recaptured on next use.
        """
        rows = int(rows)
        allocated = rows + self.capacity
        old = (
            (self.states, self.masks, self.actions, self.log_probs, self.values)
            if self.storage_rows
            else None
        )
        self.states = torch.empty((allocated, self.state_dim), device=self.device)
        self.masks = torch.empty(
            (allocated, self.action_dim), dtype=torch.bool, device=self.device
        )
        self.actions = torch.empty(allocated, dtype=torch.int64, device=self.device)
        self.log_probs = torch.empty(allocated, device=self.device)
        self.values = torch.empty(allocated, device=self.device)
        if old is not None:
            kept = min(self.storage_rows, rows)
            for new, previous in zip(
                (self.states, self.masks, self.actions, self.log_probs, self.values),
                old,
            ):
                new[:kept].copy_(previous[:kept])
        self.storage_rows = rows
        self._graphs.clear()
        self._pool = None

    # ------------------------------------------------------------------
    # Staging, capture and replay
    # ------------------------------------------------------------------
    def bucket(self, rows: int) -> int:
        """Smallest graph batch size holding ``rows`` rows per slot."""
        if rows > self.capacity:
            raise ValueError(f"{rows} rows exceed capacity {self.capacity}")
        for size in self.buckets:
            if size >= rows:
                return size
        return self.capacity

    def stage_masks(
        self, masks: np.ndarray, rows: np.ndarray, slot: int, bucket: int
    ) -> None:
        """Stage ``masks[rows]`` for ``slot`` as the policy sees them."""
        stage_effective_masks(
            masks,
            rows,
            staged_rows(self.host_masks, slot, bucket, len(rows)),
            self.play_start,
            self.play_end,
            self.attack_player,
            END_TURN_INDEX,
        )

    @staticmethod
    def _select(logits: torch.Tensor, temperature: float, out: torch.Tensor):
        """Choose into ``out`` as ``select_actions`` does; returns the
        log-probabilities of the distribution sampled, or ``None`` at 0.

        A single categorical draw is ``argmax(p / q)`` with ``q`` standard
        exponential noise, the algorithm ``torch.multinomial`` uses for one
        sample, without its host-side input checks.
        """
        if temperature == 0.0:
            torch.argmax(logits, dim=-1, out=out)
            return None
        if temperature != 1.0:
            logits = logits / temperature
        log_probs = torch.log_softmax(logits, dim=-1)
        probabilities = log_probs.exp()
        noise = torch.empty_like(probabilities).exponential_()
        torch.argmax(probabilities.div_(noise), dim=-1, out=out)
        return log_probs

    def _decide(self, bucket: int, slots: int) -> None:
        rows = slots * bucket
        observations = self._observations[:rows].view(
            slots, bucket, self.state_dim
        )
        masks = self._masks[:rows].view(slots, bucket, self.action_dim)
        logits, values = self.weights.forward(observations)
        logits = torch.where(masks, logits, float("-inf"))
        actions = self._actions[:rows].view(slots, bucket)
        if slots == 1 or self.learner_temperature == self.opponent_temperature:
            log_probs = self._select(logits, self.learner_temperature, actions)
        else:
            log_probs = self._select(
                logits[:1], self.learner_temperature, actions[:1]
            )
            self._select(logits[1:], self.opponent_temperature, actions[1:])
        if self.storage_rows:
            index = self._offset + self._row_index[:bucket]
            self.states.index_copy_(0, index, observations[0])
            self.masks.index_copy_(0, index, masks[0])
            self.actions.index_copy_(0, index, actions[0])
            chosen = (
                log_probs[0].gather(-1, actions[0].unsqueeze(-1)).squeeze(-1)
                if log_probs is not None
                else torch.zeros_like(values[0])
            )
            self.log_probs.index_copy_(0, index, chosen)
            self.values.index_copy_(0, index, values[0])

    def _graph(self, bucket: int, slots: int) -> torch.cuda.CUDAGraph:
        key = (bucket, slots)
        graph = self._graphs.get(key)
        if graph is not None:
            return graph
        if self.storage_rows:
            # Warm-up writes land past the recorded rows, which later
            # replays overwrite.
            self._offset.fill_(self.storage_rows)
        side = _warmup_stream(self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(2):
                self._decide(bucket, slots)
        torch.cuda.current_stream(self.device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        with torch.cuda.graph(graph, pool=self._pool), torch.no_grad():
            self._decide(bucket, slots)
        self._graphs[key] = graph
        return graph

    def submit(
        self,
        bucket: int,
        counts: list[int],
        *,
        offset: int = 0,
    ) -> None:
        """Copy staged rows in, replay one graph, copy every slot's actions out.

        ``counts[s]`` is the number of rows staged for slot ``s`` (missing
        trailing slots have none); with rollout storage the learner's rows
        are recorded from ``offset``. Asynchronous: the caller synchronizes
        the stream before reading ``host_actions`` or restaging rows.
        """
        if len(counts) > self.slots:
            raise ValueError("more row counts than slots")
        if max(counts, default=0) > bucket:
            raise ValueError("a slot has more rows than the bucket")
        if self.storage_rows and not 0 <= offset <= self.storage_rows - (
            counts[0] if counts else 0
        ):
            raise ValueError("rollout storage offset out of range")
        slots = self.slots if any(counts[1:]) else 1
        graph = self._graph(bucket, slots)
        for slot, count in enumerate(counts[:slots]):
            if not count:
                continue
            start = slot * bucket
            self._observations[start : start + count].copy_(
                self.host_observations[start : start + count], non_blocking=True
            )
            self._masks[start : start + count].copy_(
                self.host_masks[start : start + count], non_blocking=True
            )
        if self.storage_rows:
            self.host_offset[0] = offset
            self._offset.copy_(self.host_offset, non_blocking=True)
        graph.replay()
        rows = slots * bucket
        self.host_actions[:rows].copy_(self._actions[:rows], non_blocking=True)

    @property
    def captured_graphs(self) -> list[tuple[int, int]]:
        return sorted(self._graphs)


def staged_rows(host: torch.Tensor, slot: int, bucket: int, count: int) -> np.ndarray:
    """NumPy view of slot ``slot``'s first ``count`` staging rows."""
    start = slot * bucket
    return host[start : start + count].numpy()


_WARMUP_STREAMS: dict[int, torch.cuda.Stream] = {}


def _warmup_stream(device: torch.device) -> torch.cuda.Stream:
    """One side stream per device for pre-capture warm-up runs.

    BLAS libraries keep a workspace for every stream they run on and never
    free it (about 75 MB per stream on ROCm), so a fresh stream per capture
    would leak a workspace per graph.
    """
    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    stream = _WARMUP_STREAMS.get(index)
    if stream is None:
        stream = torch.cuda.Stream(torch.device("cuda", index))
        _WARMUP_STREAMS[index] = stream
    return stream
