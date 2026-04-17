"""Centralized, pipelined GPU inference server for multi-process training.

Runs a 3-stage pipeline (ingress → GPU → egress) in the main process, serving
batched model forward passes to simulation worker processes over
multiprocessing queues. Stages run on dedicated Python threads; CUDA kernels
release the GIL so the compute thread executes in parallel with the IPC
threads.

The GPU thread is the only thread that touches the CUDA API. It uses three
CUDA streams (copy-in, compute, copy-out) chained via stream events so that
H2D of batch N+1 overlaps with compute of batch N and D2H of batch N-1.
Host buffers are pre-allocated pinned torch tensors reused from a small pool.

Worker-facing protocol (InferenceRequest / InferenceResponse over mp.Queue)
is unchanged from the pre-pipelined server — callers in mp_sim_worker and
mp_batch_runner do not need to adapt.
"""
import queue
import threading
from dataclasses import dataclass

import numpy as np
import torch
import multiprocessing as mp

from src.ppo.ppo_actor_critic import PPOActorCritic
from src.encoding.action_encoder import END_TURN_INDEX

import logging
_log = logging.getLogger("training")


@dataclass
class InferenceRequest:
    """Worker → InferenceServer: batch of encoded states for GPU inference."""
    worker_id: int
    request_id: int
    states: np.ndarray      # [batch, state_size] float32
    masks: np.ndarray        # [batch, action_dim] uint8 (1=valid, 0=invalid)
    model_id: str = "training"  # which model to use ("training" or snapshot name)


@dataclass
class InferenceResponse:
    """InferenceServer → Worker: sampled actions with log-probs and values."""
    request_id: int
    action_indices: np.ndarray   # [batch] int64
    log_probs: np.ndarray        # [batch] float32
    values: np.ndarray           # [batch] float32


@dataclass
class WorkerResult:
    """Worker → Main: pre-merged rollout tensors (all on CPU)."""
    worker_id: int
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    masks: torch.Tensor | None
    opponent_results: dict       # {opponent_name: [wins, total]}
    episodes_completed: int


@dataclass
class WorkerError:
    """Worker → Main: exception info when a worker crashes."""
    worker_id: int
    error: str


@dataclass
class EvalWorkerResult:
    """Worker → Main: evaluation results (no rollout data)."""
    worker_id: int
    wins: int
    losses: int
    total_steps: int


# Pool size is chosen so that with Q_in / Q_out capacities of 2 we can always
# have one batch being filled by ingress, one in compute, one in egress.
_POOL_SIZE = 3
_QUEUE_CAPACITY = 2
# Small timeout on all blocking queue ops so threads re-check _shutdown.
_STAGE_TIMEOUT = 1.0


class _BufferSlot:
    """Pool entry holding reusable pinned host buffers + device buffers.

    Buffers grow on demand to the largest batch size observed so far and are
    never shrunk. All tensors are allocated lazily on first use so that a
    purely CPU run pays no pinned-memory cost.
    """

    def __init__(self, slot_idx: int, state_size: int, action_dim: int, device: torch.device):
        self.slot_idx = slot_idx
        self.state_size = state_size
        self.action_dim = action_dim
        self.device = device
        self._capacity = 0

        self.host_states: torch.Tensor | None = None   # pinned [cap, state_size] float32
        self.host_masks: torch.Tensor | None = None    # pinned [cap, action_dim] float32
        self.host_actions: torch.Tensor | None = None  # pinned [cap] int64
        self.host_logprobs: torch.Tensor | None = None  # pinned [cap] float32
        self.host_values: torch.Tensor | None = None   # pinned [cap] float32

        self.dev_states: torch.Tensor | None = None    # [cap, state_size] float32
        self.dev_masks: torch.Tensor | None = None     # [cap, action_dim] float32

    def ensure_capacity(self, needed: int) -> None:
        if needed <= self._capacity:
            return
        # Grow by doubling to amortize reallocations.
        new_cap = max(needed, self._capacity * 2 if self._capacity else needed)
        pin = self.device.type == "cuda"
        self.host_states = torch.empty(
            (new_cap, self.state_size), dtype=torch.float32, pin_memory=pin
        )
        self.host_masks = torch.empty(
            (new_cap, self.action_dim), dtype=torch.float32, pin_memory=pin
        )
        self.host_actions = torch.empty((new_cap,), dtype=torch.int64, pin_memory=pin)
        self.host_logprobs = torch.empty((new_cap,), dtype=torch.float32, pin_memory=pin)
        self.host_values = torch.empty((new_cap,), dtype=torch.float32, pin_memory=pin)

        self.dev_states = torch.empty(
            (new_cap, self.state_size), dtype=torch.float32, device=self.device
        )
        self.dev_masks = torch.empty(
            (new_cap, self.action_dim), dtype=torch.float32, device=self.device
        )
        self._capacity = new_cap


@dataclass
class _PendingBatch:
    """Ingress → GPU: a filled input buffer ready for compute."""
    slot: _BufferSlot
    batch_size: int
    model_id: str
    # Per-request slice metadata so egress can rebuild InferenceResponses.
    slices: list[tuple[int, int, int, int]]  # (worker_id, request_id, start, end)


@dataclass
class _CompletedBatch:
    """GPU → Egress: outputs copied into pinned host buffers, event to sync on."""
    slot: _BufferSlot
    batch_size: int
    slices: list[tuple[int, int, int, int]]
    done_event: torch.cuda.Event | None  # None when running on CPU
    # CUDA events bracketing forward+sample on compute_stream; used by egress
    # to attribute GPU compute time. None on the CPU path.
    compute_start: torch.cuda.Event | None = None
    compute_end: torch.cuda.Event | None = None


class InferenceServer:
    """Batched GPU inference server with a 3-stage pipeline.

    Public API matches the single-threaded server it replaces: ``start()`` /
    ``stop()`` and the ``request_queue`` + ``response_queues`` used by workers.

    Internally, three threads cooperate:

    * **Ingress**: drains ``request_queue`` (one blocking get, then non-blocking
      drain to coalesce concurrent workers), groups requests by ``model_id``,
      and stages their states/masks into a pooled pinned host buffer.
    * **GPU**: the sole CUDA caller. Issues async H2D on ``copy_in_stream``,
      runs forward + masked sampling on ``compute_stream``, and async D2H on
      ``copy_out_stream``, chained via ``wait_stream`` + ``record_stream``.
    * **Egress**: syncs the copy-out event, slices the pinned output buffer
      per request, and puts ``InferenceResponse`` onto each worker's queue.
    """

    def __init__(
        self,
        model: PPOActorCritic,
        device: torch.device,
        num_workers: int,
        ctx: mp.context.BaseContext | None = None,
        opponent_snapshots: list[tuple[str, dict]] | None = None,
    ):
        self.model = model
        self.device = device
        self._opponent_snapshots = opponent_snapshots or []

        _mp = ctx or mp
        self.request_queue: mp.Queue = _mp.Queue()
        self.response_queues: list[mp.Queue] = [_mp.Queue() for _ in range(num_workers)]

        self._shutdown = threading.Event()
        self._threads: list[threading.Thread] = []
        self._requests_served = 0
        # _requests_served is written only by the egress thread and read from
        # the main thread via the property after stop(); no lock needed.

        # Inter-stage queues and the pool free-list.
        self._q_in: queue.Queue[_PendingBatch | None] = queue.Queue(maxsize=_QUEUE_CAPACITY)
        self._q_out: queue.Queue[_CompletedBatch | None] = queue.Queue(maxsize=_QUEUE_CAPACITY)
        self._free_slots: queue.Queue[_BufferSlot] = queue.Queue()

        self._state_size: int | None = None
        self._action_dim: int | None = None

    def start(self):
        """Spin up ingress/gpu/egress threads."""
        self._shutdown.clear()
        self._requests_served = 0
        self._fatal_error: BaseException | None = None
        # Per-run instrumentation, written only by the GPU/egress threads and
        # read from the main thread after stop(). Plain dict is fine: the GIL
        # makes individual dict ops atomic and stop() joins before we read.
        self._stats = {
            "batches": 0,
            "total_batch_size": 0,
            "max_batch_size": 0,
            "q_in_wait_s": 0.0,    # GPU thread idle waiting for ingress
            "q_out_wait_s": 0.0,   # GPU thread blocked by egress backpressure
            "compute_s": 0.0,      # GPU-side forward + sample (from CUDA events)
            "wall_s": 0.0,         # GPU loop total elapsed
        }

        ingress = threading.Thread(
            target=self._stage_wrapper, args=(self._ingress_loop, "ingress"),
            name="infer-ingress", daemon=True,
        )
        gpu = threading.Thread(
            target=self._stage_wrapper, args=(self._gpu_loop, "gpu"),
            name="infer-gpu", daemon=True,
        )
        egress = threading.Thread(
            target=self._stage_wrapper, args=(self._egress_loop, "egress"),
            name="infer-egress", daemon=True,
        )
        self._threads = [ingress, gpu, egress]
        for t in self._threads:
            t.start()

    def _stage_wrapper(self, target, name: str):
        """Run a stage loop, logging any exception and triggering shutdown so
        the other stages (and the workers blocked on response queues) don't hang.
        """
        try:
            target()
        except BaseException as e:
            import traceback
            self._fatal_error = e
            print(
                f"[InferenceServer] {name} thread crashed: {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}",
                flush=True,
            )
            self._shutdown.set()
            # Unblock any workers currently waiting on response_queue.get() by
            # draining pending requests and sending empty InferenceResponses.
            # Without this, workers hang forever and _collect_results in the
            # main process only times out after its (multi-minute) deadline.
            self._drain_and_fail_pending()

    def _drain_and_fail_pending(self):
        import time as _time
        deadline = _time.monotonic() + 2.0
        while _time.monotonic() < deadline:
            try:
                req = self.request_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if req is None:
                continue
            try:
                n = int(req.states.shape[0])
                resp = InferenceResponse(
                    request_id=req.request_id,
                    action_indices=np.full((n,), END_TURN_INDEX, dtype=np.int64),
                    log_probs=np.zeros((n,), dtype=np.float32),
                    values=np.zeros((n,), dtype=np.float32),
                )
                self.response_queues[req.worker_id].put(resp)
            except Exception:
                pass

    def stop(self):
        """Signal shutdown and wait for all stages to drain."""
        self._shutdown.set()
        for t in self._threads:
            t.join(timeout=10.0)
        self._threads = []
        self._log_stats()

    def _log_stats(self):
        s = self._stats
        if s["batches"] == 0:
            return
        compute = s["compute_s"]
        wait_in = s["q_in_wait_s"]
        wait_out = s["q_out_wait_s"]
        wall = s["wall_s"]
        # GPU-thread utilization: fraction of the loop spent actually computing
        # vs blocked on either inter-stage queue. Compute is measured via CUDA
        # events so it reflects kernel time, not CPU dispatch time.
        denom = max(1e-9, compute + wait_in + wait_out)
        util = compute / denom
        avg_bs = s["total_batch_size"] / s["batches"]
        _log.info(
            "[InferenceServer] batches=%d avg_bs=%.1f max_bs=%d "
            "compute=%.2fs q_in_wait=%.2fs q_out_wait=%.2fs wall=%.2fs "
            "gpu_thread_util=%.1f%%",
            s["batches"], avg_bs, s["max_batch_size"],
            compute, wait_in, wait_out, wall, util * 100.0,
        )

    @property
    def requests_served(self) -> int:
        return self._requests_served

    # ------------------------------------------------------------------
    # Stage 1: ingress — unpickle, group, stage into pinned host buffers
    # ------------------------------------------------------------------
    def _ingress_loop(self):
        try:
            while not self._shutdown.is_set():
                try:
                    first_req = self.request_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if first_req is None:
                    break

                # Opportunistic drain to coalesce across workers.
                requests = [first_req]
                while True:
                    try:
                        req = self.request_queue.get_nowait()
                        if req is None:
                            break
                        requests.append(req)
                    except queue.Empty:
                        break

                # Lazy-init shape metadata from the first observed request.
                if self._state_size is None:
                    self._state_size = int(requests[0].states.shape[1])
                    self._action_dim = int(requests[0].masks.shape[1])
                    self._populate_pool()

                # Group requests by model_id; each group becomes one batch.
                by_model: dict[str, list[InferenceRequest]] = {}
                for req in requests:
                    by_model.setdefault(req.model_id, []).append(req)

                for model_id, model_reqs in by_model.items():
                    batch = self._stage_group(model_id, model_reqs)
                    if batch is None:
                        return
                    while not self._shutdown.is_set():
                        try:
                            self._q_in.put(batch, timeout=_STAGE_TIMEOUT)
                            break
                        except queue.Full:
                            continue
                    if self._shutdown.is_set():
                        # Return the staged slot to the pool so other stages
                        # don't wait on a slot that will never be consumed.
                        self._free_slots.put(batch.slot)
                        return
        finally:
            # Wake the GPU thread even on abnormal exit.
            try:
                self._q_in.put(None, timeout=_STAGE_TIMEOUT)
            except queue.Full:
                pass

    def _populate_pool(self):
        for i in range(_POOL_SIZE):
            slot = _BufferSlot(i, self._state_size, self._action_dim, self.device)
            self._free_slots.put(slot)

    def _stage_group(
        self, model_id: str, model_reqs: list[InferenceRequest]
    ) -> _PendingBatch | None:
        """Copy numpy states/masks from each request into a pooled pinned buffer."""
        total = sum(int(r.states.shape[0]) for r in model_reqs)

        slot: _BufferSlot | None = None
        while slot is None and not self._shutdown.is_set():
            try:
                slot = self._free_slots.get(timeout=_STAGE_TIMEOUT)
            except queue.Empty:
                continue
        if slot is None:
            return None

        slot.ensure_capacity(total)

        offset = 0
        slices: list[tuple[int, int, int, int]] = []
        for req in model_reqs:
            n = int(req.states.shape[0])
            # Copy directly into the pinned torch tensor — using the torch
            # tensor (not a numpy view) preserves the pinned-memory fast path
            # for the subsequent non_blocking H2D.
            slot.host_states[offset:offset + n].copy_(torch.from_numpy(req.states))
            slot.host_masks[offset:offset + n].copy_(
                torch.from_numpy(req.masks.astype(np.float32, copy=False))
            )
            slices.append((req.worker_id, req.request_id, offset, offset + n))
            offset += n

        return _PendingBatch(slot=slot, batch_size=total, model_id=model_id, slices=slices)

    # ------------------------------------------------------------------
    # Stage 2: GPU — async H2D, forward+sample, async D2H on three streams
    # ------------------------------------------------------------------
    def _gpu_loop(self):
        self.model.to(self.device)
        self.model.eval()

        # Build model lookup: training model + optional opponent snapshot clones.
        # Snapshots come as (name, state_dict, model_config) tuples from the
        # opponent pool; model_config is unused here since we deepcopy the
        # training model's architecture.
        models: dict[str, PPOActorCritic] = {"training": self.model}
        for snap in self._opponent_snapshots:
            snap_name, snap_sd = snap[0], snap[1]
            import copy
            opp_model = copy.deepcopy(self.model)
            opp_model.load_state_dict(snap_sd)
            opp_model.to(self.device)
            opp_model.eval()
            models[snap_name] = opp_model

        use_cuda = self.device.type == "cuda"
        copy_in_stream = torch.cuda.Stream(device=self.device) if use_cuda else None
        compute_stream = torch.cuda.Stream(device=self.device) if use_cuda else None
        copy_out_stream = torch.cuda.Stream(device=self.device) if use_cuda else None

        import time as _time
        loop_start = _time.perf_counter()
        try:
            while not self._shutdown.is_set():
                t0 = _time.perf_counter()
                try:
                    pending = self._q_in.get(timeout=_STAGE_TIMEOUT)
                except queue.Empty:
                    # Exclude idle-polling timeouts from starvation accounting
                    # — _STAGE_TIMEOUT waits when truly nothing is happening
                    # would otherwise dominate and mask real starvation signal.
                    continue
                self._stats["q_in_wait_s"] += _time.perf_counter() - t0
                if pending is None:
                    break

                model = models.get(pending.model_id, self.model)
                completed = self._run_batch(
                    pending, model, copy_in_stream, compute_stream, copy_out_stream
                )

                t1 = _time.perf_counter()
                put_ok = False
                while not self._shutdown.is_set():
                    try:
                        self._q_out.put(completed, timeout=_STAGE_TIMEOUT)
                        put_ok = True
                        break
                    except queue.Full:
                        continue
                self._stats["q_out_wait_s"] += _time.perf_counter() - t1
                if not put_ok and self._shutdown.is_set():
                    # Synchronously drain and return the slot.
                    if completed.done_event is not None:
                        completed.done_event.synchronize()
                    self._free_slots.put(completed.slot)
                    return
        finally:
            self._stats["wall_s"] = _time.perf_counter() - loop_start
            try:
                self._q_out.put(None, timeout=_STAGE_TIMEOUT)
            except queue.Full:
                pass

    def _run_batch(
        self,
        pending: _PendingBatch,
        model: PPOActorCritic,
        copy_in_stream,
        compute_stream,
        copy_out_stream,
    ) -> _CompletedBatch:
        slot = pending.slot
        n = pending.batch_size

        if copy_in_stream is None:
            # CPU path — no streams, no pinned transfers, just run inline.
            with torch.no_grad():
                states_t = slot.host_states[:n]
                masks_t = slot.host_masks[:n]
                actions, log_probs, values = _forward_and_sample(model, states_t, masks_t)
                slot.host_actions[:n].copy_(actions)
                slot.host_logprobs[:n].copy_(log_probs)
                slot.host_values[:n].copy_(values)
            return _CompletedBatch(
                slot=slot, batch_size=n, slices=pending.slices, done_event=None,
            )

        # GPU path — three streams, event-chained, using record_stream to keep
        # producer tensors alive until copy_out_stream has finished reading.
        with torch.no_grad():
            with torch.cuda.stream(copy_in_stream):
                slot.dev_states[:n].copy_(slot.host_states[:n], non_blocking=True)
                slot.dev_masks[:n].copy_(slot.host_masks[:n], non_blocking=True)

            compute_stream.wait_stream(copy_in_stream)
            # Bracket forward+sample with timing events on compute_stream so we
            # can attribute kernel time (not CPU dispatch time) per batch.
            compute_start = torch.cuda.Event(enable_timing=True)
            compute_end = torch.cuda.Event(enable_timing=True)
            compute_start.record(compute_stream)
            with torch.cuda.stream(compute_stream):
                dev_states = slot.dev_states[:n]
                dev_masks = slot.dev_masks[:n]
                logits, values = model(dev_states)

                logits = logits.masked_fill(dev_masks == 0, float('-inf'))

                # Safety: rows with zero valid actions would produce all-(-inf)
                # logits and break Categorical. Force END_TURN to 0 for those
                # rows using a data-parallel mask so no CPU↔GPU sync is needed
                # (nonzero()/len() would serialize the pipeline).
                no_valid = (dev_masks > 0).sum(dim=-1) == 0  # [N]
                end_col = logits[:, END_TURN_INDEX]
                logits[:, END_TURN_INDEX] = torch.where(
                    no_valid, torch.zeros_like(end_col), end_col
                )

                dist = torch.distributions.Categorical(
                    logits=logits, validate_args=False
                )
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                values = values.reshape(-1)
            compute_end.record(compute_stream)

            # Tell the allocator these tensors are also read on copy_out_stream,
            # so their storage is not reclaimed until that stream catches up.
            actions.record_stream(copy_out_stream)
            log_probs.record_stream(copy_out_stream)
            values.record_stream(copy_out_stream)

            copy_out_stream.wait_stream(compute_stream)
            with torch.cuda.stream(copy_out_stream):
                slot.host_actions[:n].copy_(actions, non_blocking=True)
                slot.host_logprobs[:n].copy_(log_probs, non_blocking=True)
                slot.host_values[:n].copy_(values, non_blocking=True)
                done_event = torch.cuda.Event()
                done_event.record(copy_out_stream)

        return _CompletedBatch(
            slot=slot, batch_size=n, slices=pending.slices, done_event=done_event,
            compute_start=compute_start, compute_end=compute_end,
        )

    # ------------------------------------------------------------------
    # Stage 3: egress — wait on D2H event, slice pinned output, dispatch
    # ------------------------------------------------------------------
    def _egress_loop(self):
        while not self._shutdown.is_set():
            try:
                completed = self._q_out.get(timeout=_STAGE_TIMEOUT)
            except queue.Empty:
                continue
            if completed is None:
                break

            if completed.done_event is not None:
                completed.done_event.synchronize()

            # After sync, compute_start/compute_end are guaranteed complete
            # (compute runs strictly before copy_out). Safe to read elapsed.
            if completed.compute_start is not None and completed.compute_end is not None:
                compute_ms = completed.compute_start.elapsed_time(completed.compute_end)
                self._stats["compute_s"] += compute_ms / 1000.0
            self._stats["batches"] += 1
            self._stats["total_batch_size"] += completed.batch_size
            if completed.batch_size > self._stats["max_batch_size"]:
                self._stats["max_batch_size"] = completed.batch_size

            slot = completed.slot
            n = completed.batch_size

            # numpy() on a pinned host tensor shares memory; copy defensively
            # so the buffer slot is safe to recycle before the worker unpickles.
            actions_np = slot.host_actions[:n].numpy().copy()
            logprobs_np = slot.host_logprobs[:n].numpy().copy()
            values_np = slot.host_values[:n].numpy().copy()

            for worker_id, request_id, start, end in completed.slices:
                resp = InferenceResponse(
                    request_id=request_id,
                    action_indices=actions_np[start:end],
                    log_probs=logprobs_np[start:end],
                    values=values_np[start:end],
                )
                self.response_queues[worker_id].put(resp)
                self._requests_served += 1

            self._free_slots.put(slot)


def _forward_and_sample(model: PPOActorCritic, states: torch.Tensor, masks: torch.Tensor):
    """CPU fallback path: run forward + masked categorical sample inline."""
    logits, values = model(states)
    logits = logits.masked_fill(masks == 0, float('-inf'))
    no_valid = (masks > 0).sum(dim=-1) == 0
    end_col = logits[:, END_TURN_INDEX]
    logits[:, END_TURN_INDEX] = torch.where(no_valid, torch.zeros_like(end_col), end_col)
    dist = torch.distributions.Categorical(logits=logits, validate_args=False)
    actions = dist.sample()
    log_probs = dist.log_prob(actions)
    return actions, log_probs, values.reshape(-1)
