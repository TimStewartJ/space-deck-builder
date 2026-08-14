"""Centralized GPU inference server for multi-process training.

Worker processes (training rollouts, eval, tournaments) ship encoded
states + action masks to this server over an :class:`mp.Queue` and
receive sampled actions, log-probs, and values back over per-worker
response queues. Centralizing inference lets us run one batched forward
pass per server tick instead of one per worker, which is a large
speedup whenever workers contend for the GPU.

Single-thread / single-stream design
------------------------------------
A single dedicated server thread is the only thread that ever touches
the CUDA API. It runs the obvious loop:

  1. drain pending control messages (register / unregister)
  2. drain ``request_queue`` (block on first, opportunistically drain rest)
  3. group requests by ``model_id`` and stage them into reusable buffers
  4. run forward + sampling on the default stream
  5. dispatch :class:`InferenceResponse` per slice

There is no multi-stream pipeline and no inter-stage queue. We previously
used a 3-stream / 3-thread setup (copy-in, compute, copy-out) chained via
``Stream.wait_stream`` and CUDA events, but cross-stream sync proved
unreliable on ROCm 7.2 (RX 9070), causing compute to read uninitialized
device memory and produce garbage actions. See ``docs/decisions/0001-drop-multistream-inference.md``.

Worker-facing protocol (:class:`InferenceRequest` / :class:`InferenceResponse`
over :class:`mp.Queue`) is unchanged.
"""
import queue
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.ppo.shared_io import SharedInferenceBuffers
import torch
import multiprocessing as mp

from src.ppo.ppo_actor_critic import PPOActorCritic
from src.encoding.action_encoder import END_TURN_INDEX

import logging
_log = logging.getLogger("training")


@dataclass
class InferenceRequest:
    """Worker → InferenceServer: a batch of encoded states awaiting inference.

    The payload normally lives in shared memory: the worker has already
    written rows ``[row_offset, row_offset + n_rows)`` of its ``(worker_id,
    slot)`` region, and this message only carries the coordinates. Shipping
    the arrays through the queue instead cost more in pickle and staging than
    the inference itself.

    ``states``/``masks`` are the inline fallback for callers without a shared
    block (single-process paths and tests). When they are set the server uses
    them directly and ignores the shared region.
    """
    worker_id: int
    request_id: int
    model_id: str = "training"
    slot: int = 0
    row_offset: int = 0
    n_rows: int = 0
    # When True, the response carries the masked pre-sample logits for every
    # row so the worker can compute top-k probabilities and policy entropy
    # for replay-style analysis. Off by default to keep the regular eval /
    # rollout path's wire size unchanged.
    return_logits: bool = False
    states: np.ndarray | None = None     # [batch, state_size] float32, inline fallback
    masks: np.ndarray | None = None      # [batch, action_dim] uint8, inline fallback

    @property
    def batch_size(self) -> int:
        return int(self.states.shape[0]) if self.states is not None else self.n_rows


@dataclass
class InferenceResponse:
    """InferenceServer → Worker: sampled actions with log-probs and values.

    Results are normally written straight into the worker's shared-memory
    slot; this message just says which rows are ready. ``action_indices`` /
    ``log_probs`` / ``values`` are populated only on the inline fallback path
    (no shared block) or when reporting an error.

    If ``error`` is set, the server could not serve the request (typically
    because ``model_id`` was never registered or was unregistered before the
    request arrived). The worker should raise on this field rather than use
    the (zero-filled) output arrays.

    ``logits`` is populated only when the originating request set
    ``return_logits=True``. Shape is ``[batch, action_dim]`` float32, with
    invalid actions already filled with ``-inf`` (i.e. the same masked logits
    used to sample the action). Logits stay inline because only the replay
    analysis path asks for them.
    """
    request_id: int
    slot: int = 0
    row_offset: int = 0
    n_rows: int = 0
    action_indices: np.ndarray | None = None   # [batch] int64, inline fallback
    log_probs: np.ndarray | None = None        # [batch] float32, inline fallback
    values: np.ndarray | None = None           # [batch] float32, inline fallback
    error: str | None = None
    logits: np.ndarray | None = None  # [batch, action_dim] float32, masked


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
    # Per-phase wall time for this worker's rollout loop, in seconds.
    # Aggregated by the runner to show whether workers are simulating or
    # blocked on the inference server. Empty when timing is disabled.
    timings: dict | None = None


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


# Small timeout on blocking queue ops so the server thread re-checks _shutdown.
_LOOP_TIMEOUT = 0.1


# ----------------------------------------------------------------------
# Control plane: register/unregister requests handled by the server thread
# so that only one thread ever touches the CUDA API.
# ----------------------------------------------------------------------
@dataclass
class _ControlMsg:
    op: str                        # "register" | "unregister"
    model_id: str
    model: PPOActorCritic | None   # required for op="register"
    ack: threading.Event
    error: list                    # single-element list written by GPU thread


class _Buffers:
    """Reusable host (pinned on CUDA) and device buffers, grown on demand.

    A single buffer set is enough because the server thread runs one batch
    at a time end-to-end — there is no pipelining that would require
    multiple slots. Buffers double in size on growth and never shrink, so
    after a short warm-up every batch is a no-op allocation-wise.

    Masks are kept as ``uint8`` end to end. ``_forward_and_sample`` only
    ever compares them (``masks == 0`` / ``masks > 0``), so widening them to
    float32 on the host bought nothing and cost a full array allocation plus
    conversion per request — which profiling showed was ~46% of server time
    — as well as 4x the host-to-device bytes.
    """

    def __init__(self, state_size: int, action_dim: int, device: torch.device):
        self.state_size = state_size
        self.action_dim = action_dim
        self.device = device
        self._capacity = 0

        self.host_states: torch.Tensor | None = None    # pinned [cap, state_size] float32
        self.host_masks: torch.Tensor | None = None     # pinned [cap, action_dim] uint8
        self.host_actions: torch.Tensor | None = None   # pinned [cap] int64
        self.host_logprobs: torch.Tensor | None = None  # pinned [cap] float32
        self.host_values: torch.Tensor | None = None    # pinned [cap] float32
        # Lazily allocated only when at least one batch requests logits.
        self.host_logits: torch.Tensor | None = None    # pinned [cap, action_dim] float32

        self.dev_states: torch.Tensor | None = None     # [cap, state_size] float32
        self.dev_masks: torch.Tensor | None = None      # [cap, action_dim] uint8

    def ensure_capacity(self, needed: int) -> None:
        if needed <= self._capacity:
            return
        new_cap = max(needed, self._capacity * 2 if self._capacity else needed)
        pin = self.device.type == "cuda"
        self.host_states = torch.empty(
            (new_cap, self.state_size), dtype=torch.float32, pin_memory=pin
        )
        self.host_masks = torch.empty(
            (new_cap, self.action_dim), dtype=torch.uint8, pin_memory=pin
        )
        self.host_actions = torch.empty((new_cap,), dtype=torch.int64, pin_memory=pin)
        self.host_logprobs = torch.empty((new_cap,), dtype=torch.float32, pin_memory=pin)
        self.host_values = torch.empty((new_cap,), dtype=torch.float32, pin_memory=pin)
        if self.host_logits is not None:
            self.host_logits = torch.empty(
                (new_cap, self.action_dim), dtype=torch.float32, pin_memory=pin
            )
        if self.device.type == "cuda":
            self.dev_states = torch.empty(
                (new_cap, self.state_size), dtype=torch.float32, device=self.device
            )
            self.dev_masks = torch.empty(
                (new_cap, self.action_dim), dtype=torch.uint8, device=self.device
            )
        self._capacity = new_cap

    def ensure_logits_buffer(self) -> None:
        if self.host_logits is not None:
            return
        pin = self.device.type == "cuda"
        self.host_logits = torch.empty(
            (self._capacity, self.action_dim), dtype=torch.float32, pin_memory=pin
        )


class InferenceServer:
    """Single-thread batched inference server hosting one or more PPO models.

    Hosts an arbitrary set of :class:`PPOActorCritic` instances keyed by a
    string ``model_id``. Workers tag each :class:`InferenceRequest` with
    the model they want; unknown ``model_id`` produces an error response
    rather than being silently routed elsewhere. Models can be added or
    removed at runtime via :meth:`register_model` / :meth:`unregister_model`
    without tearing down the server; both calls block until the server
    thread has applied the change so all CUDA ops stay on one thread.

    Public API: ``start()`` / ``stop()``, ``request_queue`` +
    ``response_queues`` for worker IPC, and ``register_model`` /
    ``unregister_model`` for the model registry.
    """

    def __init__(
        self,
        models: dict[str, PPOActorCritic],
        device: torch.device,
        num_workers: int,
        ctx: mp.context.BaseContext | None = None,
        shared_io: "SharedInferenceBuffers | None" = None,
    ):
        if not models:
            raise ValueError("InferenceServer requires at least one model at startup")

        self._initial_models = dict(models)
        self.device = device
        # When present, requests carry shared-memory coordinates instead of
        # arrays and results are written back in place. Falls back to inline
        # arrays when absent (single-process paths and tests).
        self.shared_io = shared_io

        _mp = ctx or mp
        self.request_queue: mp.Queue = _mp.Queue()
        self.response_queues: list[mp.Queue] = [_mp.Queue() for _ in range(num_workers)]

        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._fatal_error: BaseException | None = None
        self._requests_served = 0  # written only by the server thread

        # Control plane: writers block on ack until the server thread
        # applies the change, giving callers a synchronous contract.
        self._control_q: queue.Queue[_ControlMsg] = queue.Queue()
        self._known_model_ids: set[str] = set(models.keys())
        self._known_lock = threading.Lock()

        # Lazy-init from the first observed request shape.
        self._state_size: int | None = None
        self._action_dim: int | None = None

        self._stats = {
            "batches": 0,
            "total_batch_size": 0,
            "max_batch_size": 0,
            "compute_s": 0.0,   # GPU-side forward + sample time (CUDA events)
            "wall_s": 0.0,      # server loop total elapsed
            # Phase breakdown. ``compute_s`` measures only the CUDA-event
            # window around forward+sample, which historically left ~95% of
            # wall time unattributed. These cover the rest of the loop so a
            # slow server can be diagnosed without guessing.
            "blocked_get_s": 0.0,   # waiting for the first request (server starved)
            "drain_s": 0.0,         # opportunistically draining more requests
            "stage_s": 0.0,         # copying request arrays into pinned buffers
            "h2d_s": 0.0,           # host->device copies
            "sync_s": 0.0,          # torch.cuda.synchronize() wait
            "dispatch_s": 0.0,      # building + queueing responses
            "alloc_s": 0.0,         # pinned/device buffer growth
            "staged_rows": 0,       # rows copied into pinned buffers
            "requests": 0,          # individual worker requests served
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self):
        """Spin up the server thread."""
        self._shutdown.clear()
        self._fatal_error = None
        self._requests_served = 0
        for k in self._stats:
            self._stats[k] = 0 if isinstance(self._stats[k], int) else 0.0
        self._thread = threading.Thread(
            target=self._run, name="infer-server", daemon=True,
        )
        self._thread.start()

    def stop(self):
        """Signal shutdown, join the server thread, and release queues.

        See RCA 2026-04-17: ``mp.Queue`` feeder threads will block the
        atexit phase if left open while workers are terminated, because
        partial writes can remain buffered on the parent side.
        """
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        # Release any control-plane callers still waiting on an ack.
        self._abort_pending_controls()
        self._log_stats()
        for q in [self.request_queue, *self.response_queues]:
            try:
                q.close()
            except Exception:
                pass
            try:
                q.cancel_join_thread()
            except Exception:
                pass

    @property
    def requests_served(self) -> int:
        return self._requests_served

    # ------------------------------------------------------------------
    # Control plane — synchronous register / unregister
    # ------------------------------------------------------------------
    def register_model(self, model_id: str, model: PPOActorCritic) -> None:
        """Add a model to the server. Blocks until the server thread has
        moved it to device and made it queryable.
        """
        with self._known_lock:
            if model_id in self._known_model_ids:
                raise ValueError(f"model_id already registered: {model_id!r}")
            self._known_model_ids.add(model_id)
        try:
            self._submit_control("register", model_id, model)
        except Exception:
            # GPU-thread side failed (e.g. OOM during model.to(device), or
            # server shutdown mid-registration). Roll back the bookkeeping
            # so a retry with the same model_id is not falsely rejected.
            with self._known_lock:
                self._known_model_ids.discard(model_id)
            raise

    def unregister_model(self, model_id: str) -> None:
        """Remove a model from the server. Blocks until the server thread
        has dropped its reference.
        """
        with self._known_lock:
            if model_id not in self._known_model_ids:
                raise ValueError(f"model_id not registered: {model_id!r}")
            self._known_model_ids.discard(model_id)
        self._submit_control("unregister", model_id, None)

    def _submit_control(self, op: str, model_id: str, model: PPOActorCritic | None) -> None:
        if self._thread is None:
            raise RuntimeError("InferenceServer is not running; call start() first")
        msg = _ControlMsg(op=op, model_id=model_id, model=model,
                          ack=threading.Event(), error=[])
        self._control_q.put(msg)
        # Periodic wakeups so a crashed server thread doesn't hang us.
        while not msg.ack.wait(timeout=1.0):
            if self._shutdown.is_set():
                raise RuntimeError(
                    f"InferenceServer shut down while processing {op!r} for {model_id!r}"
                )
        if msg.error:
            raise msg.error[0]

    def _drain_control(self, models: dict[str, PPOActorCritic]) -> None:
        while True:
            try:
                msg = self._control_q.get_nowait()
            except queue.Empty:
                return
            try:
                if msg.op == "register":
                    if msg.model is None:
                        raise ValueError("register requires a model instance")
                    msg.model.to(self.device)
                    msg.model.eval()
                    models[msg.model_id] = msg.model
                elif msg.op == "unregister":
                    models.pop(msg.model_id, None)
                else:
                    raise ValueError(f"unknown control op: {msg.op!r}")
            except BaseException as e:
                msg.error.append(e)
            finally:
                msg.ack.set()

    def _abort_pending_controls(self) -> None:
        while True:
            try:
                msg = self._control_q.get_nowait()
            except queue.Empty:
                return
            msg.error.append(
                RuntimeError("InferenceServer shut down before processing control message")
            )
            msg.ack.set()

    # ------------------------------------------------------------------
    # Server thread
    # ------------------------------------------------------------------
    def _run(self):
        try:
            # Move all initial models to device on the server thread so
            # CUDA initialization happens here, not in the calling thread.
            models: dict[str, PPOActorCritic] = {}
            for mid, m in self._initial_models.items():
                m.to(self.device)
                m.eval()
                models[mid] = m
            self._initial_models = {}

            buffers: _Buffers | None = None
            use_cuda = self.device.type == "cuda"
            # Reusable timing events; recreated only if the device changes
            # (it doesn't, in practice).
            compute_start = torch.cuda.Event(enable_timing=True) if use_cuda else None
            compute_end = torch.cuda.Event(enable_timing=True) if use_cuda else None

            loop_start = time.perf_counter()
            while not self._shutdown.is_set():
                self._drain_control(models)

                # Block on the first request so an idle server doesn't busy-loop.
                _t = time.perf_counter()
                try:
                    first_req = self.request_queue.get(timeout=_LOOP_TIMEOUT)
                except queue.Empty:
                    self._stats["blocked_get_s"] += time.perf_counter() - _t
                    continue
                self._stats["blocked_get_s"] += time.perf_counter() - _t
                if first_req is None:
                    break

                # Opportunistically drain to coalesce across workers.
                _t = time.perf_counter()
                requests = [first_req]
                while True:
                    try:
                        req = self.request_queue.get_nowait()
                        if req is None:
                            continue
                        requests.append(req)
                    except queue.Empty:
                        break
                self._stats["drain_s"] += time.perf_counter() - _t
                self._stats["requests"] += len(requests)

                # Lazy-init buffer shapes. With a shared block the geometry is
                # known up front; otherwise take it from the first request's
                # inline arrays.
                if buffers is None:
                    if self.shared_io is not None:
                        self._state_size = self.shared_io.spec.state_size
                        self._action_dim = self.shared_io.spec.action_dim
                    else:
                        self._state_size = int(requests[0].states.shape[1])
                        self._action_dim = int(requests[0].masks.shape[1])
                    buffers = _Buffers(self._state_size, self._action_dim, self.device)

                # Group requests by model_id; each group becomes one batched
                # forward pass.
                by_model: dict[str, list[InferenceRequest]] = {}
                for req in requests:
                    by_model.setdefault(req.model_id, []).append(req)

                for model_id, model_reqs in by_model.items():
                    model = models.get(model_id)
                    if model is None:
                        self._send_error_batch(
                            model_reqs, f"unknown model_id: {model_id!r}",
                        )
                        continue
                    self._serve_group(
                        model, model_reqs, buffers,
                        compute_start, compute_end,
                    )

            self._stats["wall_s"] = time.perf_counter() - loop_start
        except BaseException as e:
            import traceback
            self._fatal_error = e
            print(
                f"[InferenceServer] server thread crashed: {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}",
                flush=True,
            )
            self._shutdown.set()
            # Unblock any workers currently waiting on response_queue.get().
            self._drain_and_fail_pending()
        finally:
            self._abort_pending_controls()

    def _serve_group(
        self,
        model: PPOActorCritic,
        model_reqs: list[InferenceRequest],
        buffers: _Buffers,
        compute_start,
        compute_end,
    ) -> None:
        """Stage one model's requests into ``buffers``, run forward+sample,
        and write results back for each request.

        When a shared block is attached the payload is already resident and
        staging is a copy out of it rather than out of freshly-unpickled
        queue messages, which is where the old server spent most of its time.
        """
        total = sum(r.batch_size for r in model_reqs)
        _t = time.perf_counter()
        buffers.ensure_capacity(total)
        self._stats["alloc_s"] += time.perf_counter() - _t

        any_logits = any(r.return_logits for r in model_reqs)
        if any_logits:
            buffers.ensure_logits_buffer()

        shm = self.shared_io
        host_states_np = buffers.host_states.numpy()
        host_masks_np = buffers.host_masks.numpy()

        # Stage host buffers and remember per-request slice offsets.
        _t = time.perf_counter()
        offset = 0
        slices: list[tuple[InferenceRequest, int, int]] = []
        for req in model_reqs:
            n = req.batch_size
            if req.states is not None:
                host_states_np[offset:offset + n] = req.states
                host_masks_np[offset:offset + n] = req.masks
            else:
                lo, hi = req.row_offset, req.row_offset + n
                host_states_np[offset:offset + n] = shm.states[req.worker_id, req.slot, lo:hi]
                host_masks_np[offset:offset + n] = shm.masks[req.worker_id, req.slot, lo:hi]
            slices.append((req, offset, offset + n))
            offset += n
        self._stats["stage_s"] += time.perf_counter() - _t
        self._stats["staged_rows"] += total

        # Run on the default stream. Synchronous H2D + compute + D2H
        # eliminates the cross-stream-sync class of bugs that bit us on
        # ROCm 7.2 with the previous 3-stream pipeline.
        n = total
        with torch.no_grad():
            if self.device.type == "cuda":
                _t = time.perf_counter()
                buffers.dev_states[:n].copy_(buffers.host_states[:n])
                buffers.dev_masks[:n].copy_(buffers.host_masks[:n])
                self._stats["h2d_s"] += time.perf_counter() - _t
                compute_start.record()
                actions, log_probs, values, masked_logits = _forward_and_sample(
                    model, buffers.dev_states[:n], buffers.dev_masks[:n],
                )
                compute_end.record()
                buffers.host_actions[:n].copy_(actions)
                buffers.host_logprobs[:n].copy_(log_probs)
                buffers.host_values[:n].copy_(values)
                if any_logits:
                    buffers.host_logits[:n].copy_(masked_logits)
                # Single sync flushes H2D + compute + D2H. The default
                # stream serializes all three, so this is the only sync we
                # need before reading host buffers.
                _t = time.perf_counter()
                torch.cuda.synchronize(self.device)
                self._stats["sync_s"] += time.perf_counter() - _t
                self._stats["compute_s"] += compute_start.elapsed_time(compute_end) / 1000.0
            else:
                t0 = time.perf_counter()
                actions, log_probs, values, masked_logits = _forward_and_sample(
                    model, buffers.host_states[:n], buffers.host_masks[:n],
                )
                buffers.host_actions[:n].copy_(actions)
                buffers.host_logprobs[:n].copy_(log_probs)
                buffers.host_values[:n].copy_(values)
                if any_logits:
                    buffers.host_logits[:n].copy_(masked_logits)
                self._stats["compute_s"] += time.perf_counter() - t0

        self._stats["batches"] += 1
        self._stats["total_batch_size"] += n
        if n > self._stats["max_batch_size"]:
            self._stats["max_batch_size"] = n

        # numpy() on a pinned host tensor shares memory, so results are either
        # written straight into the worker's shared slot (no copy leaves this
        # thread) or copied defensively before going inline on the queue.
        _t = time.perf_counter()
        actions_np = buffers.host_actions[:n].numpy()
        logprobs_np = buffers.host_logprobs[:n].numpy()
        values_np = buffers.host_values[:n].numpy()
        logits_np = buffers.host_logits[:n].numpy() if any_logits else None

        for req, start, end in slices:
            if req.states is None and shm is not None:
                lo, hi = req.row_offset, req.row_offset + (end - start)
                w, s = req.worker_id, req.slot
                shm.actions[w, s, lo:hi] = actions_np[start:end]
                shm.log_probs[w, s, lo:hi] = logprobs_np[start:end]
                shm.values[w, s, lo:hi] = values_np[start:end]
                resp = InferenceResponse(
                    request_id=req.request_id,
                    slot=req.slot,
                    row_offset=req.row_offset,
                    n_rows=end - start,
                    logits=(logits_np[start:end].copy() if req.return_logits else None),
                )
            else:
                resp = InferenceResponse(
                    request_id=req.request_id,
                    n_rows=end - start,
                    action_indices=actions_np[start:end].copy(),
                    log_probs=logprobs_np[start:end].copy(),
                    values=values_np[start:end].copy(),
                    logits=(logits_np[start:end].copy() if req.return_logits else None),
                )
            self.response_queues[req.worker_id].put(resp)
            self._requests_served += 1
        self._stats["dispatch_s"] += time.perf_counter() - _t

    def _send_error_batch(self, model_reqs: list[InferenceRequest], err: str) -> None:
        """Emit zero-filled error responses for every request in the group."""
        n_action = self._action_dim or 1
        for req in model_reqs:
            n = req.batch_size
            resp = InferenceResponse(
                request_id=req.request_id,
                slot=req.slot,
                row_offset=req.row_offset,
                n_rows=n,
                action_indices=np.full((n,), END_TURN_INDEX, dtype=np.int64),
                log_probs=np.zeros((n,), dtype=np.float32),
                values=np.zeros((n,), dtype=np.float32),
                error=err,
                logits=(
                    np.zeros((n, n_action), dtype=np.float32)
                    if req.return_logits else None
                ),
            )
            self.response_queues[req.worker_id].put(resp)
            self._requests_served += 1

    def _drain_and_fail_pending(self):
        """After a crash, send error responses for any queued requests so
        workers blocked on ``response_queue.get()`` don't hang until their
        own deadlines fire (which can be many minutes).
        """
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                req = self.request_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if req is None:
                continue
            try:
                # ``batch_size`` handles both transports; dereferencing
                # ``req.states`` directly would raise for shared-memory
                # requests, and the except below would swallow it — leaving
                # the worker blocked on ``response_queue.get()`` forever,
                # which is exactly what this method exists to prevent.
                n = req.batch_size
                resp = InferenceResponse(
                    request_id=req.request_id,
                    slot=req.slot,
                    row_offset=req.row_offset,
                    n_rows=n,
                    action_indices=np.full((n,), END_TURN_INDEX, dtype=np.int64),
                    log_probs=np.zeros((n,), dtype=np.float32),
                    values=np.zeros((n,), dtype=np.float32),
                    error=f"InferenceServer crashed: {self._fatal_error!r}",
                )
                self.response_queues[req.worker_id].put(resp)
            except Exception:
                pass

    def _log_stats(self):
        s = self._stats
        if s["batches"] == 0:
            return
        avg_bs = s["total_batch_size"] / s["batches"]
        _log.info(
            "[InferenceServer] batches=%d avg_bs=%.1f max_bs=%d "
            "compute=%.2fs wall=%.2fs",
            s["batches"], avg_bs, s["max_batch_size"],
            s["compute_s"], s["wall_s"],
        )
        wall = s["wall_s"] or 1e-9
        accounted = (
            s["blocked_get_s"] + s["drain_s"] + s["stage_s"] + s["h2d_s"]
            + s["compute_s"] + s["sync_s"] + s["dispatch_s"] + s["alloc_s"]
        )
        row_bytes = (self._state_size or 0) * 4 + (self._action_dim or 0)
        stage_mb = s["staged_rows"] * row_bytes / 1e6
        stage_rate = stage_mb / s["stage_s"] if s["stage_s"] > 0 else 0.0
        _log.info(
            "[InferenceServer] phases: idle=%.1fs drain=%.1fs stage=%.1fs "
            "h2d=%.1fs kernel=%.1fs sync=%.1fs dispatch=%.1fs alloc=%.1fs "
            "| accounted=%.0f%% reqs=%d stage=%.0fMB@%.0fMB/s",
            s["blocked_get_s"], s["drain_s"], s["stage_s"], s["h2d_s"],
            s["compute_s"], s["sync_s"], s["dispatch_s"], s["alloc_s"],
            accounted / wall * 100.0, s["requests"], stage_mb, stage_rate,
        )


def _forward_and_sample(model: PPOActorCritic, states: torch.Tensor, masks: torch.Tensor):
    """Run forward + masked categorical sample.

    Returns ``(actions, log_probs, values, masked_logits)``. ``masked_logits``
    is the post-mask, post-no-valid-row-fix logits tensor used for sampling;
    callers that don't need it can ignore it.
    """
    logits, values = model(states)
    logits = logits.masked_fill(masks == 0, float('-inf'))
    # Safety: rows with zero valid actions would produce all-(-inf) logits
    # and break Categorical. Force END_TURN to 0 for those rows using a
    # data-parallel mask so no CPU↔GPU sync is needed.
    no_valid = (masks > 0).sum(dim=-1) == 0
    end_col = logits[:, END_TURN_INDEX]
    logits[:, END_TURN_INDEX] = torch.where(no_valid, torch.zeros_like(end_col), end_col)
    dist = torch.distributions.Categorical(logits=logits, validate_args=False)
    actions = dist.sample()
    log_probs = dist.log_prob(actions)
    return actions, log_probs, values.reshape(-1), logits
