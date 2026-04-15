"""Centralized GPU inference server for multi-process training.

Runs as a daemon thread in the main process, serving batched model
forward passes to simulation worker processes via multiprocessing Queues.
CUDA kernels release the GIL, so the thread does not block the main process.
"""
import queue
import threading
from dataclasses import dataclass, field

import numpy as np
import torch
import multiprocessing as mp

from src.ppo.ppo_actor_critic import PPOActorCritic
from src.encoding.action_encoder import END_TURN_INDEX


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


class InferenceServer:
    """Batched GPU inference server running as a daemon thread.

    Receives InferenceRequests from worker processes via a shared request queue,
    runs the model forward pass on GPU, samples actions, and sends responses
    back via per-worker response queues. Drains the queue to accumulate
    requests from multiple workers into a single forward pass when possible.
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
        self._thread: threading.Thread | None = None
        self._requests_served = 0

    def start(self):
        """Start the inference server thread."""
        self._shutdown.clear()
        self._requests_served = 0
        self._thread = threading.Thread(target=self._serve_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal shutdown and wait for thread to exit."""
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

    @property
    def requests_served(self) -> int:
        return self._requests_served

    def _serve_loop(self):
        """Main inference loop — drains queue and batches across workers.

        Supports multiple models: the training model and optional opponent
        snapshots. Requests are grouped by model_id for batched forward passes.
        """
        self.model.to(self.device)
        self.model.eval()

        # Build model lookup: "training" → main model, snapshot names → clones
        models: dict[str, PPOActorCritic] = {"training": self.model}
        for snap_name, snap_sd in self._opponent_snapshots:
            import copy
            opp_model = copy.deepcopy(self.model)
            opp_model.load_state_dict(snap_sd)
            opp_model.to(self.device)
            opp_model.eval()
            models[snap_name] = opp_model

        while not self._shutdown.is_set():
            # Block for the first request
            try:
                first_req = self.request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if first_req is None:  # shutdown sentinel
                break

            # Drain any additional queued requests (non-blocking)
            requests = [first_req]
            while True:
                try:
                    req = self.request_queue.get_nowait()
                    if req is None:
                        break
                    requests.append(req)
                except queue.Empty:
                    break

            # Group requests by model_id for batched processing
            by_model: dict[str, list] = {}
            for req in requests:
                by_model.setdefault(req.model_id, []).append(req)

            for model_id, model_reqs in by_model.items():
                model = models.get(model_id, self.model)

                all_states = np.concatenate([r.states for r in model_reqs], axis=0)
                all_masks = np.concatenate([r.masks for r in model_reqs], axis=0)

                with torch.no_grad():
                    states_t = torch.from_numpy(all_states).to(self.device)
                    masks_t = torch.from_numpy(all_masks.astype(np.float32)).to(self.device)
                    logits, values = model(states_t)

                    logits = logits.masked_fill(masks_t == 0, float('-inf'))

                    # Safety: ensure every row has at least one valid action
                    valid_counts = (masks_t > 0).sum(dim=-1)
                    bad_rows = (valid_counts == 0).nonzero(as_tuple=True)[0]
                    if len(bad_rows) > 0:
                        logits[bad_rows, END_TURN_INDEX] = 0.0

                    dist = torch.distributions.Categorical(
                        logits=logits, validate_args=False
                    )
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions)

                # Split results back per request and send responses
                actions_np = actions.cpu().numpy().reshape(-1)
                log_probs_np = log_probs.cpu().numpy().reshape(-1)
                values_np = values.cpu().numpy().reshape(-1)

                offset = 0
                for req in model_reqs:
                    batch_size = req.states.shape[0]
                    resp = InferenceResponse(
                        request_id=req.request_id,
                        action_indices=actions_np[offset:offset + batch_size],
                        log_probs=log_probs_np[offset:offset + batch_size],
                        values=values_np[offset:offset + batch_size],
                    )
                    self.response_queues[req.worker_id].put(resp)
                    offset += batch_size
                    self._requests_served += 1
