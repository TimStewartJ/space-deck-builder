"""Shared-memory transport for the worker ↔ inference-server round trip.

Why this exists
---------------
Rollout workers used to ship encoded states to the inference server as numpy
arrays inside ``mp.Queue`` messages. Every decision therefore paid: pickle in
the worker, a pipe write, unpickle in the server thread, then a copy into a
pinned staging buffer. Profiling a 16000-episode update measured ~4.7 GB
moved that way, with the server's staging copy running at well under
100 MB/s because it was reading cold, freshly-unpickled allocations. Staging
plus queue drain accounted for roughly 70 s of an ~81 s server budget, on a
single thread, while twenty worker processes sat 57% blocked waiting for it.

The fix is not a faster copy, it is a copy in a better place. Workers write
their batch straight into a preallocated shared region and send only small
metadata; the server reads that region in place and uploads from it. The one
remaining per-row copy happens on the worker side, where it runs twenty ways
in parallel, rather than on the server thread, where it is serial.

Layout
------
One :class:`~multiprocessing.shared_memory.SharedMemory` block carved into
fixed regions and exposed as numpy views::

    states    [workers, slots, max_rows, state_size]  float32   worker -> server
    masks     [workers, slots, max_rows, action_dim]  uint8     worker -> server
    actions   [workers, slots, max_rows]              int64     server -> worker
    log_probs [workers, slots, max_rows]              float32   server -> worker
    values    [workers, slots, max_rows]              float32   server -> worker

``slots`` is the pipelining depth. Two slots let a worker keep one request in
flight on the GPU while it fills the other, so its CPU work overlaps the
server's instead of alternating with it.

Ordering
--------
The existing request/response queues still carry the happens-before edge: a
worker writes its rows, then enqueues a request; the server writes results,
then enqueues a response; the worker only reads results after receiving that
response. A slot is owned by exactly one side at a time, so no additional
locking is required.
"""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np


@dataclass(frozen=True)
class SharedIOSpec:
    """Everything a child process needs to attach to the shared block.

    Small and picklable — this is what gets passed to worker processes,
    never the buffers themselves.
    """
    name: str
    workers: int
    slots: int
    max_rows: int
    state_size: int
    action_dim: int

    @property
    def nbytes(self) -> int:
        w, s, r = self.workers, self.slots, self.max_rows
        return (
            w * s * r * self.state_size * 4      # states   float32
            + w * s * r * self.action_dim        # masks    uint8
            + w * s * r * 8                      # actions  int64
            + w * s * r * 4                      # log_probs float32
            + w * s * r * 4                      # values   float32
        )


class SharedInferenceBuffers:
    """Typed numpy views over one shared-memory block.

    Create in the parent with ``create()``; attach in a child with
    ``attach()``. The creating process owns the block and must call
    :meth:`close` (which unlinks) when the run finishes.
    """

    def __init__(self, spec: SharedIOSpec, shm: shared_memory.SharedMemory,
                 owner: bool):
        self.spec = spec
        self._shm = shm
        self._owner = owner

        w, s, r = spec.workers, spec.slots, spec.max_rows
        off = 0

        def view(dtype, shape):
            nonlocal off
            count = int(np.prod(shape))
            nbytes = count * np.dtype(dtype).itemsize
            arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf, offset=off)
            off += nbytes
            return arr

        self.states = view(np.float32, (w, s, r, spec.state_size))
        self.masks = view(np.uint8, (w, s, r, spec.action_dim))
        self.actions = view(np.int64, (w, s, r))
        self.log_probs = view(np.float32, (w, s, r))
        self.values = view(np.float32, (w, s, r))

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def create(cls, workers: int, slots: int, max_rows: int,
               state_size: int, action_dim: int) -> "SharedInferenceBuffers":
        spec = SharedIOSpec(
            name="", workers=workers, slots=slots, max_rows=max_rows,
            state_size=state_size, action_dim=action_dim,
        )
        shm = shared_memory.SharedMemory(create=True, size=spec.nbytes)
        spec = SharedIOSpec(
            name=shm.name, workers=workers, slots=slots, max_rows=max_rows,
            state_size=state_size, action_dim=action_dim,
        )
        return cls(spec, shm, owner=True)

    @classmethod
    def attach(cls, spec: SharedIOSpec) -> "SharedInferenceBuffers":
        shm = shared_memory.SharedMemory(name=spec.name, create=False)
        return cls(spec, shm, owner=False)

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release this process's mapping; the creator also unlinks.

        Views are dropped first because ``SharedMemory.close()`` fails on
        Windows while any memoryview still references the block.
        """
        self.states = self.masks = None
        self.actions = self.log_probs = self.values = None
        try:
            self._shm.close()
        except Exception:
            pass
        if self._owner:
            try:
                self._shm.unlink()
            except Exception:
                pass

    def __enter__(self) -> "SharedInferenceBuffers":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
