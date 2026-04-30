# 0001 — Drop the multi-stream InferenceServer pipeline

**Status:** Accepted (2026-04-21)
**Context branch:** `main` (pre-`v0.1-baseline`)

## Context

`InferenceServer` (the centralized batched inference service used by
multi-process training, eval, and Elo tournaments) was originally built
as a 3-stage pipeline running on three CUDA streams:

  * `copy_in_stream` — async H2D of the staged batch
  * `compute_stream` — model forward + masked categorical sample
  * `copy_out_stream` — async D2H of actions / log-probs / values

with cross-stream synchronization via `Stream.wait_stream(...)` and
`record_stream(...)` to keep producer tensors alive across stream
boundaries. Three Python threads (ingress, GPU, egress) drove the
stages and a small pool of pinned-memory buffer slots let work overlap
across batches.

## The bug we hit

On ROCm 7.2 (RX 9070, gfx1201) cross-stream synchronization in this
shape was effectively a no-op. The compute stream began the forward
pass before the H2D copy on `copy_in_stream` had finished, so the model
read uninitialized device memory and produced near-random actions.
Symptoms:

  * `python -m src eval --opponents heuristic` reported ~1% win rate
    for a checkpoint the Elo CLI scored at 76–78%.
  * `BatchRunner` (single-process, no streams) was correct (~74%).
  * The CPU code path through the same server (no streams used) was
    correct (~82%).
  * Forcing `torch.cuda.synchronize()` after the H2D copy on the GPU
    path also restored correctness (~74%).
  * Adding an explicit `Stream.wait_event(...)` instead of
    `wait_stream(...)` did **not** help — both forms of cross-stream
    sync are unreliable on this combination.

The bug was masked during training because rollout collection used the
same broken PPO inference path that the policy update was optimizing
against. In self-play runs, PPO opponents would share that same path as
well. Only when we ran the standalone `eval` CLI against a fixed
non-PPO opponent did the floor drop out.

## Decision

**Tear out the multi-stream pipeline. The InferenceServer is now
single-thread, single-stream.**

The server runs one dedicated thread that drains the request queue
(blocking on the first request, opportunistically draining the rest),
groups by `model_id`, stages into a single growing pinned-host /
device buffer set, runs forward + sampling on the default stream, and
dispatches responses. There are no inter-stage queues, no per-batch
events, no `record_stream`/`wait_stream`, and no buffer pool.

Specifically removed:

  * `copy_in_stream`, `compute_stream`, `copy_out_stream`
  * `_q_in`, `_q_out`, `_free_slots` queues
  * Ingress and egress threads (the GPU thread does both jobs)
  * `_PendingBatch.done_event`, `_CompletedBatch` (and per-batch
    `compute_start`/`compute_end` events on separate streams)
  * `_BufferSlot` pool (replaced by a single `_Buffers` instance)
  * GPU-thread instrumentation that only made sense in a pipelined
    world (`q_in_wait_s`, `q_out_wait_s`)

Pinned host memory is kept because synchronous H2D from pinned memory
is still meaningfully faster than from pageable memory.

Public API is unchanged: `InferenceServer(models, device, num_workers,
ctx)`, `start()`/`stop()`, `request_queue` / `response_queues`,
`register_model`/`unregister_model`, `requests_served`, plus the
unchanged `InferenceRequest` / `InferenceResponse` wire format.

## Consequences

**Pro**

  * Correctness on ROCm (and on every other backend, since the bug
    was a latent assumption about cross-stream sync semantics).
  * ~4x less code in `mp_inference_server.py`, no thread-coordination
    state machine to reason about.
  * One thread = one source of truth for CUDA ops, which already had
    to be enforced manually by routing all `register_model` /
    `unregister_model` calls through the GPU thread.

**Con**

  * No more H2D / compute / D2H overlap. For our actual workload
    (batch sizes in the low tens to low hundreds, single-GPU,
    AMD RX 9070) the GPU is not the bottleneck — CPU-side game
    simulation in worker processes dominates wall-clock — so this
    overlap was never the throughput limiter.

If we ever do hit a regime where overlap matters (much larger batches
or much faster CPU sim), the right fix is **not** to reintroduce
cross-stream sync; it is either to use CUDA Graphs (which serialize
H2D + compute + D2H by construction) or to run multiple independent
server threads each with their own model replica.

## Verification

After the rewrite, the standalone eval CLI reports the same win rates
as the in-process `BatchRunner` for the same checkpoint and seed. The
gauntlet (`eval_protocol.md`) is the long-term regression test.
