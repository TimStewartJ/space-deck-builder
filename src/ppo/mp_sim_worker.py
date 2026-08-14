"""Simulation worker process for multi-process training.

Each worker runs game simulation and state encoding on CPU, sending
encoded states to the InferenceServer for GPU inference via
multiprocessing Queues. Workers merge their own rollout data locally
and return a single pre-merged result to the main process.
"""
import random
import time as _time
import traceback
from typing import Optional

import numpy as np
import torch
import multiprocessing as mp

from src.config import DataConfig, PPOConfig
from src.engine.game import Game
from src.engine.actions import ActionType
from src.ai.agent import Agent
from src.ai.random_agent import RandomAgent
from src.ai.simple_agent import SimpleAgent
from src.ai.heuristic_agent import HeuristicAgent
from src.cards.card import Card
from src.cards.registry import CardRegistry
from src.encoding.state_encoder import encode_state, get_state_size
from src.encoding.action_encoder import get_action_space_size, END_TURN_INDEX
from src.encoding.action_context import build_action_context, ActionContext
from src.ppo.rollout_buffer import RolloutBuffer
from src.ppo.mp_inference_server import (
    InferenceRequest, InferenceResponse, WorkerResult, WorkerError,
    EvalWorkerResult,
)
from src.utils.logger import set_disabled


# Agent factory registry (mirrors opponent_pool.AGENT_REGISTRY)
_AGENT_FACTORIES = {
    "random":    lambda name: RandomAgent(name),
    "heuristic": lambda name: HeuristicAgent(name),
    "simple":    lambda name: SimpleAgent(name),
}


# Number of independent inference batches a worker keeps in flight. Two lets
# the worker encode one group of games while the other group's batch is on
# the GPU; one reproduces the old strictly-alternating behaviour.
_PIPELINE_GROUPS = 2

# Env var selecting one worker id to run under cProfile, e.g.
# ``SDB_PROFILE_WORKER=0``. Rollout workers are spawned children, so an
# in-process profiler is the only way to see where their CPU actually goes;
# external timers can only bound a phase, not attribute it. Off unless set.
_PROFILE_WORKER_ENV = "SDB_PROFILE_WORKER"


def _maybe_start_profile(worker_id: int):
    """Return a running cProfile.Profile if this worker is the selected one."""
    import os
    want = os.environ.get(_PROFILE_WORKER_ENV)
    if want is None or str(worker_id) != want.strip():
        return None
    import cProfile
    pr = cProfile.Profile()
    pr.enable()
    return pr


def _maybe_finish_profile(profiler, worker_id: int) -> None:
    """Write per-function stats for the profiled worker to ``logs/``."""
    if profiler is None:
        return
    try:
        import pstats
        import pathlib
        profiler.disable()
        out_dir = pathlib.Path("logs")
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = out_dir / f"worker_profile_w{worker_id}"
        profiler.dump_stats(str(stem.with_suffix(".prof")))
        with open(stem.with_suffix(".txt"), "w", encoding="utf-8") as fh:
            st = pstats.Stats(profiler, stream=fh)
            st.sort_stats("tottime").print_stats(40)
    except Exception:
        pass


class _DummyAgent(Agent):
    """Placeholder agent for the PPO player slot."""
    def make_decision(self, game_state):
        raise RuntimeError("DummyAgent.make_decision should never be called")


class _TourneyPPOSlot(_DummyAgent):
    """Placeholder PPO agent for a tournament pairing that carries a model_id.

    The tournament worker's batched loop identifies PPO slots by isinstance
    check against this class and routes their decisions through the
    InferenceServer using the attached ``model_id``. ``make_decision`` is
    never called; the batched loop bypasses it.
    """

    def __init__(self, name: str, model_id: str):
        super().__init__(name)
        self.model_id = model_id


class _ServerPPOAgent(Agent):
    """PPO opponent that routes inference through the InferenceServer on GPU.

    Instead of running a local model on CPU, this agent encodes the game
    state and sends it to the central InferenceServer for GPU inference.
    Used for self-play opponents in multiprocess training.
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        card_names: list[str],
        card_index_map: dict[str, int],
        action_dim: int,
        worker_id: int,
        request_queue,
        response_queue,
    ):
        super().__init__(name)
        self.model_id = model_id
        self.card_names = card_names
        self.card_index_map = card_index_map
        self.action_dim = action_dim
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self._request_counter = 0
        self._state_buf = np.zeros(get_state_size(card_names), dtype=np.float32)
        self._mask_buf = np.zeros(action_dim, dtype=bool)

    def make_decision(self, game_state):
        """Encode state, send to GPU server, return chosen action."""
        player = game_state.current_player

        ctx = build_action_context(
            game_state, player, self.card_index_map,
            self.action_dim, mask_buf=self._mask_buf,
        )
        encode_state(
            game_state, is_current_player_training=True,
            cards=self.card_names,
            card_index_map=self.card_index_map,
            state_buf=self._state_buf,
            can_buy=ctx.can_buy,
            has_actions=ctx.has_meaningful,
        )

        # Build single-element batch
        states_np = self._state_buf.copy().reshape(1, -1)
        masks_np = ctx.mask.astype(np.uint8).reshape(1, -1)
        if ctx.has_meaningful:
            masks_np[0, END_TURN_INDEX] = 0

        req_id = self._request_counter
        self._request_counter += 1
        self.request_queue.put(InferenceRequest(
            worker_id=self.worker_id,
            request_id=req_id,
            states=states_np,
            masks=masks_np,
            model_id=self.model_id,
        ))

        response = self.response_queue.get()
        if response.error is not None:
            raise RuntimeError(
                f"InferenceServer rejected request for model_id={self.model_id!r}: "
                f"{response.error}"
            )
        act_idx = int(response.action_indices[0])
        return ctx.resolvers[act_idx]


def _parse_opponent_spec(spec: str) -> list[tuple[str, float]]:
    """Parse opponent spec into (name, weight) pairs."""
    entries = []
    has_weights = ":" in spec
    for part in spec.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if ":" in part:
            name, w = part.split(":", 1)
            entries.append((name.strip(), float(w.strip())))
        else:
            entries.append((part, 0.0))
    if not has_weights:
        eq = 1.0 / len(entries) if entries else 1.0
        entries = [(n, eq) for n, _ in entries]
    total = sum(w for _, w in entries)
    return [(n, w / total) for n, w in entries] if total > 0 else entries


def _build_opponent_factory(
    opponent_spec: str,
    snapshot_names: list[str] | None = None,
    self_play_ratio: float = 0.5,
    pfsp_weights: list[float] | None = None,
    card_names: list[str] | None = None,
    card_index_map: dict[str, int] | None = None,
    action_dim: int = 0,
    worker_id: int = 0,
    request_queue=None,
    response_queue=None,
    registry=None,
):
    """Build an opponent factory from a spec string and optional snapshot config.

    When snapshot_names are provided, self_play_ratio fraction of calls return
    a _ServerPPOAgent that routes inference through the InferenceServer on GPU.
    Only snapshot names are needed — the actual state_dicts live on the server.
    """
    entries = _parse_opponent_spec(opponent_spec)
    names = [n for n, _ in entries]
    weights = [w for _, w in entries]

    snaps = snapshot_names or []
    pfsp_w = pfsp_weights or [1.0] * len(snaps)

    def factory() -> Agent:
        # Decide snapshot vs fixed opponent
        if snaps and random.random() < self_play_ratio:
            idx = random.choices(range(len(snaps)), weights=pfsp_w, k=1)[0]
            snap_name = snaps[idx]
            return _ServerPPOAgent(
                name=snap_name,
                model_id=snap_name,
                card_names=card_names,
                card_index_map=card_index_map,
                action_dim=action_dim,
                worker_id=worker_id,
                request_queue=request_queue,
                response_queue=response_queue,
            )
        chosen = random.choices(names, weights=weights, k=1)[0]
        return _AGENT_FACTORIES[chosen](chosen.capitalize())

    return factory


def _advance_non_ppo(game: Game, training_agent_name: str):
    """Advance the game past non-neural-net decisions.

    Stops when the current player is either the training agent or a
    _ServerPPOAgent opponent (both need GPU inference via the server).
    """
    while not game.is_game_over:
        player = game.current_player
        if player.name == training_agent_name:
            break
        if isinstance(player.agent, _ServerPPOAgent):
            break
        action = player.make_decision(game)
        game.apply_decision(action)


def _start_game(
    cards: list[Card],
    card_index_map: dict[str, int],
    opponent_factory,
    training_agent_name: str = "PPO",
):
    """Initialize a new game. Returns (game, rollout_buffer, opponent_name)."""
    opponent = opponent_factory()
    game = Game(cards, card_index_map=card_index_map)
    game.add_player(training_agent_name, _DummyAgent(training_agent_name))
    game.add_player(opponent.name, opponent)
    game.start_game()
    buf = RolloutBuffer()
    return game, buf, opponent.name


def _finish_game(
    i: int,
    games: list,
    buffers: list,
    game_opponents: list,
    completed_rollouts: list,
    opponent_results: dict,
    ppo_config: PPOConfig,
    training_agent_name: str = "PPO",
):
    """Handle a completed game: record result, compute reward, finalize rollout."""
    winner = games[i].get_winner()
    won = winner == training_agent_name

    opp_name = game_opponents[i]
    if opp_name is not None:
        entry = opponent_results.get(opp_name)
        if entry is None:
            entry = [0, 0]
            opponent_results[opp_name] = entry
        if won:
            entry[0] += 1
        entry[1] += 1

    if buffers[i] is None or len(buffers[i]) == 0:
        games[i] = None
        buffers[i] = None
        game_opponents[i] = None
        return

    reward = 1.0 if won else -1.0
    buffers[i].fill_last_reward(reward)
    # Compute GAE on CPU — no GPU needed
    rollout = buffers[i].finish(
        gamma=ppo_config.gamma,
        lam=ppo_config.lam,
        device=torch.device("cpu"),
        normalize=False,
    )
    completed_rollouts.append(rollout)
    games[i] = None
    buffers[i] = None
    game_opponents[i] = None


def sim_worker_main(
    worker_id: int,
    num_episodes: int,
    num_concurrent: int,
    data_config_dict: dict,
    card_names: list[str],
    card_index_map: dict[str, int],
    action_dim: int,
    state_size: int,
    ppo_config_dict: dict,
    opponent_spec: str,
    seed: int,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    result_queue: mp.Queue,
    snapshot_names: list[str] | None = None,
    self_play_ratio: float = 0.5,
    pfsp_weights: list[float] | None = None,
    shared_io_spec=None,
):
    """Entry point for a simulation worker process.

    Runs game simulation and state encoding on CPU. Sends encoded states
    to the InferenceServer for GPU inference. Merges rollout data locally
    and returns a single pre-merged result via result_queue.

    card_names and card_index_map are passed from the parent process to
    guarantee encoding consistency across all workers.

    For self-play, snapshot_names lists the opponent models available on
    the InferenceServer. Workers create _ServerPPOAgent opponents that
    route decisions through the server for GPU inference.

    ``shared_io_spec`` describes the shared-memory block used to exchange
    batches with the server. When omitted the worker falls back to sending
    arrays inline on the queue.
    """
    shared_io = None
    profiler = _maybe_start_profile(worker_id)
    try:
        if shared_io_spec is not None:
            from src.ppo.shared_io import SharedInferenceBuffers
            shared_io = SharedInferenceBuffers.attach(shared_io_spec)
        _sim_worker_inner(
            worker_id, num_episodes, num_concurrent, data_config_dict,
            card_names, card_index_map, action_dim, state_size,
            ppo_config_dict, opponent_spec, seed,
            request_queue, response_queue, result_queue,
            snapshot_names=snapshot_names,
            self_play_ratio=self_play_ratio,
            pfsp_weights=pfsp_weights,
            shared_io=shared_io,
        )
    except Exception:
        result_queue.put(WorkerError(
            worker_id=worker_id,
            error=traceback.format_exc(),
        ))
    finally:
        _maybe_finish_profile(profiler, worker_id)
        if shared_io is not None:
            shared_io.close()


def _sim_worker_inner(
    worker_id, num_episodes, num_concurrent, data_config_dict,
    card_names, card_index_map, action_dim, state_size,
    ppo_config_dict, opponent_spec, seed,
    request_queue, response_queue, result_queue,
    snapshot_names=None, self_play_ratio=0.5, pfsp_weights=None,
    shared_io=None,
):
    """Core worker logic, separated for clean error wrapping."""
    # Worker initialization
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    torch.set_num_threads(1)  # avoid inter-worker contention
    set_disabled(True)

    ppo_cfg = PPOConfig.from_dict(ppo_config_dict)

    # Load cards independently (avoids pickling Card/Effect objects)
    data_cfg = DataConfig.from_dict(data_config_dict)
    cards = data_cfg.load_cards()
    registry = data_cfg.build_registry(cards)

    # Verify encoding alignment with parent process
    assert registry.card_names == card_names, (
        f"Worker {worker_id} card_names mismatch: "
        f"expected {len(card_names)} cards, got {len(registry.card_names)}"
    )

    training_agent_name = "PPO"
    opponent_factory = _build_opponent_factory(
        opponent_spec,
        snapshot_names=snapshot_names,
        self_play_ratio=self_play_ratio,
        pfsp_weights=pfsp_weights,
        card_names=card_names,
        card_index_map=card_index_map,
        action_dim=action_dim,
        worker_id=worker_id,
        request_queue=request_queue,
        response_queue=response_queue,
        registry=registry,
    )

    # Pre-allocate reusable buffers for encoding. The state buffer is a
    # (num_concurrent, state_size) matrix so each game slot owns a
    # contiguous row view — encode_state fills the row in place and the
    # rollout buffer captures a per-row copy. The mask buffer is the
    # same shape in bool; fancy-indexing it selects the active rows for
    # a batched inference request (view(uint8) is zero-copy since bool
    # and uint8 share byte width).
    states_rows_buf = np.zeros((num_concurrent, state_size), dtype=np.float32)
    masks_rows_buf = np.zeros((num_concurrent, action_dim), dtype=bool)

    # Game slots
    games: list[Optional[Game]] = [None] * num_concurrent
    buffers: list[Optional[RolloutBuffer]] = [None] * num_concurrent
    game_opponents: list[Optional[str]] = [None] * num_concurrent

    completed_rollouts: list[tuple] = []
    opponent_results: dict[str, list[int]] = {}
    episodes_started = 0
    episodes_completed = 0
    request_id = 0

    # Per-phase wall time. ``wait`` is the diagnostic that matters: it is
    # time the worker spends blocked on the inference server doing no
    # simulation, which is pure lost throughput in a synchronous loop.
    T = {"advance": 0.0, "finish": 0.0, "encode": 0.0,
         "send": 0.0, "wait": 0.0, "apply": 0.0,
         # Batch-shape diagnostics: how many game slots actually had a
         # pending decision each iteration. A large gap between this and
         # num_concurrent means the slots are draining out (episode tail),
         # which shrinks GPU batches and multiplies round trips.
         "iters": 0.0, "pending_sum": 0.0,
         # Sub-phases of encode, to separate real work from loop overhead.
         "ctx": 0.0, "enc": 0.0}
    _loop_start = _time.perf_counter()

    # Split game slots into pipeline groups. Each group submits its own
    # inference request, so while one group's batch is on the GPU the worker
    # is advancing and encoding the other. Without this the worker's CPU time
    # and the server's GPU time add up instead of overlapping, which
    # profiling measured as ~40% of worker wall spent blocked.
    n_groups = (
        _PIPELINE_GROUPS
        if shared_io is not None and num_concurrent >= 2 * _PIPELINE_GROUPS
        else 1
    )
    # Contiguous blocks, not a stride. Groups index into per-slot row buffers
    # (``states_rows_buf`` is [slots, state_size]), so an interleaved
    # partition would touch every other row and destroy cache locality on the
    # hottest loop in the worker.
    _bounds = [
        (num_concurrent * g // n_groups, num_concurrent * (g + 1) // n_groups)
        for g in range(n_groups)
    ]
    groups = [list(range(lo, hi)) for lo, hi in _bounds]

    # Responses for a group that arrive while another group is being
    # collected, keyed by request_id.
    response_stash: dict[int, "InferenceResponse"] = {}

    def _retire_and_refill(slot_ids):
        """Retire finished games in these slots and start replacements."""
        nonlocal episodes_started, episodes_completed
        for i in slot_ids:
            if games[i] is None or not games[i].is_game_over:
                continue
            _finish_game(
                i, games, buffers, game_opponents,
                completed_rollouts, opponent_results, ppo_cfg,
                training_agent_name,
            )
            episodes_completed += 1
            if episodes_started < num_episodes:
                games[i], buffers[i], game_opponents[i] = _start_game(
                    cards, card_index_map, opponent_factory, training_agent_name,
                )
                episodes_started += 1
                _advance_non_ppo(games[i], training_agent_name)
                if games[i].is_game_over:
                    _finish_game(
                        i, games, buffers, game_opponents,
                        completed_rollouts, opponent_results, ppo_cfg,
                        training_agent_name,
                    )
                    episodes_completed += 1

    def _submit(slot_ids, shm_slot):
        """Advance, retire, encode and submit one group.

        Returns the metadata needed to apply the responses later, or None
        when the group has no games needing a decision.
        """
        nonlocal request_id

        _t = _time.perf_counter()
        for i in slot_ids:
            if games[i] is not None and not games[i].is_game_over:
                _advance_non_ppo(games[i], training_agent_name)
        T["advance"] += _time.perf_counter() - _t

        _t = _time.perf_counter()
        _retire_and_refill(slot_ids)
        T["finish"] += _time.perf_counter() - _t

        # Each game slot owns a row in ``states_rows_buf`` / ``masks_rows_buf``
        # so encode_state and build_action_context fill them in place with no
        # intermediate copies.
        pending_indices: list[int] = []
        pending_resolvers: list[dict] = []
        pending_is_training: list[bool] = []
        pending_model_ids: list[str] = []

        _t = _time.perf_counter()
        for i in slot_ids:
            if games[i] is None or games[i].is_game_over:
                continue
            player = games[i].current_player
            is_training = (player.name == training_agent_name)
            is_server_opp = isinstance(player.agent, _ServerPPOAgent)
            if not is_training and not is_server_opp:
                continue

            _t1 = _time.perf_counter()
            ctx = build_action_context(
                games[i], player, card_index_map,
                action_dim, mask_buf=masks_rows_buf[i],
            )
            _t2 = _time.perf_counter()
            T["ctx"] += _t2 - _t1
            encode_state(
                games[i], is_current_player_training=is_training,
                cards=card_names,
                card_index_map=card_index_map,
                state_buf=states_rows_buf[i],
                can_buy=ctx.can_buy,
                has_actions=ctx.has_meaningful,
                return_numpy=True,
            )
            T["enc"] += _time.perf_counter() - _t2
            # Suppress END_TURN inline so we don't touch the mask again later.
            if ctx.has_meaningful:
                masks_rows_buf[i, END_TURN_INDEX] = False

            pending_indices.append(i)
            pending_resolvers.append(ctx.resolvers)
            pending_is_training.append(is_training)
            pending_model_ids.append(
                "training" if is_training else player.agent.model_id
            )
        T["encode"] += _time.perf_counter() - _t
        T["iters"] += 1
        T["pending_sum"] += len(pending_indices)

        if not pending_indices:
            return None

        _t = _time.perf_counter()
        model_groups: dict[str, list[int]] = {}
        for j, mid in enumerate(pending_model_ids):
            model_groups.setdefault(mid, []).append(j)

        request_ids_by_mid: dict[str, tuple[int, list[int]]] = {}
        if shared_io is not None:
            shm_states = shared_io.states[worker_id, shm_slot]
            shm_masks = shared_io.masks[worker_id, shm_slot]
            row_off = 0
            for mid, indices in model_groups.items():
                rows = [pending_indices[j] for j in indices]
                n = len(rows)
                shm_states[row_off:row_off + n] = states_rows_buf[rows]
                shm_masks[row_off:row_off + n] = masks_rows_buf[rows].view(np.uint8)
                request_queue.put(InferenceRequest(
                    worker_id=worker_id,
                    request_id=request_id,
                    model_id=mid,
                    slot=shm_slot,
                    row_offset=row_off,
                    n_rows=n,
                ))
                request_ids_by_mid[mid] = (request_id, indices)
                request_id += 1
                row_off += n
        else:
            states_np = states_rows_buf[pending_indices]
            masks_np = masks_rows_buf[pending_indices].view(np.uint8)
            for mid, indices in model_groups.items():
                request_queue.put(InferenceRequest(
                    worker_id=worker_id,
                    request_id=request_id,
                    model_id=mid,
                    states=states_np[indices],
                    masks=masks_np[indices],
                ))
                request_ids_by_mid[mid] = (request_id, indices)
                request_id += 1
        T["send"] += _time.perf_counter() - _t

        return (pending_indices, pending_resolvers, pending_is_training,
                request_ids_by_mid)

    def _collect(meta):
        """Wait for one group's responses and apply the sampled actions.

        With several groups in flight the response queue interleaves them, so
        responses belonging to another group are stashed rather than dropped.
        """
        pending_indices, pending_resolvers, pending_is_training, by_mid = meta
        wanted = {req_id: indices for _, (req_id, indices) in by_mid.items()}

        all_responses: dict[int, tuple] = {}

        def _record(resp, indices):
            if resp.action_indices is not None:
                acts, lps, vals = (
                    resp.action_indices, resp.log_probs, resp.values,
                )
            else:
                lo = resp.row_offset
                hi = lo + resp.n_rows
                w, s = worker_id, resp.slot
                acts = shared_io.actions[w, s, lo:hi]
                lps = shared_io.log_probs[w, s, lo:hi]
                vals = shared_io.values[w, s, lo:hi]
            for k, j in enumerate(indices):
                all_responses[j] = (int(acts[k]), float(lps[k]), float(vals[k]))

        _t = _time.perf_counter()
        # Anything stashed by a previous _collect for this group.
        for req_id in list(wanted):
            resp = response_stash.pop(req_id, None)
            if resp is not None:
                _record(resp, wanted.pop(req_id))

        while wanted:
            response: InferenceResponse = response_queue.get()
            if response.error is not None:
                raise RuntimeError(
                    f"InferenceServer error (worker={worker_id}): {response.error}"
                )
            indices = wanted.pop(response.request_id, None)
            if indices is None:
                # Belongs to another in-flight group; keep it for that caller.
                response_stash[response.request_id] = response
                continue
            _record(response, indices)
        T["wait"] += _time.perf_counter() - _t

        # Record in the rollout buffer only for the training agent. The buffer
        # accepts numpy + Python primitives directly, so we copy the per-slot
        # row once and skip all torch wrapping.
        _t = _time.perf_counter()
        for j, i in enumerate(pending_indices):
            act_idx, log_prob, value = all_responses[j]
            action = pending_resolvers[j][act_idx]

            if pending_is_training[j]:
                buffers[i].add(
                    states_rows_buf[i].copy(),
                    act_idx,
                    log_prob,
                    value,
                    reward=0.0,
                    done=False,
                    mask=masks_rows_buf[i].copy(),
                )

            games[i].apply_decision(action)
        T["apply"] += _time.perf_counter() - _t

    # Fill initial game slots
    active_count = min(num_concurrent, num_episodes)
    for i in range(active_count):
        games[i], buffers[i], game_opponents[i] = _start_game(
            cards, card_index_map, opponent_factory, training_agent_name,
        )
        episodes_started += 1

    # Prime every group so there is always a request in flight, then service
    # them round-robin: collecting group g happens while the other groups'
    # batches are already queued on the server.
    inflight: list[object] = [None] * n_groups
    for g in range(n_groups):
        if episodes_completed >= num_episodes:
            break
        inflight[g] = _submit(groups[g], g)

    while episodes_completed < num_episodes and any(
        m is not None for m in inflight
    ):
        for g in range(n_groups):
            if inflight[g] is None:
                continue
            _collect(inflight[g])
            inflight[g] = None
            if episodes_completed < num_episodes:
                inflight[g] = _submit(groups[g], g)

    T["total"] = _time.perf_counter() - _loop_start

    # Merge rollouts locally before sending (reduces IPC overhead)
    if completed_rollouts:
        S, A, OL, R, Adv, M = zip(*completed_rollouts)
        has_masks = all(m is not None for m in M)
        result_queue.put(WorkerResult(
            worker_id=worker_id,
            states=torch.cat(S),
            actions=torch.cat(A),
            log_probs=torch.cat(OL),
            returns=torch.cat(R),
            advantages=torch.cat(Adv),
            masks=torch.cat(M) if has_masks else None,
            opponent_results=opponent_results,
            episodes_completed=episodes_completed,
            timings=T,
        ))
    else:
        # Edge case: all games ended without PPO decisions
        result_queue.put(WorkerResult(
            worker_id=worker_id,
            states=torch.empty(0),
            actions=torch.empty(0, dtype=torch.int64),
            log_probs=torch.empty(0),
            returns=torch.empty(0),
            advantages=torch.empty(0),
            masks=None,
            opponent_results=opponent_results,
            episodes_completed=episodes_completed,
            timings=T,
        ))


def sim_worker_eval(
    worker_id: int,
    num_games: int,
    num_concurrent: int,
    data_config_dict: dict,
    card_names: list[str],
    card_index_map: dict[str, int],
    action_dim: int,
    state_size: int,
    opponent_spec: str,
    seed: int,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    result_queue: mp.Queue,
    *,
    snapshot_names: list[str] | None = None,
    self_play_ratio: float = 0.5,
    pfsp_weights: list[float] | None = None,
    shared_io_spec=None,
):
    """Entry point for an evaluation worker process.

    Same game loop as sim_worker_main but without rollout collection.
    Returns win/loss/step counts. Supports PPO snapshot opponents when
    snapshot_names are provided (routed through InferenceServer).
    """
    shared_io = None
    try:
        if shared_io_spec is not None:
            from src.ppo.shared_io import SharedInferenceBuffers
            shared_io = SharedInferenceBuffers.attach(shared_io_spec)
        _sim_worker_eval_inner(
            worker_id, num_games, num_concurrent, data_config_dict,
            card_names, card_index_map, action_dim, state_size,
            opponent_spec, seed, request_queue, response_queue, result_queue,
            snapshot_names=snapshot_names,
            self_play_ratio=self_play_ratio,
            pfsp_weights=pfsp_weights,
            shared_io=shared_io,
        )
    except Exception:
        result_queue.put(WorkerError(
            worker_id=worker_id,
            error=traceback.format_exc(),
        ))
    finally:
        if shared_io is not None:
            shared_io.close()


def _sim_worker_eval_inner(
    worker_id, num_games, num_concurrent, data_config_dict,
    card_names, card_index_map, action_dim, state_size,
    opponent_spec, seed, request_queue, response_queue, result_queue,
    snapshot_names=None, self_play_ratio=0.5, pfsp_weights=None,
    shared_io=None,
):
    """Core eval worker logic.

    Handles both builtin and PPO snapshot opponents. When snapshot_names
    are provided, creates _ServerPPOAgent opponents that route inference
    through the InferenceServer, and collects decisions for both training
    agent and opponent in a single batched request grouped by model_id.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    torch.set_num_threads(1)
    set_disabled(True)

    data_cfg = DataConfig.from_dict(data_config_dict)
    cards = data_cfg.load_cards()
    registry = data_cfg.build_registry(cards)

    assert registry.card_names == card_names, (
        f"Eval worker {worker_id} card_names mismatch"
    )

    training_agent_name = "PPO"
    opponent_factory = _build_opponent_factory(
        opponent_spec,
        snapshot_names=snapshot_names,
        self_play_ratio=self_play_ratio,
        pfsp_weights=pfsp_weights,
        card_names=card_names,
        card_index_map=card_index_map,
        action_dim=action_dim,
        worker_id=worker_id,
        request_queue=request_queue,
        response_queue=response_queue,
        registry=registry,
    )

    states_rows_buf = np.zeros((num_concurrent, state_size), dtype=np.float32)
    masks_rows_buf = np.zeros((num_concurrent, action_dim), dtype=bool)

    games: list[Optional[Game]] = [None] * num_concurrent
    step_counts: list[int] = [0] * num_concurrent

    wins = 0
    losses = 0
    total_steps = 0
    games_started = 0
    games_completed = 0
    request_id = 0

    active_count = min(num_concurrent, num_games)
    for i in range(active_count):
        opponent = opponent_factory()
        game = Game(cards, card_index_map=card_index_map)
        game.add_player(training_agent_name, _DummyAgent(training_agent_name))
        game.add_player(opponent.name, opponent)
        game.start_game()
        games[i] = game
        step_counts[i] = 0
        games_started += 1

    while games_completed < num_games:
        for i in range(num_concurrent):
            if games[i] is not None and not games[i].is_game_over:
                _advance_non_ppo(games[i], training_agent_name)

        for i in range(num_concurrent):
            if games[i] is None or not games[i].is_game_over:
                continue
            winner = games[i].get_winner()
            if winner == training_agent_name:
                wins += 1
            else:
                losses += 1
            total_steps += step_counts[i]
            games_completed += 1
            if games_started < num_games:
                opponent = opponent_factory()
                game = Game(cards, card_index_map=card_index_map)
                game.add_player(training_agent_name, _DummyAgent(training_agent_name))
                game.add_player(opponent.name, opponent)
                game.start_game()
                games[i] = game
                step_counts[i] = 0
                games_started += 1
                _advance_non_ppo(games[i], training_agent_name)
                if games[i].is_game_over:
                    w = games[i].get_winner()
                    if w == training_agent_name:
                        wins += 1
                    else:
                        losses += 1
                    total_steps += step_counts[i]
                    games_completed += 1
                    games[i] = None
            else:
                games[i] = None

        # Collect ALL pending neural-net decisions — both training agent
        # and _ServerPPOAgent opponents. Group by model_id for batched
        # inference through the server. Each game slot owns a row in the
        # pre-allocated state/mask buffers — encode_state and
        # build_action_context fill them in place, and fancy-indexing the
        # active rows produces the contiguous batch tensors with no
        # extra per-row copies.
        pending_indices: list[int] = []
        pending_resolvers: list[dict] = []
        pending_model_ids: list[str] = []

        for i in range(num_concurrent):
            if games[i] is None or games[i].is_game_over:
                continue
            player = games[i].current_player
            is_training = (player.name == training_agent_name)
            is_server_opp = isinstance(player.agent, _ServerPPOAgent)
            if not is_training and not is_server_opp:
                continue

            ctx = build_action_context(
                games[i], player, card_index_map,
                action_dim, mask_buf=masks_rows_buf[i],
            )
            encode_state(
                games[i], is_current_player_training=is_training,
                cards=card_names,
                card_index_map=card_index_map,
                state_buf=states_rows_buf[i],
                can_buy=ctx.can_buy,
                has_actions=ctx.has_meaningful,
                return_numpy=True,
            )
            if ctx.has_meaningful:
                masks_rows_buf[i, END_TURN_INDEX] = False

            pending_indices.append(i)
            pending_resolvers.append(ctx.resolvers)
            pending_model_ids.append(
                "training" if is_training else player.agent.model_id
            )

        if not pending_indices:
            continue

        # Publish the batch and request inference. Mirrors the training
        # worker: with a shared block the payload never touches the queue.
        model_groups: dict[str, list[int]] = {}
        for j, mid in enumerate(pending_model_ids):
            model_groups.setdefault(mid, []).append(j)

        all_responses: dict[int, int] = {}
        request_ids_by_mid: dict[str, tuple[int, list[int]]] = {}

        if shared_io is not None:
            slot = 0
            shm_states = shared_io.states[worker_id, slot]
            shm_masks = shared_io.masks[worker_id, slot]
            row_off = 0
            for mid, indices in model_groups.items():
                rows = [pending_indices[j] for j in indices]
                n = len(rows)
                shm_states[row_off:row_off + n] = states_rows_buf[rows]
                shm_masks[row_off:row_off + n] = masks_rows_buf[rows].view(np.uint8)
                request_queue.put(InferenceRequest(
                    worker_id=worker_id,
                    request_id=request_id,
                    model_id=mid,
                    slot=slot,
                    row_offset=row_off,
                    n_rows=n,
                ))
                request_ids_by_mid[mid] = (request_id, indices)
                request_id += 1
                row_off += n
        else:
            states_np = states_rows_buf[pending_indices]
            masks_np = masks_rows_buf[pending_indices].view(np.uint8)
            for mid, indices in model_groups.items():
                request_queue.put(InferenceRequest(
                    worker_id=worker_id,
                    request_id=request_id,
                    model_id=mid,
                    states=states_np[indices],
                    masks=masks_np[indices],
                ))
                request_ids_by_mid[mid] = (request_id, indices)
                request_id += 1

        for _ in range(len(model_groups)):
            response: InferenceResponse = response_queue.get()
            if response.error is not None:
                raise RuntimeError(
                    f"InferenceServer error (worker={worker_id}): {response.error}"
                )
            for mid, (req_id, indices) in request_ids_by_mid.items():
                if response.request_id != req_id:
                    continue
                if response.action_indices is not None:
                    acts = response.action_indices
                else:
                    lo = response.row_offset
                    hi = lo + response.n_rows
                    acts = shared_io.actions[worker_id, response.slot, lo:hi]
                for k, j in enumerate(indices):
                    all_responses[j] = int(acts[k])
                break

        for k, i in enumerate(pending_indices):
            act_idx = all_responses[k]
            action = pending_resolvers[k][act_idx]
            games[i].apply_decision(action)
            step_counts[i] += 1

    result_queue.put(EvalWorkerResult(
        worker_id=worker_id,
        wins=wins,
        losses=losses,
        total_steps=total_steps,
    ))


# ======================================================================
# Parallel tournament worker
# ======================================================================

from dataclasses import dataclass as _dataclass


@_dataclass
class PairingTask:
    """Coordinator -> tournament worker: one pairing to play."""
    pairing_id: int
    i: int                          # participant index of side A
    j: int                          # participant index of side B
    num_games: int
    seed: int
    side_a_kind: str                # "ppo" | "builtin"
    side_b_kind: str
    side_a_id: str                  # model_id (if ppo) or agent_type (if builtin)
    side_b_id: str
    # When set, the worker records per-decision data for side A and writes a
    # gzipped JSONL replay file to ``replay_output_path``. Side A is always a
    # CheckpointParticipant in elo_tournament's ppo_pairings (checkpoints are
    # built before builtins), so focusing on side A is sufficient to capture
    # PPO behavior. ``replay_output_path`` is required when this is True.
    collect_replays: bool = False
    replay_output_path: str | None = None


@_dataclass
class TournamentResult:
    """Tournament worker -> coordinator: result of one pairing."""
    pairing_id: int
    i: int
    j: int
    wins_a: int
    wins_b: int
    # Set to the path that the worker wrote when ``collect_replays`` was True.
    # ``None`` means no replay file was produced (collection wasn't requested,
    # or every game ended before side A acted — extremely rare).
    replay_path: str | None = None


def _build_slot(kind: str, ident: str, name: str) -> Agent:
    """Build a game-engine agent for one side of a tournament pairing."""
    if kind == "ppo":
        return _TourneyPPOSlot(name, model_id=ident)
    if kind == "builtin":
        factory = _AGENT_FACTORIES.get(ident)
        if factory is None:
            raise ValueError(f"Unknown builtin agent type: {ident!r}")
        return factory(name)
    raise ValueError(f"Unknown side kind: {kind!r}")


def _advance_tourney_non_ppo(game: Game) -> None:
    """Advance a tournament game past all non-PPO decisions.

    Stops when the current player is a ``_TourneyPPOSlot``, which needs
    inference from the server. Built-in agents run inline.
    """
    while not game.is_game_over:
        player = game.current_player
        if isinstance(player.agent, _TourneyPPOSlot):
            break
        action = player.make_decision(game)
        game.apply_decision(action)


def _play_pairing(
    task: "PairingTask",
    worker_id: int,
    num_concurrent: int,
    card_names: list,
    card_index_map: dict,
    cards: list,
    action_dim: int,
    state_size: int,
    request_queue,
    response_queue,
) -> tuple[int, int, str | None]:
    """Play ``task.num_games`` games for one pairing and return
    ``(wins_a, wins_b, replay_path)``. Uses the server's batched inference
    path for any PPO slots.

    When ``task.collect_replays`` is True, records per-decision data for
    side A only (which is always a PPO checkpoint in elo_tournament's
    ppo_pairings — checkpoints are built before builtins). The completed
    replay is written to ``task.replay_output_path`` via a temp file +
    atomic rename, and that path is returned. If collection wasn't
    requested, returns ``None`` for the path.
    """
    random.seed(task.seed)
    np.random.seed(task.seed % (2**32))

    name_a = f"A_{task.side_a_id}"
    name_b = f"B_{task.side_b_id}"
    # Names must differ even when both sides share a model_id (self-mirror
    # pairing) so that Game can track players distinctly.
    if name_a == name_b:
        name_b = name_b + "_2"

    states_rows_buf = np.zeros((num_concurrent, state_size), dtype=np.float32)
    masks_rows_buf = np.zeros((num_concurrent, action_dim), dtype=bool)

    games: list[Optional[Game]] = [None] * num_concurrent
    wins_a = 0
    wins_b = 0
    games_started = 0
    games_completed = 0
    request_id = 0

    # Replay collection — built lazily so the non-replay path pays nothing.
    collector = None
    replay_path: str | None = None
    if task.collect_replays:
        if not task.replay_output_path:
            raise ValueError(
                f"PairingTask {task.pairing_id} sets collect_replays=True "
                "but replay_output_path is empty"
            )
        from src.analysis.replay_collector import ReplayCollector
        collector = ReplayCollector(card_names, action_dim)

    def _new_game(slot: int) -> Game:
        g = Game(cards, card_index_map=card_index_map)
        g.add_player(name_a, _build_slot(task.side_a_kind, task.side_a_id, name_a))
        g.add_player(name_b, _build_slot(task.side_b_kind, task.side_b_id, name_b))
        g.start_game()
        if collector is not None:
            collector.start_game(slot)
        return g

    def _record_result(slot: int, g: Game) -> None:
        nonlocal wins_a, wins_b, games_completed
        winner = g.get_winner()
        if winner == name_a:
            wins_a += 1
        else:
            wins_b += 1
        games_completed += 1
        if collector is not None:
            # Tag with model identity so the dashboard can group / compare
            # decisions across pairings without consulting external metadata.
            collector.finish_game(
                slot=slot,
                winner=winner,
                total_turns=g.stats.total_turns,
                opponent_type=task.side_b_id,
                player_model=task.side_a_id,
                opponent_model=task.side_b_id,
            )

    active = min(num_concurrent, task.num_games)
    for i in range(active):
        games[i] = _new_game(i)
        games_started += 1
        _advance_tourney_non_ppo(games[i])
        if games[i].is_game_over:
            _record_result(i, games[i])
            games[i] = None

    while games_completed < task.num_games:
        # Retire finished games and start replacements up to num_games.
        for i in range(num_concurrent):
            if games[i] is None or not games[i].is_game_over:
                continue
            _record_result(i, games[i])
            if games_started < task.num_games:
                games[i] = _new_game(i)
                games_started += 1
                _advance_tourney_non_ppo(games[i])
                if games[i].is_game_over:
                    _record_result(i, games[i])
                    games[i] = None
            else:
                games[i] = None

        # Collect pending PPO decisions across all active slots, group
        # them by model_id, dispatch one batched request per model_id.
        pending_indices: list[int] = []
        pending_resolvers: list[dict] = []
        pending_model_ids: list[str] = []
        # Whether each pending slot belongs to side A. Used to gate replay
        # recording so that PPO-vs-PPO mirror matches don't mix both sides
        # into one focal-player file.
        pending_is_side_a: list[bool] = []

        for i in range(num_concurrent):
            if games[i] is None or games[i].is_game_over:
                continue
            player = games[i].current_player
            if not isinstance(player.agent, _TourneyPPOSlot):
                continue
            ctx = build_action_context(
                games[i], player, card_index_map,
                action_dim, mask_buf=masks_rows_buf[i],
            )
            encode_state(
                games[i], is_current_player_training=True,
                cards=card_names,
                card_index_map=card_index_map,
                state_buf=states_rows_buf[i],
                can_buy=ctx.can_buy,
                has_actions=ctx.has_meaningful,
                return_numpy=True,
            )
            if ctx.has_meaningful:
                masks_rows_buf[i, END_TURN_INDEX] = False

            pending_indices.append(i)
            pending_resolvers.append(ctx.resolvers)
            pending_model_ids.append(player.agent.model_id)
            pending_is_side_a.append(player.name == name_a)

        if not pending_indices:
            # No PPO slots waiting — advance any remaining games whose
            # current player is a builtin, then loop. The outer
            # games_completed check will exit once everything's done.
            progressed = False
            for i in range(num_concurrent):
                if games[i] is not None and not games[i].is_game_over:
                    _advance_tourney_non_ppo(games[i])
                    progressed = True
            if not progressed:
                break
            continue

        states_np = states_rows_buf[pending_indices]
        masks_np = masks_rows_buf[pending_indices].view(np.uint8)

        model_groups: dict[str, list[int]] = {}
        for k, mid in enumerate(pending_model_ids):
            model_groups.setdefault(mid, []).append(k)

        # Per request-id, remember which pending-k indices it covers and
        # whether any of those need logits returned for replay recording.
        request_ids_by_mid: dict[str, tuple[int, list[int], bool]] = {}
        for mid, indices in model_groups.items():
            # When collecting replays, only request logits if at least one
            # slot in this model-group is side A. Side-B slices in mirror
            # matches stay on the cheap path.
            need_logits = collector is not None and any(
                pending_is_side_a[k] for k in indices
            )
            request_queue.put(InferenceRequest(
                worker_id=worker_id,
                request_id=request_id,
                states=states_np[indices],
                masks=masks_np[indices],
                model_id=mid,
                return_logits=need_logits,
            ))
            request_ids_by_mid[mid] = (request_id, indices, need_logits)
            request_id += 1

        # Per pending-k: (action_idx, value, logits_row_or_None).
        all_responses: dict[int, tuple[int, float, np.ndarray | None]] = {}
        for _ in range(len(model_groups)):
            response: InferenceResponse = response_queue.get()
            if response.error is not None:
                raise RuntimeError(
                    f"InferenceServer error (worker={worker_id}): {response.error}"
                )
            for mid, (req_id, indices, need_logits) in request_ids_by_mid.items():
                if response.request_id == req_id:
                    for k_idx, k in enumerate(indices):
                        logits_row = (
                            response.logits[k_idx] if (need_logits and response.logits is not None) else None
                        )
                        all_responses[k] = (
                            int(response.action_indices[k_idx]),
                            float(response.values[k_idx]),
                            logits_row,
                        )
                    break

        for k, i in enumerate(pending_indices):
            action_idx, value, logits_row = all_responses[k]
            action = pending_resolvers[k][action_idx]

            if collector is not None and pending_is_side_a[k]:
                # record_decision computes top-k probs and entropy from a
                # torch.Tensor logits row; convert here so the collector's
                # tensor branch (rather than its zeroed numpy fallback) runs.
                logits_t = (
                    torch.from_numpy(logits_row) if logits_row is not None
                    else torch.zeros(action_dim, dtype=torch.float32)
                )
                collector.record_decision(
                    slot=i,
                    game=games[i],
                    player=games[i].current_player,
                    logits=logits_t,
                    value=value,
                    mask=masks_rows_buf[i],
                    action_idx=action_idx,
                    action_type=action.type.name,
                    action_card_id=action.card_id,
                )

            games[i].apply_decision(action)
            # Advance past any subsequent builtin turns inline so the
            # next iteration only has to consider PPO slots.
            _advance_tourney_non_ppo(games[i])

    if collector is not None and task.replay_output_path:
        # Atomic write so a worker crash mid-save can't leave a partial
        # gzip file behind that downstream tooling would try to parse.
        import os as _os
        tmp_path = task.replay_output_path + ".tmp"
        collector.save(tmp_path)
        _os.replace(tmp_path, task.replay_output_path)
        replay_path = task.replay_output_path

    return wins_a, wins_b, replay_path


def tournament_worker_main(
    worker_id: int,
    num_concurrent: int,
    data_config_dict: dict,
    card_names: list,
    card_index_map: dict,
    action_dim: int,
    state_size: int,
    task_queue,
    result_queue,
    request_queue,
    response_queue,
) -> None:
    """Tournament worker process entry point.

    Pulls :class:`PairingTask`s off ``task_queue`` and emits
    :class:`TournamentResult`s on ``result_queue``. Exits when a ``None``
    sentinel is received.

    Neural-net inference is served by a long-lived :class:`InferenceServer`
    in the main process; the worker only runs game logic + state encoding.
    On any exception, emits a :class:`WorkerError` and exits so the
    coordinator can fail fast instead of hanging on the result queue.
    """
    torch.set_num_threads(1)
    set_disabled(True)

    data_cfg = DataConfig.from_dict(data_config_dict)
    cards = data_cfg.load_cards()
    registry = data_cfg.build_registry(cards)
    assert registry.card_names == card_names, (
        f"Tournament worker {worker_id} card_names mismatch"
    )

    try:
        while True:
            task = task_queue.get()
            if task is None:
                return
            wins_a, wins_b, replay_path = _play_pairing(
                task, worker_id, num_concurrent,
                card_names, card_index_map, cards,
                action_dim, state_size,
                request_queue, response_queue,
            )
            result_queue.put(TournamentResult(
                pairing_id=task.pairing_id, i=task.i, j=task.j,
                wins_a=wins_a, wins_b=wins_b,
                replay_path=replay_path,
            ))
    except BaseException:
        try:
            result_queue.put(WorkerError(
                worker_id=worker_id, error=traceback.format_exc(),
            ))
        except Exception:
            pass






