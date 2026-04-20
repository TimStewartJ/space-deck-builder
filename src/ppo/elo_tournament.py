"""Elo tournament: round-robin evaluation between PPO checkpoints and
built-in agents (random, heuristic, simple).

Pairings that involve at least one PPO checkpoint run in parallel through
a long-lived :class:`InferenceServer` hosting every checkpoint keyed by
its label; a fleet of tournament worker processes drains a shared
pairing queue. Built-in vs. built-in pairings run inline in the
coordinator since they don't need the GPU.

Usage:
    python -m src elo --checkpoints "models/ppo_agent_0415_*upd*0.pth" --games-per-pair 50
    python -m src elo --checkpoints "models/*.pth" --agents random,heuristic --num-workers 4
"""
from __future__ import annotations

import glob
import math
import os
import time
from dataclasses import dataclass
from typing import Callable

import torch

from src.config import DataConfig, ModelConfig, load_checkpoint
from src.encoding.action_encoder import get_action_space_size
from src.ppo.ppo_actor_critic import PPOActorCritic
from src.ai.random_agent import RandomAgent
from src.ai.simple_agent import SimpleAgent
from src.ai.heuristic_agent import HeuristicAgent
from src.engine.game import Game
from src.utils.logger import log, set_disabled


INITIAL_ELO = 1000.0
_MLE_ITERATIONS = 100
_MLE_BASE = 10.0
_MLE_SPREAD = 400.0

# Built-in agent types that can participate in tournaments
BUILTIN_AGENT_TYPES = {"random", "heuristic", "simple"}

_AGENT_FACTORIES: dict[str, Callable] = {
    "random": lambda name: RandomAgent(name),
    "heuristic": lambda name: HeuristicAgent(name),
    "simple": lambda name: SimpleAgent(name),
}


def _make_opponent_factory(
    state_dict: dict, card_names: list[str], device: str,
    model_config: ModelConfig | None = None,
) -> Callable:
    """Create an opponent factory from an in-memory state_dict.

    Builds one shared model on the target device. Each factory call returns
    a lightweight PPOAgent wrapper pointing to that shared model, avoiding
    per-game GPU allocations.
    """
    from src.encoding.state_encoder import get_state_size
    from src.ai.ppo_agent import PPOAgent

    state_dim = get_state_size(card_names)
    action_dim = get_action_space_size(card_names)
    cfg = model_config or ModelConfig()

    shared_model = PPOActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        num_cards=len(card_names),
        model_config=cfg,
    ).to(device)
    shared_model.load_state_dict(state_dict)
    shared_model.eval()

    def factory() -> PPOAgent:
        agent = PPOAgent(
            "Opponent", card_names,
            device=device, main_device=device, simulation_device=device,
            model_config=cfg,
        )
        agent.model = shared_model
        return agent

    return factory


@dataclass
class EloResult:
    """Results for a single participant after the tournament."""
    name: str
    path: str
    elo: float
    games_played: int
    wins: int
    losses: int

    @property
    def win_rate(self) -> float:
        return self.wins / self.games_played if self.games_played > 0 else 0.0


@dataclass
class CheckpointParticipant:
    """A PPO checkpoint participant in the tournament."""
    label: str
    path: str
    state_dict: dict
    model_config: ModelConfig


@dataclass
class BuiltinParticipant:
    """A built-in rule-based agent participant (random, heuristic, simple)."""
    label: str
    agent_type: str

    @property
    def path(self) -> str:
        return ""


Participant = CheckpointParticipant | BuiltinParticipant


def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score for player A given both ratings."""
    return 1.0 / (1.0 + _MLE_BASE ** ((rating_b - rating_a) / _MLE_SPREAD))


def compute_mle_ratings(
    n: int,
    match_wins: dict[tuple[int, int], int],
    match_games: dict[tuple[int, int], int],
    *,
    iterations: int = _MLE_ITERATIONS,
    anchor: int = 0,
) -> list[float]:
    """Compute maximum likelihood Elo ratings from round-robin results.

    Finds ratings that maximize the likelihood of the observed win rates
    using iterative minorization-maximization (the same algorithm used by
    BayesElo and Ordo). Converges in ~50-100 iterations for typical
    tournament sizes.

    The anchor player's rating is fixed at INITIAL_ELO to set the scale.

    Args:
        n: Number of participants.
        match_wins: {(i, j): wins_for_i} for each pairing where i < j.
        match_games: {(i, j): total_games} for each pairing.
        iterations: Number of optimization iterations.
        anchor: Index of the player whose rating is fixed (prevents drift).

    Returns: List of n ratings.
    """
    ratings = [INITIAL_ELO] * n

    for _ in range(iterations):
        new_ratings = list(ratings)
        for i in range(n):
            if i == anchor:
                continue

            # Accumulate observed and expected wins across all opponents
            observed_wins = 0.0
            expected_wins = 0.0

            for j in range(n):
                if i == j:
                    continue
                key = (min(i, j), max(i, j))
                games = match_games.get(key, 0)
                if games == 0:
                    continue

                wins_ij = match_wins.get(key, 0)
                # If i is the second index in the key, wins for i = games - wins stored
                if key[0] == i:
                    w_i = wins_ij
                else:
                    w_i = games - wins_ij

                observed_wins += w_i
                expected_wins += games * expected_score(ratings[i], ratings[j])

            if expected_wins > 0:
                # Multiplicative update: shift rating to make expected match observed
                ratio = observed_wins / expected_wins
                new_ratings[i] = ratings[i] + _MLE_SPREAD * math.log10(ratio)

        ratings = new_ratings

    return ratings


def _extract_label(path: str) -> str:
    """Build a short human-readable label from a checkpoint filename.

    Prefers an ``upd<N>`` suffix combined with the training-run timestamp
    (``MMDD_HHMM``) so labels are unique when comparing checkpoints from
    different runs (e.g. a sum-pool sweep vs an attention-pool sweep). Falls
    back to the bare ``upd<N>`` tag, and finally to the filename stem.

    Examples:
        models/ppo_agent_0416_2308_upd30_wins200.pth -> '0416_2308_upd30'
        models/ppo_agent_upd30.pth                   -> 'upd30'
        models/custom_name.pth                       -> 'custom_name'
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("_")
    upd = next((p for p in parts if p.startswith("upd")), None)
    # Pattern from ppo_trainer: ppo_agent_<MMDD>_<HHMM>_upd<N>_wins<W>
    # The two tokens immediately preceding 'upd' are the date/time stamp.
    if upd is not None:
        idx = parts.index(upd)
        if idx >= 2 and parts[idx - 1].isdigit() and parts[idx - 2].isdigit():
            return f"{parts[idx - 2]}_{parts[idx - 1]}_{upd}"
        return upd
    return stem


def _validate_checkpoints(
    participants: list[CheckpointParticipant],
) -> None:
    """Placeholder for cross-checkpoint compatibility validation.

    Participants may have different model architectures (e.g. sum vs attention
    pooling, mlp vs attention actor, different hidden sizes or card_emb_dims).
    Every ``ModelConfig`` field is purely internal to a single model's forward
    pass — runtime compatibility across participants is determined by the
    shared card set (which fixes state_dim, action_dim, and num_cards), not by
    ModelConfig. Each participant's model is instantiated from its own
    ``model_config`` in ``_make_opponent_factory``, so heterogeneous
    architectures play correctly. Comparing them is the whole point of ELO.

    Kept as an extension point for future checks that are genuinely required
    for play (e.g. divergent card sets / state encodings).
    """
    return


def _play_builtin_games(
    agent_type_a: str,
    agent_type_b: str,
    cards: list,
    card_index_map: dict[str, int],
    num_games: int,
) -> tuple[int, int]:
    """Play games between two built-in agents without neural net inference.

    Uses the same Game/add_player/start_game flow as BatchRunner for
    consistency. Returns (wins_a, wins_b).
    """
    factory_a = _AGENT_FACTORIES[agent_type_a]
    factory_b = _AGENT_FACTORIES[agent_type_b]
    name_a = f"{agent_type_a.capitalize()}_A"
    name_b = f"{agent_type_b.capitalize()}_B"

    wins_a = 0
    wins_b = 0

    for _ in range(num_games):
        game = Game(cards, card_index_map=card_index_map)
        game.add_player(name_a, factory_a(name_a))
        game.add_player(name_b, factory_b(name_b))
        game.start_game()

        while not game.is_game_over:
            player = game.current_player
            action = player.make_decision(game)
            game.apply_decision(action)

        winner = game.get_winner()
        if winner == name_a:
            wins_a += 1
        else:
            wins_b += 1

    return wins_a, wins_b


def build_participants(
    checkpoint_paths: list[str],
    agent_types: list[str] | None = None,
) -> list[Participant]:
    """Build participant list from checkpoint paths and/or agent type names.

    Loads and validates checkpoints, creates BuiltinParticipant entries for
    each requested agent type.
    """
    participants: list[Participant] = []

    # Load checkpoint participants
    checkpoint_participants: list[CheckpointParticipant] = []
    for path in checkpoint_paths:
        ckpt = load_checkpoint(path, map_location="cpu")
        label = _extract_label(path)
        saved_cfg = ckpt.get("config", {}).get("model")
        model_config = ModelConfig.from_dict(saved_cfg) if saved_cfg else ModelConfig()
        cp = CheckpointParticipant(
            label=label, path=path,
            state_dict=ckpt["model_state_dict"],
            model_config=model_config,
        )
        checkpoint_participants.append(cp)
        participants.append(cp)

    # Validate checkpoint compatibility
    _validate_checkpoints(checkpoint_participants)

    # Add builtin agent participants
    for agent_type in (agent_types or []):
        if agent_type not in BUILTIN_AGENT_TYPES:
            raise ValueError(
                f"Unknown agent type '{agent_type}'. "
                f"Valid types: {', '.join(sorted(BUILTIN_AGENT_TYPES))}"
            )
        participants.append(BuiltinParticipant(
            label=agent_type, agent_type=agent_type,
        ))

    return participants


def run_tournament(
    checkpoint_paths: list[str],
    data_cfg: DataConfig,
    games_per_pair: int = 50,
    device: str | None = None,
    num_concurrent: int | None = None,
    agent_types: list[str] | None = None,
    num_workers: int = 1,
    collect_replays: bool = False,
    replay_output_dir: str | None = None,
) -> list[EloResult]:
    """Run a round-robin Elo tournament between checkpoints and/or agents.

    Pairings are dispatched in parallel: one long-lived
    :class:`InferenceServer` hosts every PPO checkpoint keyed by its
    label, and ``num_workers`` tournament worker processes pull
    :class:`PairingTask`s off a shared queue. N pairings run concurrently
    whenever ``num_workers > 1``. Built-in vs. built-in pairings skip the
    server and run inline in the coordinator.

    Game.start_game() randomizes player order, so both sides get fair
    first/second player distribution.

    Returns a list of EloResult sorted by Elo descending.
    """
    from src.config import DeviceConfig, RunConfig
    if device is None:
        device = DeviceConfig().simulation_device
    device = DeviceConfig.resolve(device)
    if num_concurrent is None:
        num_concurrent = RunConfig().num_concurrent
    if games_per_pair < 1:
        raise ValueError(f"games_per_pair must be positive, got {games_per_pair}")
    if num_concurrent < 1:
        raise ValueError(f"num_concurrent must be positive, got {num_concurrent}")
    if num_workers < 1:
        raise ValueError(f"num_workers must be positive, got {num_workers}")

    # Build participant list
    participants = build_participants(checkpoint_paths, agent_types)
    n = len(participants)
    if n < 2:
        raise ValueError(
            f"Need at least 2 participants for a tournament, got {n}"
        )

    # Load cards
    cards = data_cfg.load_cards()
    card_names = data_cfg.get_card_names(cards)
    registry = data_cfg.build_registry(cards)
    action_dim = get_action_space_size(card_names)
    from src.encoding.state_encoder import get_state_size
    state_dim = get_state_size(card_names)

    labels = [p.label for p in participants]
    log(f"Participants ({n}): {', '.join(labels)}")

    # Match result accumulators for MLE rating computation
    match_wins: dict[tuple[int, int], int] = {}
    match_games: dict[tuple[int, int], int] = {}
    wins_total = [0] * n
    losses_total = [0] * n
    games_total = [0] * n

    total_pairings = n * (n - 1) // 2
    log(f"Running {total_pairings} pairings x {games_per_pair} games = "
        f"{total_pairings * games_per_pair} total games")
    if num_workers > 1:
        log(f"Using {num_workers} workers running pairings in parallel")
    log("")

    # Partition pairings: builtin-vs-builtin stay in-process; any pairing
    # with at least one PPO side goes through the parallel tournament
    # workers via the InferenceServer.
    ppo_pairings: list[tuple[int, int]] = []
    builtin_pairings: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            p_a = participants[i]
            p_b = participants[j]
            if isinstance(p_a, CheckpointParticipant) or isinstance(p_b, CheckpointParticipant):
                ppo_pairings.append((i, j))
            else:
                builtin_pairings.append((i, j))

    start = time.time()
    pairings_done = 0

    # Play builtin-vs-builtin pairings inline — no server needed.
    set_disabled(True)
    for (i, j) in builtin_pairings:
        pair_start = time.time()
        p_a = participants[i]
        p_b = participants[j]
        wins_a, wins_b = _play_builtin_games(
            p_a.agent_type, p_b.agent_type,
            cards, registry.card_index_map, games_per_pair,
        )
        pair_elapsed = time.time() - pair_start
        actual_games = wins_a + wins_b
        match_wins[(i, j)] = wins_a
        match_games[(i, j)] = actual_games
        wins_total[i] += wins_a
        losses_total[i] += wins_b
        wins_total[j] += wins_b
        losses_total[j] += wins_a
        games_total[i] += actual_games
        games_total[j] += actual_games
        pairings_done += 1
        wr_a = wins_a / actual_games * 100 if actual_games > 0 else 0
        log(f"  [{pairings_done}/{total_pairings}] {labels[i]} vs {labels[j]}: "
            f"{wins_a}/{actual_games} ({wr_a:.0f}%)  ({pair_elapsed:.1f}s)")
    set_disabled(False)

    # Dispatch PPO-involving pairings through a parallel worker fleet.
    if ppo_pairings:
        pairings_done = _run_parallel_pairings(
            ppo_pairings=ppo_pairings,
            participants=participants,
            labels=labels,
            games_per_pair=games_per_pair,
            cards=cards,
            card_names=card_names,
            card_index_map=registry.card_index_map,
            action_dim=action_dim,
            state_dim=state_dim,
            device=device,
            num_workers=num_workers,
            num_concurrent=num_concurrent,
            data_cfg=data_cfg,
            match_wins=match_wins,
            match_games=match_games,
            wins_total=wins_total,
            losses_total=losses_total,
            games_total=games_total,
            pairings_done=pairings_done,
            total_pairings=total_pairings,
        )

    elapsed = time.time() - start
    log(f"\nGames complete in {elapsed:.1f}s")

    # Compute maximum likelihood Elo ratings from all match results
    ratings = compute_mle_ratings(n, match_wins, match_games)

    # Build results sorted by Elo
    results = []
    for i in range(n):
        results.append(EloResult(
            name=labels[i],
            path=participants[i].path,
            elo=ratings[i],
            games_played=games_total[i],
            wins=wins_total[i],
            losses=losses_total[i],
        ))
    results.sort(key=lambda r: r.elo, reverse=True)

    # Print leaderboard
    log("\n" + "=" * 60)
    log("ELO LEADERBOARD")
    log("=" * 60)
    log(f"{'Rank':<5} {'Name':<15} {'Elo':>6} {'W':>5} {'L':>5} {'Win%':>6}")
    log("-" * 60)
    for rank, r in enumerate(results, 1):
        log(f"{rank:<5} {r.name:<15} {r.elo:>6.0f} {r.wins:>5} {r.losses:>5} "
            f"{r.win_rate * 100:>5.1f}%")
    log("=" * 60)

    # Collect replays and generate comparative dashboard if requested.
    # Runs a second pass over all PPO pairings using run_analysis to
    # collect per-decision data. This is separate from the main
    # tournament to avoid slowing down the rating computation.
    if collect_replays:
        import os as _os
        from datetime import datetime as _dt
        if replay_output_dir is None:
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            replay_output_dir = _os.path.join("analysis", f"elo_{ts}")
        _os.makedirs(replay_output_dir, exist_ok=True)

        log(f"\nCollecting replays for {len(ppo_pairings)} PPO pairings...")
        replay_files = _collect_tournament_replays(
            ppo_pairings=ppo_pairings,
            participants=participants,
            labels=labels,
            games_per_pair=min(games_per_pair, 50),  # cap replay collection
            cards=cards,
            card_names=card_names,
            card_index_map=registry.card_index_map,
            action_dim=action_dim,
            state_dim=state_dim,
            device=device,
            num_workers=num_workers,
            num_concurrent=num_concurrent,
            output_dir=replay_output_dir,
            data_cfg=data_cfg,
        )

        if replay_files:
            try:
                from src.analysis.dashboard import generate_comparative_dashboard
                dash_path = generate_comparative_dashboard(
                    replay_paths=replay_files,
                    elo_results=results,
                    output_path=_os.path.join(replay_output_dir, "tournament_dashboard.html"),
                )
                log(f"Comparative dashboard: {dash_path}")
            except Exception as e:
                log(f"Warning: dashboard generation failed: {e}")

    return results


def _dispatch_parallel_pairings(
    *,
    tasks: list,
    model_registry: dict,
    num_workers: int,
    num_concurrent: int,
    data_cfg: DataConfig,
    card_names: list[str],
    card_index_map: dict,
    action_dim: int,
    state_dim: int,
    device: str,
):
    """Run a list of ``PairingTask``s through one shared
    :class:`InferenceServer` and a fleet of tournament workers, yielding
    :class:`TournamentResult`s as they arrive.

    Owns the full lifecycle: server start, worker spawn, task + sentinel
    enqueue, and shutdown in ``finally``. Raises immediately on any
    :class:`WorkerError` so callers fail fast instead of hanging.

    Both the regular Elo path and the replay-collection path use this
    helper. ``model_registry`` should already include every PPO model
    referenced by any task in ``tasks``; built-in agents don't register.
    """
    import multiprocessing as mp
    from src.ppo.mp_inference_server import InferenceServer, WorkerError
    from src.ppo.mp_sim_worker import (
        TournamentResult, tournament_worker_main,
    )

    if not tasks:
        return

    ctx = mp.get_context("spawn")
    actual_workers = min(num_workers, len(tasks))
    server = InferenceServer(
        models=model_registry, device=torch.device(device),
        num_workers=actual_workers, ctx=ctx,
    )
    server.start()

    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    # Dividing num_concurrent by worker count keeps total GPU batch size
    # roughly comparable to the serial path.
    per_worker_concurrent = max(1, num_concurrent // actual_workers)

    processes = []
    try:
        for worker_id in range(actual_workers):
            proc = ctx.Process(
                target=tournament_worker_main,
                args=(
                    worker_id,
                    per_worker_concurrent,
                    data_cfg.to_dict(),
                    card_names,
                    card_index_map,
                    action_dim,
                    state_dim,
                    task_queue,
                    result_queue,
                    server.request_queue,
                    server.response_queues[worker_id],
                ),
            )
            proc.start()
            processes.append(proc)

        for task in tasks:
            task_queue.put(task)
        for _ in range(actual_workers):
            task_queue.put(None)

        remaining = len(tasks)
        while remaining > 0:
            item = result_queue.get()
            if isinstance(item, WorkerError):
                # Fail fast: raising shuts down the server and terminates
                # workers via the finally block below.
                raise RuntimeError(f"Tournament worker error:\n{item.error}")
            assert isinstance(item, TournamentResult)
            yield item
            remaining -= 1

    finally:
        server.stop()
        for proc in processes:
            proc.join(timeout=5.0)
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)


def _build_model_registry(
    participants: list, card_names: list[str],
    action_dim: int, state_dim: int,
) -> dict:
    """Build the ``label -> PPOActorCritic`` registry the InferenceServer
    needs. One model per :class:`CheckpointParticipant`; built-ins are
    served inline by workers and don't appear here.
    """
    models: dict[str, PPOActorCritic] = {}
    for p in participants:
        if isinstance(p, CheckpointParticipant):
            m = PPOActorCritic(
                state_dim=state_dim, action_dim=action_dim,
                num_cards=len(card_names), model_config=p.model_config,
            )
            m.load_state_dict(p.state_dict)
            m.eval()
            models[p.label] = m
    return models


def _kind_and_id(p) -> tuple[str, str]:
    """Map a participant to the (kind, id) pair the worker uses to build
    the right player slot for a pairing.
    """
    if isinstance(p, CheckpointParticipant):
        return ("ppo", p.label)
    return ("builtin", p.agent_type)


def _collect_tournament_replays(
    *,
    ppo_pairings: list[tuple[int, int]],
    participants: list,
    labels: list[str],
    games_per_pair: int,
    cards: list,
    card_names: list[str],
    card_index_map: dict,
    action_dim: int,
    state_dim: int,
    device: str,
    num_workers: int,
    num_concurrent: int,
    output_dir: str,
    data_cfg: 'DataConfig',
) -> list[str]:
    """Run a replay-collection pass over PPO pairings in parallel.

    Reuses the regular tournament's :class:`InferenceServer` + worker fleet
    via :func:`_dispatch_parallel_pairings`. Each pairing produces one
    ``{labels[i]}_vs_{labels[j]}.json.gz`` file in ``output_dir``; the
    worker writes it atomically and reports the path back via
    :class:`TournamentResult.replay_path`.

    Side A is recorded (the focal PPO checkpoint). In `ppo_pairings` side A
    is always a CheckpointParticipant since checkpoints precede builtins
    in :func:`build_participants`.
    """
    import os
    import random as _random
    from src.ppo.mp_sim_worker import PairingTask

    model_registry = _build_model_registry(
        participants, card_names, action_dim, state_dim,
    )

    tasks: list[PairingTask] = []
    for pairing_id, (i, j) in enumerate(ppo_pairings):
        ka, ida = _kind_and_id(participants[i])
        kb, idb = _kind_and_id(participants[j])
        replay_path = os.path.join(
            output_dir, f"{labels[i]}_vs_{labels[j]}.json.gz"
        )
        tasks.append(PairingTask(
            pairing_id=pairing_id, i=i, j=j,
            num_games=games_per_pair,
            seed=_random.randint(0, 1_000_000_000),
            side_a_kind=ka, side_b_kind=kb,
            side_a_id=ida, side_b_id=idb,
            collect_replays=True,
            replay_output_path=replay_path,
        ))

    replay_files: list[str] = []
    t0 = time.time()
    n_done = 0
    for result in _dispatch_parallel_pairings(
        tasks=tasks,
        model_registry=model_registry,
        num_workers=num_workers,
        num_concurrent=num_concurrent,
        data_cfg=data_cfg,
        card_names=card_names,
        card_index_map=card_index_map,
        action_dim=action_dim,
        state_dim=state_dim,
        device=device,
    ):
        n_done += 1
        actual_games = result.wins_a + result.wins_b
        wr = result.wins_a / actual_games * 100 if actual_games > 0 else 0
        elapsed = time.time() - t0
        pairing_label = f"{labels[result.i]}_vs_{labels[result.j]}"
        log(f"  Replay [{n_done}/{len(tasks)}] {pairing_label}: "
            f"{result.wins_a}/{actual_games} ({wr:.0f}%)  "
            f"({elapsed:.1f}s elapsed)")
        if result.replay_path:
            replay_files.append(result.replay_path)

    return replay_files


def _run_parallel_pairings(
    *,
    ppo_pairings: list[tuple[int, int]],
    participants: list,
    labels: list[str],
    games_per_pair: int,
    cards: list,
    card_names: list[str],
    card_index_map: dict,
    action_dim: int,
    state_dim: int,
    device: str,
    num_workers: int,
    num_concurrent: int,
    data_cfg: DataConfig,
    match_wins: dict,
    match_games: dict,
    wins_total: list,
    losses_total: list,
    games_total: list,
    pairings_done: int,
    total_pairings: int,
) -> int:
    """Run all PPO-involving pairings in parallel through one shared
    :class:`InferenceServer` and a fleet of tournament workers.

    Returns the updated ``pairings_done`` counter. Accumulates results
    into the caller-supplied ``match_*`` / ``wins_total`` / ``losses_total``
    / ``games_total`` structures by participant index.
    """
    import random as _random
    from src.ppo.mp_sim_worker import PairingTask

    model_registry = _build_model_registry(
        participants, card_names, action_dim, state_dim,
    )

    tasks: list[PairingTask] = []
    for pairing_id, (i, j) in enumerate(ppo_pairings):
        ka, ida = _kind_and_id(participants[i])
        kb, idb = _kind_and_id(participants[j])
        tasks.append(PairingTask(
            pairing_id=pairing_id, i=i, j=j,
            num_games=games_per_pair,
            seed=_random.randint(0, 1_000_000_000),
            side_a_kind=ka, side_b_kind=kb,
            side_a_id=ida, side_b_id=idb,
        ))

    # Mark every pairing as "started" at dispatch time for a reasonable
    # per-pairing wall-time estimate. Parallel pairings will show
    # overlapping durations; that's expected.
    t0 = time.time()
    for result in _dispatch_parallel_pairings(
        tasks=tasks,
        model_registry=model_registry,
        num_workers=num_workers,
        num_concurrent=num_concurrent,
        data_cfg=data_cfg,
        card_names=card_names,
        card_index_map=card_index_map,
        action_dim=action_dim,
        state_dim=state_dim,
        device=device,
    ):
        i, j = result.i, result.j
        wins_a, wins_b = result.wins_a, result.wins_b
        actual_games = wins_a + wins_b
        match_wins[(i, j)] = wins_a
        match_games[(i, j)] = actual_games
        wins_total[i] += wins_a
        losses_total[i] += wins_b
        wins_total[j] += wins_b
        losses_total[j] += wins_a
        games_total[i] += actual_games
        games_total[j] += actual_games
        pairings_done += 1
        wr_a = wins_a / actual_games * 100 if actual_games > 0 else 0
        elapsed = time.time() - t0
        log(f"  [{pairings_done}/{total_pairings}] {labels[i]} vs {labels[j]}: "
            f"{wins_a}/{actual_games} ({wr_a:.0f}%)  ({elapsed:.1f}s)")

    return pairings_done



def resolve_checkpoint_paths(patterns: list[str]) -> list[str]:
    """Expand glob patterns into sorted, deduplicated checkpoint paths.

    Handles Windows where shell glob expansion is unreliable.
    """
    paths: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        expanded = glob.glob(pattern)
        if not expanded:
            # Treat as a literal path
            if os.path.isfile(pattern):
                expanded = [pattern]
            else:
                log(f"Warning: no files matched pattern '{pattern}'")
                continue
        for p in expanded:
            norm = os.path.normpath(p)
            if norm not in seen:
                seen.add(norm)
                paths.append(norm)

    # Sort by filename for deterministic ordering
    paths.sort(key=lambda p: os.path.basename(p))
    return paths
