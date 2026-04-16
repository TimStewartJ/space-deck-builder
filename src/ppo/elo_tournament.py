"""Elo tournament: round-robin evaluation between PPO checkpoints and
built-in agents (random, heuristic, simple).

Supports multi-worker parallelism for PPO-vs-builtin pairings via
MultiProcessBatchRunner, and direct game simulation for builtin-vs-builtin.

Usage:
    python -m src elo --checkpoints "models/ppo_agent_0415_*upd*0.pth" --games-per-pair 50
    python -m src elo --checkpoints "models/*.pth" --agents random,heuristic --num-workers 4
"""
from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass
from typing import Callable

import torch

from src.config import DataConfig, ModelConfig, load_checkpoint
from src.encoding.action_encoder import get_action_space_size
from src.ppo.batch_runner import BatchRunner
from src.ppo.ppo_actor_critic import PPOActorCritic
from src.ai.ppo_agent import PPOAgent
from src.ai.random_agent import RandomAgent
from src.ai.simple_agent import SimpleAgent
from src.ai.heuristic_agent import HeuristicAgent
from src.engine.game import Game
from src.utils.logger import log, set_disabled


K_FACTOR = 32
INITIAL_ELO = 1000.0

# Built-in agent types that can participate in tournaments
BUILTIN_AGENT_TYPES = {"random", "heuristic", "simple"}

_AGENT_FACTORIES: dict[str, Callable] = {
    "random": lambda name: RandomAgent(name),
    "heuristic": lambda name: HeuristicAgent(name),
    "simple": lambda name: SimpleAgent(name),
}


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
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def elo_update(
    rating_a: float, rating_b: float, wins_a: int, total_games: int
) -> tuple[float, float]:
    """Compute new Elo ratings from aggregate match results.

    Uses the standard batch Elo formula: compute expected score from
    current ratings, compare to actual win rate, and apply a single
    update scaled by K and number of games.

    Returns (new_rating_a, new_rating_b).
    """
    if total_games == 0:
        return rating_a, rating_b

    e_a = expected_score(rating_a, rating_b)
    actual_a = wins_a / total_games
    delta = K_FACTOR * (actual_a - e_a)

    return rating_a + delta, rating_b - delta


def _extract_label(path: str) -> str:
    """Build a short human-readable label from a checkpoint filename.

    Extracts the update number if present, otherwise uses the filename stem.
    E.g. 'models/ppo_agent_0415_0348_upd200_wins95.pth' -> 'upd200'
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    for part in stem.split("_"):
        if part.startswith("upd"):
            return part
    return stem


def _validate_checkpoints(
    participants: list[CheckpointParticipant],
) -> None:
    """Verify all checkpoint participants share compatible model configs.

    Raises ValueError on mismatch so we fail fast before the tournament.
    """
    if len(participants) < 2:
        return
    ref = participants[0]
    for p in participants[1:]:
        if p.model_config != ref.model_config:
            raise ValueError(
                f"Model config mismatch between {ref.path} and {p.path}:\n"
                f"  {ref.path}: {ref.model_config}\n"
                f"  {p.path}: {p.model_config}"
            )


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


def _play_ppo_vs_ppo(
    ckpt_a: CheckpointParticipant,
    ckpt_b: CheckpointParticipant,
    cards: list,
    card_names: list[str],
    action_dim: int,
    state_dim: int,
    device: str,
    num_concurrent: int,
    games_per_pair: int,
    data_cfg: DataConfig,
    num_workers: int,
    registry=None,
) -> tuple[int, int]:
    """Play games between two PPO checkpoints with batched GPU inference.

    Uses MultiProcessBatchRunner with Player B loaded as a snapshot on the
    InferenceServer, so both players get batched GPU inference. Player A
    is the "training" model, Player B is the snapshot opponent.
    Returns (wins_a, wins_b).
    """
    from src.ppo.mp_batch_runner import MultiProcessBatchRunner

    model_a = PPOActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        num_cards=len(card_names),
        model_config=ckpt_a.model_config,
    ).to(device)
    model_a.load_state_dict(ckpt_a.state_dict)
    model_a.eval()

    per_worker_concurrent = max(1, num_concurrent // max(1, num_workers))

    runner = MultiProcessBatchRunner(
        model=model_a,
        card_names=card_names,
        cards=cards,
        action_dim=action_dim,
        device=device,
        data_config=data_cfg,
        opponent_spec="random",  # fallback spec, unused when self_play_ratio=1.0
        num_concurrent=per_worker_concurrent,
        num_workers=num_workers,
        registry=registry,
        snapshot_state_dicts=[(ckpt_b.label, ckpt_b.state_dict)],
        self_play_ratio=1.0,
    )

    set_disabled(True)
    wins_a, losses_a, _ = runner.run_eval(games_per_pair)
    set_disabled(False)

    del model_a, runner
    if device != "cpu":
        torch.cuda.empty_cache()

    return wins_a, losses_a


def _play_ppo_vs_builtin(
    ckpt: CheckpointParticipant,
    builtin: BuiltinParticipant,
    cards: list,
    card_names: list[str],
    action_dim: int,
    state_dim: int,
    device: str,
    num_concurrent: int,
    games_per_pair: int,
    data_cfg: DataConfig,
    num_workers: int,
    registry=None,
) -> tuple[int, int]:
    """Play games between a PPO checkpoint and a built-in agent.

    Uses MultiProcessBatchRunner when num_workers > 1 for parallelism,
    otherwise falls back to single-process BatchRunner.
    Returns (wins_ppo, wins_builtin).
    """
    model = PPOActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        num_cards=len(card_names),
        model_config=ckpt.model_config,
    ).to(device)
    model.load_state_dict(ckpt.state_dict)
    model.eval()

    if num_workers > 1:
        from src.ppo.mp_batch_runner import MultiProcessBatchRunner
        # num_concurrent is a global target; divide across workers
        per_worker_concurrent = max(1, num_concurrent // num_workers)
        runner = MultiProcessBatchRunner(
            model=model,
            card_names=card_names,
            cards=cards,
            action_dim=action_dim,
            device=device,
            data_config=data_cfg,
            opponent_spec=builtin.agent_type,
            num_concurrent=per_worker_concurrent,
            num_workers=num_workers,
            registry=registry,
        )
    else:
        factory = _AGENT_FACTORIES[builtin.agent_type]
        runner = BatchRunner(
            model=model,
            card_names=card_names,
            cards=cards,
            action_dim=action_dim,
            device=torch.device(device),
            opponent_factory=lambda: factory(builtin.agent_type.capitalize()),
            num_concurrent=num_concurrent,
        )

    set_disabled(True)
    wins_ppo, losses_ppo, _ = runner.run_eval(games_per_pair)
    set_disabled(False)

    del model, runner
    if device != "cpu":
        torch.cuda.empty_cache()

    return wins_ppo, losses_ppo


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
) -> list[EloResult]:
    """Run a round-robin Elo tournament between checkpoints and/or agents.

    Supports three pairing types:
    - PPO vs PPO: MultiProcessBatchRunner with snapshot (both batched on GPU)
    - PPO vs Builtin: BatchRunner or MultiProcessBatchRunner (multi-worker)
    - Builtin vs Builtin: direct game simulation (no neural net)

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

    # Initialize Elo ratings and stats
    ratings = [INITIAL_ELO] * n
    wins_total = [0] * n
    losses_total = [0] * n
    games_total = [0] * n

    total_pairings = n * (n - 1) // 2
    log(f"Running {total_pairings} pairings x {games_per_pair} games = "
        f"{total_pairings * games_per_pair} total games")
    if num_workers > 1:
        log(f"Using {num_workers} workers for PPO pairings")
    log("")

    pairing_num = 0
    start = time.time()

    for i in range(n):
        for j in range(i + 1, n):
            pairing_num += 1
            p_a = participants[i]
            p_b = participants[j]
            pair_start = time.time()

            # Dispatch to the appropriate pairing handler
            if isinstance(p_a, CheckpointParticipant) and isinstance(p_b, CheckpointParticipant):
                wins_a, wins_b = _play_ppo_vs_ppo(
                    p_a, p_b, cards, card_names, action_dim, state_dim,
                    device, num_concurrent, games_per_pair, data_cfg,
                    num_workers, registry=registry,
                )
            elif isinstance(p_a, CheckpointParticipant) and isinstance(p_b, BuiltinParticipant):
                wins_a, wins_b = _play_ppo_vs_builtin(
                    p_a, p_b, cards, card_names, action_dim, state_dim,
                    device, num_concurrent, games_per_pair, data_cfg,
                    num_workers, registry=registry,
                )
            elif isinstance(p_a, BuiltinParticipant) and isinstance(p_b, CheckpointParticipant):
                # Swap so PPO is always the batched player
                wins_b, wins_a = _play_ppo_vs_builtin(
                    p_b, p_a, cards, card_names, action_dim, state_dim,
                    device, num_concurrent, games_per_pair, data_cfg,
                    num_workers, registry=registry,
                )
            else:
                # Both builtin
                set_disabled(True)
                wins_a, wins_b = _play_builtin_games(
                    p_a.agent_type, p_b.agent_type,
                    cards, registry.card_index_map, games_per_pair,
                )
                set_disabled(False)

            pair_elapsed = time.time() - pair_start

            # Use actual completed games (handles partial MP eval results)
            actual_games = wins_a + wins_b

            # Update stats
            wins_total[i] += wins_a
            losses_total[i] += wins_b
            wins_total[j] += wins_b
            losses_total[j] += wins_a
            games_total[i] += actual_games
            games_total[j] += actual_games

            # Update Elo
            ratings[i], ratings[j] = elo_update(
                ratings[i], ratings[j], wins_a, actual_games
            )

            wr_a = wins_a / actual_games * 100 if actual_games > 0 else 0
            log(f"  [{pairing_num}/{total_pairings}] {labels[i]} vs {labels[j]}: "
                f"{wins_a}/{actual_games} ({wr_a:.0f}%) -> "
                f"Elo {ratings[i]:.0f} / {ratings[j]:.0f}  ({pair_elapsed:.1f}s)")

    elapsed = time.time() - start
    log(f"\nTournament complete in {elapsed:.1f}s")
    log("")

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

    return results


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
