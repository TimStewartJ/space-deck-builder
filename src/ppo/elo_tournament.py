"""Elo tournament: round-robin evaluation between PPO checkpoints.

Loads N checkpoint files, plays every pair head-to-head using BatchRunner,
and computes Elo ratings to measure relative strength across training.

Usage:
    python -m src elo --checkpoints "models/ppo_agent_0415_*upd*0.pth" --games-per-pair 50
"""
from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass

import torch

from src.config import DataConfig, ModelConfig, load_checkpoint
from src.encoding.action_encoder import get_action_space_size
from src.ppo.batch_runner import BatchRunner
from src.ppo.ppo_actor_critic import PPOActorCritic
from src.ai.ppo_agent import PPOAgent
from src.utils.logger import log, set_disabled


K_FACTOR = 32
INITIAL_ELO = 1000.0


@dataclass
class EloResult:
    """Results for a single checkpoint after the tournament."""
    name: str
    path: str
    elo: float
    games_played: int
    wins: int
    losses: int

    @property
    def win_rate(self) -> float:
        return self.wins / self.games_played if self.games_played > 0 else 0.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score for player A given both ratings."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def elo_update(
    rating_a: float, rating_b: float, wins_a: int, total_games: int
) -> tuple[float, float]:
    """Compute new Elo ratings from aggregate match results.

    Simulates per-game updates: recalculates expected scores after each
    game result so that lopsided pairings don't produce inflated swings.

    Returns (new_rating_a, new_rating_b).
    """
    r_a, r_b = rating_a, rating_b
    losses_a = total_games - wins_a

    # Apply wins first, then losses (order within same-result games
    # doesn't matter since ratings are symmetric)
    for _ in range(wins_a):
        e_a = expected_score(r_a, r_b)
        r_a += K_FACTOR * (1.0 - e_a)
        r_b += K_FACTOR * (0.0 - (1.0 - e_a))

    for _ in range(losses_a):
        e_a = expected_score(r_a, r_b)
        r_a += K_FACTOR * (0.0 - e_a)
        r_b += K_FACTOR * (1.0 - (1.0 - e_a))

    return r_a, r_b


def _extract_label(path: str) -> str:
    """Build a short human-readable label from a checkpoint filename.

    Extracts the update number if present, otherwise uses the filename stem.
    E.g. 'models/ppo_agent_0415_0348_upd200_wins95.pth' → 'upd200'
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    for part in stem.split("_"):
        if part.startswith("upd"):
            return part
    return stem


def _validate_checkpoints(
    checkpoints: list[dict], paths: list[str]
) -> None:
    """Verify all checkpoints share compatible model configs.

    Raises ValueError on mismatch so we fail fast before the tournament.
    """
    ref_cfg = checkpoints[0].get("config", {}).get("model")
    for i, ckpt in enumerate(checkpoints[1:], start=1):
        cfg = ckpt.get("config", {}).get("model")
        if cfg != ref_cfg:
            raise ValueError(
                f"Model config mismatch between {paths[0]} and {paths[i]}:\n"
                f"  {paths[0]}: {ref_cfg}\n"
                f"  {paths[i]}: {cfg}"
            )


def _make_opponent_factory(
    state_dict: dict, card_names: list[str], device: str,
    model_config: ModelConfig | None = None,
) -> callable:
    """Create an opponent factory from an in-memory state_dict.

    Builds one shared model on the target device. Each factory call returns
    a lightweight PPOAgent wrapper pointing to that shared model, avoiding
    per-game GPU allocations.
    """
    from src.encoding.state_encoder import get_state_size

    state_dim = get_state_size(card_names)
    action_dim = get_action_space_size(card_names)
    cfg = model_config or ModelConfig()

    # Build and load the model once
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
        # Replace the default model with the shared pre-loaded one
        agent.model = shared_model
        return agent

    return factory


def run_tournament(
    checkpoint_paths: list[str],
    data_cfg: DataConfig,
    games_per_pair: int = 50,
    device: str = "cpu",
    num_concurrent: int = 32,
) -> list[EloResult]:
    """Run a round-robin Elo tournament between checkpoints.

    For each pair (A, B), runs games_per_pair games with A as the batched
    player and B as the per-step opponent. Game.start_game() randomizes
    player order, so both sides get fair first/second player distribution.

    Returns a list of EloResult sorted by Elo descending.
    """
    if len(checkpoint_paths) < 2:
        raise ValueError("Need at least 2 checkpoints for a tournament")
    if games_per_pair < 1:
        raise ValueError(f"games_per_pair must be positive, got {games_per_pair}")
    if num_concurrent < 1:
        raise ValueError(f"num_concurrent must be positive, got {num_concurrent}")

    # Load cards and card names
    cards = data_cfg.load_cards()
    card_names = data_cfg.get_card_names(cards)
    action_dim = get_action_space_size(card_names)

    # Load all checkpoints into memory
    log(f"Loading {len(checkpoint_paths)} checkpoints...")
    checkpoints: list[dict] = []
    labels: list[str] = []
    for path in checkpoint_paths:
        ckpt = load_checkpoint(path, map_location="cpu")
        checkpoints.append(ckpt)
        labels.append(_extract_label(path))
    log(f"Participants: {', '.join(labels)}")

    # Validate compatibility
    _validate_checkpoints(checkpoints, checkpoint_paths)

    # Resolve model config from the first checkpoint
    saved_model_cfg = checkpoints[0].get("config", {}).get("model")
    model_config = ModelConfig.from_dict(saved_model_cfg) if saved_model_cfg else ModelConfig()

    # Pre-compute dimensions
    from src.encoding.state_encoder import get_state_size
    state_dim = get_state_size(card_names)

    # Initialize Elo ratings and stats
    n = len(checkpoints)
    ratings = [INITIAL_ELO] * n
    wins_total = [0] * n
    losses_total = [0] * n
    games_total = [0] * n

    total_pairings = n * (n - 1) // 2
    log(f"Running {total_pairings} pairings x {games_per_pair} games = "
        f"{total_pairings * games_per_pair} total games")
    log("")

    pairing_num = 0
    start = time.time()

    for i in range(n):
        for j in range(i + 1, n):
            pairing_num += 1

            # Build model for player A (batched inference via BatchRunner)
            model_a = PPOActorCritic(
                state_dim=state_dim,
                action_dim=action_dim,
                num_cards=len(card_names),
                model_config=model_config,
            ).to(device)
            model_a.load_state_dict(checkpoints[i]["model_state_dict"])
            model_a.eval()

            # Build opponent factory for player B (shared model, per-step inference)
            opp_factory = _make_opponent_factory(
                checkpoints[j]["model_state_dict"], card_names, device,
                model_config=model_config,
            )

            runner = BatchRunner(
                model=model_a,
                card_names=card_names,
                cards=cards,
                action_dim=action_dim,
                device=torch.device(device),
                opponent_factory=opp_factory,
                num_concurrent=num_concurrent,
            )

            # Suppress verbose game-level logging during eval
            set_disabled(True)
            wins_a, losses_a, _ = runner.run_eval(games_per_pair)
            set_disabled(False)

            # Update stats
            wins_total[i] += wins_a
            losses_total[i] += losses_a
            wins_total[j] += losses_a  # B's wins = A's losses
            losses_total[j] += wins_a
            games_total[i] += games_per_pair
            games_total[j] += games_per_pair

            # Update Elo
            ratings[i], ratings[j] = elo_update(
                ratings[i], ratings[j], wins_a, games_per_pair
            )

            wr_a = wins_a / games_per_pair * 100
            log(f"  [{pairing_num}/{total_pairings}] {labels[i]} vs {labels[j]}: "
                f"{wins_a}/{games_per_pair} ({wr_a:.0f}%) -> "
                f"Elo {ratings[i]:.0f} / {ratings[j]:.0f}")

            # Clean up GPU memory between pairings
            del model_a, runner
            if device != "cpu":
                torch.cuda.empty_cache()

    elapsed = time.time() - start
    log(f"\nTournament complete in {elapsed:.1f}s")
    log("")

    # Build results sorted by Elo
    results = []
    for i in range(n):
        results.append(EloResult(
            name=labels[i],
            path=checkpoint_paths[i],
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
