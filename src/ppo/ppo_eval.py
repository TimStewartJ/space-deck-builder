"""Standalone evaluation: run a trained model against fixed opponent types.

Extracted from the trainer's mid-training eval loop so it can be used
independently via ``python -m src eval`` or called programmatically.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from src.config import DataConfig, DeviceConfig, ModelConfig, PPOConfig, load_checkpoint
from src.encoding.action_encoder import get_action_space_size
from src.encoding.state_encoder import get_state_size
from src.ppo.batch_runner import BatchRunner
from src.ppo.opponent_pool import OpponentPool
from src.ppo.ppo_actor_critic import PPOActorCritic
from src.utils.logger import set_disabled, set_verbose


@dataclass
class OpponentResult:
    """Per-opponent-type evaluation results."""
    opponent: str
    wins: int
    games: int
    win_rate: float
    avg_steps: float


@dataclass
class EvalResult:
    """Aggregated evaluation results."""
    per_opponent: list[OpponentResult]
    total_wins: int
    total_games: int
    overall_win_rate: float
    elapsed_seconds: float


def _distribute_games(total: int, num_buckets: int, *, min_per_bucket: int = 0) -> list[int]:
    """Spread *total* games across *num_buckets*, distributing the remainder.

    When *min_per_bucket* > 0, every bucket gets at least that many games
    (the effective total may exceed *total* to satisfy the minimum).
    """
    base = total // num_buckets
    remainder = total % num_buckets
    schedule = [base + (1 if i < remainder else 0) for i in range(num_buckets)]
    if min_per_bucket > 0:
        schedule = [max(min_per_bucket, g) for g in schedule]
    return schedule


def evaluate(
    model: PPOActorCritic,
    *,
    data_cfg: DataConfig,
    device: str,
    opponents: str = "random,heuristic,simple",
    eval_games: int = 100,
    num_concurrent: int = 1024,
    num_workers: int = 1,
    ppo_config: PPOConfig | None = None,
    label: str | None = None,
    min_games_per_opponent: int = 0,
) -> EvalResult:
    """Run evaluation games against each opponent type and return results.

    Args:
        model: A trained PPOActorCritic model (will be moved to *device*).
        data_cfg: Card/data configuration (cards and registry are built internally).
        device: Resolved device string (e.g. ``"cuda"`` or ``"cpu"``).
        opponents: Comma-separated opponent type names (weights ignored).
        eval_games: Total games to distribute across opponent types.
        num_concurrent: Maximum concurrent games in BatchRunner.
        num_workers: Worker processes (1 = single-process, >1 = multi-process).
        ppo_config: Optional PPO config forwarded to BatchRunner.
        label: Optional label printed in the header (e.g. ``"update 10"``).
        min_games_per_opponent: Minimum games per opponent type. When > 0, the
            effective total may exceed *eval_games* to ensure coverage.

    Returns:
        An :class:`EvalResult` with per-opponent and aggregate statistics.
    """
    cards = data_cfg.load_cards()
    registry = data_cfg.build_registry(cards)
    card_names = registry.card_names
    action_dim = get_action_space_size(card_names)

    # Parse and normalize opponent types (strip weights, lowercase for registry compat)
    opp_types = [entry.split(":")[0].strip().lower() for entry in opponents.split(",")]
    pool = OpponentPool(opponent_spec=",".join(opp_types))

    games_schedule = _distribute_games(
        eval_games, len(opp_types), min_per_bucket=min_games_per_opponent,
    )

    header = "Evaluating"
    if label:
        header += f" ({label})"
    header += f" — {eval_games} games × {len(opp_types)} opponent types"
    print(header)

    use_mp = num_workers > 1
    per_opponent: list[OpponentResult] = []
    total_wins = 0
    total_games = 0
    start = time.perf_counter()

    for opp_type, num_games in zip(opp_types, games_schedule):
        if num_games == 0:
            continue

        if use_mp:
            from src.ppo.mp_batch_runner import MultiProcessBatchRunner
            runner = MultiProcessBatchRunner(
                model=model,
                card_names=card_names,
                cards=cards,
                action_dim=action_dim,
                device=torch.device(device),
                data_config=data_cfg,
                opponent_spec=opp_type,
                num_concurrent=min(num_games, num_concurrent),
                num_workers=num_workers,
                ppo_config=ppo_config,
                registry=registry,
            )
        else:
            factory = pool.make_factory_for_type(opp_type)
            runner = BatchRunner(
                model=model,
                card_names=card_names,
                cards=cards,
                action_dim=action_dim,
                device=torch.device(device),
                opponent_factory=factory,
                num_concurrent=min(num_games, num_concurrent),
                ppo_config=ppo_config,
                registry=registry,
            )

        wins, losses, eval_steps = runner.run_eval(num_games)
        win_rate = wins / num_games
        avg_steps = eval_steps / num_games if num_games > 0 else 0

        print(f"  vs {opp_type}: {wins}/{num_games} wins "
              f"({win_rate:.0%}), avg {avg_steps:.0f} steps/game")

        per_opponent.append(OpponentResult(
            opponent=opp_type,
            wins=wins,
            games=num_games,
            win_rate=win_rate,
            avg_steps=avg_steps,
        ))
        total_wins += wins
        total_games += num_games

    elapsed = time.perf_counter() - start
    overall_rate = total_wins / total_games if total_games > 0 else 0
    print(f"  Overall: {total_wins}/{total_games} wins ({overall_rate:.0%}) "
          f"in {elapsed:.1f}s")

    return EvalResult(
        per_opponent=per_opponent,
        total_wins=total_wins,
        total_games=total_games,
        overall_win_rate=overall_rate,
        elapsed_seconds=elapsed,
    )


def load_and_evaluate(
    *,
    model_path: str | None = None,
    load_latest: bool = False,
    data_cfg: DataConfig | None = None,
    device: str | None = None,
    opponents: str = "random,heuristic,simple",
    eval_games: int = 100,
    num_concurrent: int = 1024,
    num_workers: int = 1,
) -> EvalResult:
    """Load a checkpoint and run evaluation — convenience wrapper for CLI use.

    Handles model loading, device resolution, and dimension validation before
    delegating to :func:`evaluate`.
    """
    if data_cfg is None:
        data_cfg = DataConfig()

    if device is None:
        device = DeviceConfig().simulation_device
    device = DeviceConfig.resolve(device)

    set_verbose(False)
    set_disabled(True)

    # Resolve model path
    if model_path is None or load_latest:
        from src.ppo.ppo_simulate import get_latest_model
        resolved = get_latest_model(data_cfg.models_dir)
        if resolved is None:
            raise FileNotFoundError(
                "No model found. Provide --model or train one first."
            )
        model_path = resolved

    print(f"Model: {model_path}")
    print(f"Device: {device}")

    # Load checkpoint and reconstruct model
    ckpt = load_checkpoint(model_path, map_location=device)
    saved_model_cfg = ckpt.get("config", {}).get("model")
    model_config = (ModelConfig.from_dict(saved_model_cfg)
                    if saved_model_cfg else ModelConfig())

    cards = data_cfg.load_cards()
    registry = data_cfg.build_registry(cards)
    card_names = registry.card_names
    state_dim = get_state_size(card_names)
    action_dim = get_action_space_size(card_names)

    model = PPOActorCritic(
        state_dim, action_dim, len(card_names), model_config=model_config,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return evaluate(
        model,
        data_cfg=data_cfg,
        device=device,
        opponents=opponents,
        eval_games=eval_games,
        num_concurrent=num_concurrent,
        num_workers=num_workers,
    )
