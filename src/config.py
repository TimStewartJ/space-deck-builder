"""Centralized configuration dataclasses for the Star Realms AI project.

All training, model, game, and runtime parameters are defined here as the
single source of truth. Modules receive config objects rather than defining
their own defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class GameConfig:
    """Game rules and constants. Changing these affects state encoding
    normalization, so the encoder must be updated in tandem."""
    starting_health: int = 50
    trade_row_size: int = 5
    turn_cap: int = 1000
    explorer_count: int = 10
    num_scouts: int = 8
    num_vipers: int = 2
    first_player_hand_size: int = 3
    hand_size: int = 5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GameConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DataConfig:
    """Paths and card-set filtering."""
    cards_path: str = "data/cards.csv"
    filter_sets: list[str] = field(default_factory=lambda: ["Core Set"])
    starter_card_names: list[str] = field(
        default_factory=lambda: ["Scout", "Viper", "Explorer"]
    )
    models_dir: str = "models"

    def load_cards(self):
        """Load and filter trade deck cards from CSV."""
        from src.cards.loader import load_trade_deck_cards
        return load_trade_deck_cards(
            self.cards_path,
            filter_sets=self.filter_sets,
            log_cards=False,
        )

    def get_card_names(self, cards) -> list[str]:
        """Build the canonical card name list: unique trade-deck names + starters."""
        names = list(dict.fromkeys(c.name for c in cards))
        for starter in self.starter_card_names:
            if starter not in names:
                names.append(starter)
        return names

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelConfig:
    """Neural network architecture parameters."""
    card_emb_dim: int = 4
    actor_hidden_sizes: list[int] = field(
        default_factory=lambda: [1024, 1024, 512]
    )
    critic_hidden_sizes: list[int] = field(
        default_factory=lambda: [1024, 1024, 512]
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PPOConfig:
    """PPO algorithm hyperparameters."""
    lr: float = 3e-4
    gamma: float = 0.995
    lam: float = 0.99
    clip_eps: float = 0.3
    epochs: int = 4
    batch_size: int = 1024
    entropy_coef: float = 0.025
    grad_clip: float = 0.5
    critic_loss_coef: float = 0.5
    adv_norm: str = "per_episode"  # "per_episode" | "global"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PPOConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RunConfig:
    """Training run topology and schedule."""
    episodes: int = 1024
    updates: int = 4
    num_workers: int = 4
    games_per_worker: int = 16
    num_concurrent: int = 64
    eval_every: int = 5
    eval_games: int = 100
    self_play: bool = False
    opponents: str = "random"
    self_play_ratio: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DeviceConfig:
    """Device placement for training and simulation."""
    main_device: str = "cuda"
    simulation_device: str = "cuda"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DeviceConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SimConfig:
    """Simulation / evaluation run settings."""
    games: int = 50
    player2_random: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SimConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# Current checkpoint schema version
CHECKPOINT_VERSION = 1


def save_checkpoint(
    path: str,
    model_state_dict: dict,
    *,
    ppo_config: PPOConfig | None = None,
    model_config: ModelConfig | None = None,
    run_config: RunConfig | None = None,
    device_config: DeviceConfig | None = None,
    game_config: GameConfig | None = None,
    data_config: DataConfig | None = None,
    update: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a versioned checkpoint with config metadata."""
    import torch
    from datetime import datetime, timezone

    checkpoint = {
        "schema_version": CHECKPOINT_VERSION,
        "model_state_dict": model_state_dict,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {},
    }
    if ppo_config is not None:
        checkpoint["config"]["ppo"] = ppo_config.to_dict()
    if model_config is not None:
        checkpoint["config"]["model"] = model_config.to_dict()
    if run_config is not None:
        checkpoint["config"]["run"] = run_config.to_dict()
    if device_config is not None:
        checkpoint["config"]["device"] = device_config.to_dict()
    if game_config is not None:
        checkpoint["config"]["game"] = game_config.to_dict()
    if data_config is not None:
        checkpoint["config"]["data"] = data_config.to_dict()
    if update is not None:
        checkpoint["update"] = update
    if extra is not None:
        checkpoint.update(extra)

    torch.save(checkpoint, path)


def load_checkpoint(path: str, map_location=None) -> dict[str, Any]:
    """Load a checkpoint, handling both legacy (raw state_dict) and versioned formats.

    Returns a dict with at least ``model_state_dict``. Versioned checkpoints
    also include ``config``, ``schema_version``, ``timestamp``, etc.
    """
    import torch

    data = torch.load(path, map_location=map_location)

    # Legacy format: bare state_dict (keys are parameter names like "actor.0.weight")
    if "model_state_dict" not in data:
        return {"model_state_dict": data, "schema_version": 0, "config": {}}

    return data
