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
    # Matches the physical Star Realms deck (10 Explorer cards in the box).
    # Not unlimited — this is intentional for physical-deck fidelity.
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

    def build_registry(self, cards=None):
        """Build a CardRegistry from loaded cards.

        If *cards* is not provided, loads them from CSV first.
        The registry owns the canonical card_names and card_index_map.
        """
        from src.cards.registry import build_registry
        if cards is None:
            cards = self.load_cards()
        return build_registry(cards, self.starter_card_names)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelConfig:
    """Neural network architecture parameters.

    The model uses per-zone card embeddings with sum pooling and a shared
    feature trunk feeding separate actor/critic heads.
    """
    card_emb_dim: int = 32
    trunk_hidden_sizes: list[int] = field(
        default_factory=lambda: [256, 256]
    )
    actor_head_sizes: list[int] = field(
        default_factory=lambda: [128]
    )
    critic_head_sizes: list[int] = field(
        default_factory=lambda: [128]
    )
    # "mlp" uses a flat MLP actor head; "attention" uses query-key dot-product
    # scoring against dedicated action card embeddings.
    actor_type: str = "mlp"
    # "sum" presence-weighted sum pooling over each zone (original behavior);
    # "attention" replaces the pooling operator with a learned per-zone query
    # that softmax-weights cards by relevance before the weighted sum. Output
    # shape fed into the trunk is identical in both modes.
    pool_type: str = "sum"

    _VALID_ACTOR_TYPES = {"mlp", "attention"}
    _VALID_POOL_TYPES = {"sum", "attention"}

    def __post_init__(self):
        if self.actor_type not in self._VALID_ACTOR_TYPES:
            raise ValueError(
                f"Unknown actor_type {self.actor_type!r}. "
                f"Valid: {', '.join(sorted(self._VALID_ACTOR_TYPES))}"
            )
        if self.pool_type not in self._VALID_POOL_TYPES:
            raise ValueError(
                f"Unknown pool_type {self.pool_type!r}. "
                f"Valid: {', '.join(sorted(self._VALID_POOL_TYPES))}"
            )
        for name in ("trunk_hidden_sizes", "actor_head_sizes", "critic_head_sizes"):
            sizes = getattr(self, name)
            if not isinstance(sizes, list) or len(sizes) == 0:
                raise ValueError(f"{name} must be a non-empty list of ints, got {sizes!r}")
            if any((not isinstance(n, int)) or n <= 0 for n in sizes):
                raise ValueError(f"{name} entries must be positive ints, got {sizes!r}")
        if not isinstance(self.card_emb_dim, int) or self.card_emb_dim <= 0:
            raise ValueError(f"card_emb_dim must be a positive int, got {self.card_emb_dim!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PPOConfig:
    """PPO algorithm hyperparameters."""
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    epochs: int = 4
    batch_size: int = 8192
    entropy_coef: float = 0.025
    grad_clip: float = 0.5
    critic_loss_coef: float = 0.5
    lr_end: float = 1e-5

    def __post_init__(self):
        if self.lr_end > self.lr:
            raise ValueError(
                f"lr_end ({self.lr_end}) must be <= lr ({self.lr})"
            )
        if self.lr_end < 0:
            raise ValueError(
                f"lr_end must be non-negative, got {self.lr_end}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PPOConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RunConfig:
    """Training run topology and schedule."""
    episodes: int = 16000
    updates: int = 200
    num_workers: int = 20
    games_per_worker: int = 16
    num_concurrent: int | None = None
    eval_every: int = 5
    eval_games: int = 3200
    self_play: bool = False
    opponents: str = "random,heuristic,simple"
    self_play_ratio: float = 0.5
    self_play_ratio_start: float = 0.0
    self_play_schedule: str = "linear"
    pfsp_mode: str = "uniform"
    # Path to a checkpoint to resume training from.When set, model weights,
    # optimizer state, LR scheduler state, snapshot pool manifest, and the
    # update counter are restored. ``updates`` is interpreted as "additional
    # updates to run beyond the resumed update count."
    resume: str | None = None
    # Override the cosine LR scheduler horizon (T_max + 1). When None, the
    # scheduler horizon is derived as ``(start_update - 1) + updates`` — i.e.
    # this run's last update. Set this to the FINAL target update of a
    # multi-process training arc (e.g. 200 when chunking 100→200 in 25-step
    # chunks) so each chunk re-pins T_max to the same overall horizon and
    # the cosine LR curve flows smoothly across processes instead of
    # collapsing to the floor at the end of every chunk.
    lr_horizon: int | None = None

    _VALID_SCHEDULES = {"constant", "linear", "cosine"}

    def __post_init__(self):
        if self.self_play_schedule not in self._VALID_SCHEDULES:
            raise ValueError(
                f"Unknown self_play_schedule {self.self_play_schedule!r}. "
                f"Valid: {', '.join(sorted(self._VALID_SCHEDULES))}"
            )
        if not (0.0 <= self.self_play_ratio_start <= 1.0):
            raise ValueError(
                f"self_play_ratio_start must be in [0, 1], got {self.self_play_ratio_start}"
            )
        if not (0.0 <= self.self_play_ratio <= 1.0):
            raise ValueError(
                f"self_play_ratio must be in [0, 1], got {self.self_play_ratio}"
            )
        if self.self_play_ratio_start > self.self_play_ratio:
            raise ValueError(
                f"self_play_ratio_start ({self.self_play_ratio_start}) must be "
                f"<= self_play_ratio ({self.self_play_ratio})"
            )

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

    @staticmethod
    def resolve(device: str) -> str:
        """Return *device* if available, falling back to ``"cpu"``.

        Call this before any ``torch.device()`` / ``.to()`` to safely
        default to CUDA without crashing on CPU-only machines.
        """
        if device in ("cpu", ""):
            return "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                return device
        except Exception:
            pass
        return "cpu"

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


# Current checkpoint schema version. V3 adds optional resume-training
# metadata: optimizer state, LR scheduler state, and a snapshot pool
# manifest. These are weight-compatible with V2 — V2 checkpoints still
# load fine, they just can't be resumed from (only weight-loaded).
CHECKPOINT_VERSION = 3

# Schema versions whose model weights are compatible with the current
# architecture. Loading a checkpoint outside this set raises ValueError.
# Pre-V2 (V0/V1) used a flat embedding + separate actor/critic MLPs and
# is not loadable.
COMPAT_CHECKPOINT_VERSIONS = {2, 3}


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
    optimizer_state_dict: dict | None = None,
    scheduler_state_dict: dict | None = None,
    pool_manifest: dict | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a versioned checkpoint with config metadata.

    Resume-training metadata (optimizer, scheduler, pool_manifest) is
    optional — checkpoints that omit it can still be loaded for
    inference/evaluation but cannot be resumed from.
    """
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
    if optimizer_state_dict is not None:
        checkpoint["optimizer_state_dict"] = optimizer_state_dict
    if scheduler_state_dict is not None:
        checkpoint["scheduler_state_dict"] = scheduler_state_dict
    if pool_manifest is not None:
        checkpoint["pool_manifest"] = pool_manifest
    if extra is not None:
        checkpoint.update(extra)

    torch.save(checkpoint, path)


def load_checkpoint(path: str, map_location=None) -> dict[str, Any]:
    """Load a checkpoint, handling both legacy (raw state_dict) and versioned formats.

    Returns a dict with at least ``model_state_dict``. Versioned checkpoints
    also include ``config``, ``schema_version``, ``timestamp``, etc.

    Raises ``ValueError`` if the checkpoint's schema version is newer than
    the current code supports, or if it predates the current architecture
    (v1 weights are incompatible with v2 model shapes).
    """
    import torch

    data = torch.load(path, map_location=map_location)

    # Legacy format: bare state_dict (keys are parameter names like "actor.0.weight")
    if "model_state_dict" not in data:
        data = {"model_state_dict": data, "schema_version": 0, "config": {}}

    saved_version = data.get("schema_version", 0)
    if saved_version > CHECKPOINT_VERSION:
        raise ValueError(
            f"Checkpoint {path} has schema version {saved_version}, but this "
            f"code only supports up to version {CHECKPOINT_VERSION}. "
            f"Update your code to load this checkpoint."
        )
    if saved_version not in COMPAT_CHECKPOINT_VERSIONS:
        raise ValueError(
            f"Checkpoint {path} has schema version {saved_version}, which is "
            f"incompatible with the current model architecture (compatible "
            f"versions: {sorted(COMPAT_CHECKPOINT_VERSIONS)}). V1 checkpoints "
            f"used a flat embedding + separate actor/critic MLPs; V2+ uses "
            f"per-zone embeddings + shared trunk. Please retrain from scratch."
        )

    return data
