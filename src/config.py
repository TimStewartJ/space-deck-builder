"""Centralized configuration dataclasses for the Star Realms AI project.

All training, model, game, and runtime parameters are defined here as the
single source of truth. Modules receive config objects rather than defining
their own defaults.
"""
from __future__ import annotations

from contextlib import contextmanager
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
    # When True, build per-slot tokens by fusing card_emb + zone_emb with
    # static card features (cost, defense, type flags, faction one-hots) from
    # CardRegistry, then project to emb_dim before pooling. When False, tokens
    # are just card_emb + zone_emb (legacy behavior). Token features remove
    # the burden on the embedding table to memorize static card metadata and
    # are a structural prerequisite for a transformer encoder backbone.
    # Requires a CardRegistry to be passed to PPOActorCritic.
    token_features: bool = False

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
    # Episodes are finite (turn-capped) and reward is a single terminal +/-1,
    # so returns stay bounded in [-1, 1] at any gamma. Undiscounted credit
    # assignment is the correct choice here: at gamma=0.99 a ~100-decision
    # episode discounts the win signal to ~0.37 at episode start, which
    # systematically biases the policy toward immediate tempo over the
    # deck-building investments that decide games. GAE (lam) still does the
    # variance reduction.
    gamma: float = 1.0
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
    engine: str = "object"
    vec_threads: int = 4
    # Limit distinct snapshot models active in one vector rollout. The full
    # pool remains eligible across updates; a PFSP-weighted subset is selected
    # per resident population to avoid fragmenting inference into tiny batches.
    # Zero disables the limit.
    vec_max_active_snapshots: int = 3
    episodes: int = 16000
    updates: int = 200
    num_workers: int = 20
    games_per_worker: int = 16
    num_concurrent: int | None = None
    eval_every: int = 5
    eval_games: int = 3200
    # Evaluation can use a different worker topology from rollout collection.
    # None inherits num_workers.
    eval_num_workers: int | None = None
    self_play: bool = False
    opponents: str = "random,heuristic,simple"
    self_play_ratio: float = 0.5
    self_play_ratio_start: float = 0.0
    self_play_schedule: str = "linear"
    # Absolute final update used to evaluate the self-play schedule. Leave
    # unset for a single-process run; chunked resumes pin every process to the
    # same horizon so linear and cosine curricula remain continuous.
    self_play_horizon: int | None = None
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
    _VALID_ENGINES = {"object", "vec"}

    def __post_init__(self):
        if self.engine not in self._VALID_ENGINES:
            raise ValueError(
                f"Unknown engine {self.engine!r}. "
                f"Valid: {', '.join(sorted(self._VALID_ENGINES))}"
            )
        if self.vec_threads <= 0:
            raise ValueError(
                f"vec_threads must be positive, got {self.vec_threads}"
            )
        if self.vec_max_active_snapshots < 0:
            raise ValueError(
                "vec_max_active_snapshots must be non-negative, got "
                f"{self.vec_max_active_snapshots}"
            )
        if self.eval_num_workers is not None and self.eval_num_workers <= 0:
            raise ValueError(
                "eval_num_workers must be positive when set, got "
                f"{self.eval_num_workers}"
            )
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
        if self.self_play_horizon is not None and self.self_play_horizon <= 0:
            raise ValueError(
                "self_play_horizon must be positive when set, got "
                f"{self.self_play_horizon}"
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


@dataclass(frozen=True)
class TrainingPreset:
    """A named, published training recipe.

    ``train_defaults`` maps ``python -m src train`` argument destinations to
    values. ``--preset`` applies them as parser defaults, so any explicit flag
    still wins. The chunk and evaluation fields describe how
    ``scripts/train_reference.ps1`` runs the arc in fresh processes: interim
    chunks evaluate ``interim_eval_games`` and the final chunk runs the locked
    ``final_eval_games`` gauntlet with exact seat alternation.
    """
    name: str
    description: str
    train_defaults: dict[str, Any]
    chunk_size: int
    interim_eval_games: int
    final_eval_games: int
    eval_num_workers: int
    results: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


TRAINING_PRESETS: dict[str, TrainingPreset] = {
    "reference-selfplay": TrainingPreset(
        name="reference-selfplay",
        description=(
            "Optimized snapshot self-play league: vector engine, "
            "400 updates x 2,000 episodes, fixed opponents plus constant 50% "
            "variance-PFSP self-play with three active snapshots"
        ),
        train_defaults={
            "engine": "vec",
            "vec_threads": 4,
            "vec_max_active_snapshots": 3,
            "episodes": 2000,
            "updates": 400,
            "eval_every": 25,
            "eval_games": 3200,
            "num_workers": 16,
            "opponents": "random,heuristic,simple",
            "self_play": True,
            "self_play_ratio": 0.5,
            "self_play_schedule": "constant",
            "pfsp": "variance",
        },
        chunk_size=25,
        interim_eval_games=3200,
        final_eval_games=15000,
        eval_num_workers=16,
        results="results/optimized_selfplay_baseline.md",
    ),
}


def get_training_preset(name: str) -> TrainingPreset:
    """Return a published training preset by name."""
    try:
        return TRAINING_PRESETS[name]
    except KeyError:
        valid = ", ".join(sorted(TRAINING_PRESETS))
        raise ValueError(
            f"Unknown training preset {name!r}. Valid: {valid}"
        ) from None


# Current checkpoint schema version. V4 adds optional per-card static
# feature tokens (cost, defense, faction, ally faction, type) gated by
# ``ModelConfig.token_features``. V2/V3 checkpoints remain weight-loadable
# as long as their saved ``model_config.token_features`` is False (or
# absent — defaults to False); cross-mode loads (saved-True vs runtime-False
# or vice versa) raise a clear error at load time.
CHECKPOINT_VERSION = 4

# Schema versions whose model weights are compatible with the current
# architecture. Loading a checkpoint outside this set raises ValueError.
# Pre-V2 (V0/V1) used a flat embedding + separate actor/critic MLPs and
# is not loadable.
COMPAT_CHECKPOINT_VERSIONS = {2, 3, 4}


def capture_rng_state() -> dict[str, Any]:
    """Capture every RNG stream that affects reproducible training."""
    import random

    import numpy as np
    import torch

    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG streams captured by :func:`capture_rng_state`."""
    import random

    import numpy as np
    import torch

    required = {"python", "numpy", "torch_cpu"}
    missing = required.difference(state)
    if missing:
        raise ValueError(
            "checkpoint RNG state is missing: "
            + ", ".join(sorted(missing))
        )
    numpy_state = state["numpy"]
    random.setstate(state["python"])
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["state"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"])

    cuda_states = state.get("torch_cuda")
    if cuda_states is None:
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "checkpoint contains accelerator RNG state, but no accelerator "
            "is available for an exact training resume"
        )
    device_count = torch.cuda.device_count()
    if len(cuda_states) != device_count:
        raise RuntimeError(
            "checkpoint contains RNG state for "
            f"{len(cuda_states)} accelerator(s), but {device_count} are "
            "available"
        )
    torch.cuda.set_rng_state_all(cuda_states)


@contextmanager
def preserve_rng_state():
    """Prevent evaluation or diagnostics from advancing training RNG streams."""
    state = capture_rng_state()
    try:
        yield
    finally:
        restore_rng_state(state)


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
    rng_state: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a versioned checkpoint with config metadata.

    Resume-training metadata (optimizer, scheduler, pool manifest, RNG state) is
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
    if rng_state is not None:
        checkpoint["rng_state"] = rng_state
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
