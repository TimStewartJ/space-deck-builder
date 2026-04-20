"""OpponentPool: configurable, weighted opponent sampling for PPO training.

Manages a pool of fixed agent types (random, heuristic, simple) and optional
PPO snapshots for self-play. Supports Prioritized Fictitious Self-Play (PFSP)
for adaptive snapshot weighting based on win rate estimates.
"""
from __future__ import annotations

import math
import random
from typing import Callable

from src.ai.agent import Agent
from src.ai.random_agent import RandomAgent
from src.ai.simple_agent import SimpleAgent
from src.ai.heuristic_agent import HeuristicAgent


# Registry of fixed (non-learned) agent types and their constructors
AGENT_REGISTRY: dict[str, Callable[[str], Agent]] = {
    "random":    lambda name: RandomAgent(name),
    "heuristic": lambda name: HeuristicAgent(name),
    "simple":    lambda name: SimpleAgent(name),
}

VALID_AGENT_NAMES = set(AGENT_REGISTRY.keys())

PFSP_MODES = {"uniform", "hard", "variance"}

# Supported snapshot-pool eviction strategies. ``fifo`` drops the oldest
# snapshot when the pool is over cap (preserves legacy behavior for callers
# that don't track update numbers). ``geometric`` keeps a log-spaced ladder
# of ages — the cap snapshots whose update numbers best match the target
# ages [1, 2, 4, 8, ..., 2^(cap-1)] relative to the most recently-added
# snapshot's update — so the pool always covers a wide band of historical
# selves instead of collapsing to near-clones of the current policy.
EVICTION_MODES = {"fifo", "geometric"}


class OpponentPool:
    """Weighted pool of opponent agents for training and evaluation.

    Fixed agents are always available. PPO snapshots are added during
    self-play and stored separately with a configurable pool cap.

    When pfsp_mode is "hard" or "variance", snapshot selection uses
    Prioritized Fictitious Self-Play instead of uniform random:
      - "hard": prioritizes opponents the agent loses to most
      - "variance": prioritizes opponents near 50% win rate (max learning signal)

    Win rates are tracked as exponential moving averages (EMA) so that
    estimates adapt as the training policy improves.
    """

    def __init__(
        self,
        opponent_spec: str | None = None,
        self_play_ratio: float | None = None,
        snapshot_cap: int = 10,
        pfsp_mode: str | None = None,
        pfsp_ema_alpha: float = 0.3,
        snapshot_eviction: str | None = None,
    ):
        from src.config import RunConfig
        _defaults = RunConfig()
        if opponent_spec is None:
            opponent_spec = _defaults.opponents
        if self_play_ratio is None:
            self_play_ratio = _defaults.self_play_ratio
        if pfsp_mode is None:
            pfsp_mode = _defaults.pfsp_mode
        if snapshot_eviction is None:
            snapshot_eviction = _defaults.snapshot_eviction
        if pfsp_mode not in PFSP_MODES:
            raise ValueError(
                f"Unknown PFSP mode {pfsp_mode!r}. "
                f"Valid modes: {', '.join(sorted(PFSP_MODES))}"
            )
        if snapshot_eviction not in EVICTION_MODES:
            raise ValueError(
                f"Unknown snapshot_eviction {snapshot_eviction!r}. "
                f"Valid modes: {', '.join(sorted(EVICTION_MODES))}"
            )
        self.entries: list[tuple[str, float]] = _parse_opponent_spec(opponent_spec)
        self.self_play_ratio = self_play_ratio
        self.snapshot_cap = snapshot_cap
        self.pfsp_mode = pfsp_mode
        self.snapshot_eviction = snapshot_eviction
        self._pfsp_ema_alpha = pfsp_ema_alpha

        # PPO snapshot state_dicts stored as (name, state_dict, model_config) tuples
        self._snapshots: list[tuple[str, dict, "ModelConfig | None"]] = []
        # On-disk paths for snapshots, keyed by name. Recorded by the trainer
        # after each checkpoint save so that the pool can be reconstructed
        # from a manifest on resume without bloating each checkpoint with
        # full state_dicts. Snapshots without a known path are omitted from
        # the saved manifest (e.g., snapshots loaded from a manifest before
        # the trainer has had a chance to re-save them).
        self._snapshot_paths: dict[str, str] = {}
        # Training update number at which each snapshot was captured. Used by
        # the geometric eviction mode to pick log-spaced ages. Optional: when
        # add_snapshot is called without ``update``, the snapshot participates
        # only in FIFO eviction (it has no known age).
        self._snapshot_updates: dict[str, int] = {}
        # EMA win rates per snapshot for PFSP (new snapshots start at 0.5)
        self._snapshot_ema: dict[str, float] = {}
        # Cumulative games observed per snapshot (for confidence weighting)
        self._snapshot_games: dict[str, int] = {}

    @property
    def has_snapshots(self) -> bool:
        return len(self._snapshots) > 0

    @property
    def opponent_types(self) -> list[str]:
        """Unique fixed agent type names in the pool."""
        return [name for name, _ in self.entries]

    def add_snapshot(self, state_dict: dict, name: str,
                     model_config=None, *, update: int | None = None) -> None:
        """Add a PPO snapshot for self-play. Enforces ``snapshot_cap``.

        Args:
            state_dict: model weights from the snapshot.
            name: identifier for the snapshot (e.g. "PPO_5").
            model_config: ModelConfig used to build the model architecture.
                Required to reconstruct the correct actor head type.
            update: training update number at which this snapshot was saved.
                Required for geometric eviction to position the snapshot on
                the log-spaced age ladder. When omitted (or when any pool
                member lacks an update), eviction falls back to FIFO for
                this operation.
        """
        self._snapshots.append((name, state_dict, model_config))
        if update is not None:
            self._snapshot_updates[name] = update
        self._snapshot_ema.setdefault(name, 0.5)
        self._snapshot_games.setdefault(name, 0)
        if len(self._snapshots) > self.snapshot_cap:
            self._enforce_cap()

    def _enforce_cap(self) -> None:
        """Shrink ``self._snapshots`` back down to ``snapshot_cap`` members.

        Uses geometric eviction when the mode is enabled and every current
        snapshot has a known update number. Falls back to FIFO otherwise
        so callers that don't track updates (tests, older code paths) see
        unchanged behavior.
        """
        if len(self._snapshots) <= self.snapshot_cap:
            return
        have_all_updates = all(
            name in self._snapshot_updates for name, _, _ in self._snapshots
        )
        use_geometric = self.snapshot_eviction == "geometric" and have_all_updates
        if use_geometric:
            keep_indices = self._geometric_keep_indices()
        else:
            keep_indices = list(range(len(self._snapshots)))[-self.snapshot_cap:]
        keep_set = set(keep_indices)
        for idx, (ev_name, _, _) in enumerate(self._snapshots):
            if idx not in keep_set:
                self._drop_snapshot_state(ev_name)
        self._snapshots = [self._snapshots[i] for i in keep_indices]

    def _geometric_keep_indices(self) -> list[int]:
        """Select ``snapshot_cap`` indices along a log-spaced age ladder.

        Target ages are distributed in log space between 0 (newest) and
        the full history available in the current (oversized) pool,
        i.e. ``newest_update - oldest_update``. Each target claims the
        still-available snapshot whose update is closest to
        ``newest_update - target_age``. Endpoints are pinned — the
        newest and oldest pool members are always retained — while the
        intermediate slots spread geometrically so the pool always
        covers a wide band of opponent ages rather than collapsing to
        near-clones of the current policy. When the pool is too small
        to populate all cap slots with distinct snapshots, duplicate
        targets share the selection queue and fewer than cap snapshots
        may be retained; this naturally resolves into a full ladder as
        training progresses and more history accumulates.
        """
        updates = [self._snapshot_updates[name] for name, _, _ in self._snapshots]
        newest_update = max(updates)
        oldest_update = min(updates)
        history = max(0, newest_update - oldest_update)
        target_ages = _log_spaced_ages(self.snapshot_cap, history)
        available = set(range(len(self._snapshots)))
        chosen: list[int] = []
        for target_age in target_ages:
            if not available:
                break
            target_update = newest_update - target_age
            best_idx = min(
                available,
                key=lambda i: (abs(updates[i] - target_update), -updates[i]),
            )
            chosen.append(best_idx)
            available.remove(best_idx)
        return sorted(chosen)

    def _drop_snapshot_state(self, name: str) -> None:
        """Remove all per-snapshot bookkeeping for a name being evicted."""
        self._snapshot_ema.pop(name, None)
        self._snapshot_games.pop(name, None)
        self._snapshot_paths.pop(name, None)
        self._snapshot_updates.pop(name, None)

    def set_snapshot_path(self, name: str, path: str) -> None:
        """Record the on-disk path for a previously-added snapshot.

        Called by the trainer after each per-update ``save_checkpoint`` so
        that the in-memory snapshot can later be rehydrated from disk on
        resume. Snapshots without a recorded path are skipped when building
        the manifest (i.e., they exist in this process only).
        """
        self._snapshot_paths[name] = path

    def to_manifest(self) -> dict:
        """Serialize the snapshot pool to a small dict for checkpointing.

        Includes only snapshots whose on-disk path is known. The manifest
        records the snapshot order (so PFSP eviction order is preserved on
        resume), each snapshot's path, and the PFSP statistics (EMA win
        rate and cumulative games) needed to continue PFSP weighting.
        """
        snapshots = [
            {
                "name": name,
                "path": self._snapshot_paths[name],
                # ``update`` is nullable; absent for snapshots added pre-
                # update-tracking. Included here so geometric eviction can
                # resume its age ladder after a --resume restart.
                "update": self._snapshot_updates.get(name),
            }
            for name, _, _ in self._snapshots
            if name in self._snapshot_paths
        ]
        return {
            "snapshots": snapshots,
            "ema": {n: self._snapshot_ema.get(n, 0.5) for n in (s["name"] for s in snapshots)},
            "games": {n: self._snapshot_games.get(n, 0) for n in (s["name"] for s in snapshots)},
        }

    def load_from_manifest(
        self,
        manifest: dict,
        *,
        models_dir: str | None = None,
        log_fn: Callable[[str], None] | None = None,
    ) -> int:
        """Rehydrate snapshots from a manifest produced by ``to_manifest``.

        Each manifest entry's checkpoint is read from disk and its
        model_state_dict + saved ModelConfig are pushed into the pool.
        Missing or unreadable paths are skipped with a warning so a partial
        pool still functions (degrading gracefully to "fewer self-play
        opponents" rather than aborting training).

        Args:
            manifest: dict produced by ``to_manifest``.
            models_dir: if a manifest path is relative, resolve it against
                this directory (defaults to the path as-is).
            log_fn: optional callback for warnings about missing paths;
                defaults to silent.

        Returns:
            Number of snapshots successfully loaded.
        """
        from src.config import ModelConfig, load_checkpoint
        from pathlib import Path

        warn = log_fn if log_fn is not None else (lambda _msg: None)
        loaded = 0
        for entry in manifest.get("snapshots", []):
            name = entry["name"]
            raw_path = entry["path"]
            path = Path(raw_path)
            if not path.is_absolute() and models_dir is not None and not path.exists():
                path = Path(models_dir) / path.name
            if not path.exists():
                warn(f"Pool manifest: snapshot {name!r} path missing ({raw_path}); skipping.")
                continue
            try:
                ckpt = load_checkpoint(str(path), map_location="cpu")
            except Exception as exc:
                warn(f"Pool manifest: failed to load {raw_path}: {exc}; skipping.")
                continue
            saved_mcfg = ckpt.get("config", {}).get("model")
            mcfg = ModelConfig.from_dict(saved_mcfg) if saved_mcfg else None
            self._snapshots.append((name, ckpt["model_state_dict"], mcfg))
            self._snapshot_paths[name] = str(path)
            manifest_update = entry.get("update")
            if manifest_update is None:
                # Older manifests didn't persist updates. Try recovering
                # from the checkpoint payload itself (the trainer writes
                # ``update`` as a top-level key) before falling back to
                # parsing the conventional ``PPO_<N>`` naming scheme.
                manifest_update = ckpt.get("update")
                if manifest_update is None and name.startswith("PPO_"):
                    suffix = name.split("_", 1)[1]
                    if suffix.isdigit():
                        manifest_update = int(suffix)
            if manifest_update is not None:
                self._snapshot_updates[name] = int(manifest_update)
            ema_map = manifest.get("ema", {})
            games_map = manifest.get("games", {})
            self._snapshot_ema[name] = ema_map.get(name, 0.5)
            self._snapshot_games[name] = games_map.get(name, 0)
            loaded += 1
        # Enforce snapshot cap in case the manifest stored more entries than
        # the current configuration permits. Uses the same eviction strategy
        # as live adds so a resumed pool is compacted consistently.
        self._enforce_cap()
        return loaded

    def update_results(self, results: dict[str, tuple[int, int]]) -> None:
        """Update PFSP EMA win rates from a training batch.

        Alpha is scaled by sample size so that a 1-game result barely moves
        the EMA while a full-sized batch applies the configured alpha.

        Args:
            results: Maps opponent name → (wins_by_training_agent, total_games).
                     Only names matching current snapshots are processed.
        """
        # Games per batch that count as "full confidence" for alpha scaling
        reference_games = 10
        snapshot_names = {name for name, _, _ in self._snapshots}
        base_alpha = self._pfsp_ema_alpha
        for name, (wins, total) in results.items():
            if name not in snapshot_names or total == 0:
                continue
            batch_wr = wins / total
            # Scale alpha by sample count to dampen noisy small samples
            effective_alpha = base_alpha * min(total / reference_games, 1.0)
            old_ema = self._snapshot_ema.get(name, 0.5)
            self._snapshot_ema[name] = effective_alpha * batch_wr + (1 - effective_alpha) * old_ema
            self._snapshot_games[name] = self._snapshot_games.get(name, 0) + total

    def get_pfsp_summary(self) -> dict[str, dict[str, float]]:
        """Return per-snapshot EMA win rates and normalized PFSP weights."""
        snap_names = [n for n, _, _ in self._snapshots]
        if not snap_names:
            return {}
        weights = self._pfsp_weights(snap_names)
        total_w = sum(weights)
        return {
            name: {
                "ema_win_rate": self._snapshot_ema.get(name, 0.5),
                "weight": w / total_w if total_w > 0 else 1.0 / len(snap_names),
            }
            for name, w in zip(snap_names, weights)
        }

    def _pfsp_weights(self, snapshot_names: list[str]) -> list[float]:
        """Compute PFSP sampling weights for the given snapshots.

        Returns uniform weights when pfsp_mode is "uniform". Otherwise:
          - "hard": weight = 1 - win_rate  (favor opponents we lose to)
          - "variance": weight = wr * (1 - wr)  (favor ~50% matchups)

        Low-sample snapshots are blended toward uniform weights using a
        confidence ramp, preventing untested snapshots from dominating
        (especially in variance mode where wr=0.5 yields the max weight).
        """
        if self.pfsp_mode == "uniform" or not snapshot_names:
            return [1.0] * len(snapshot_names)

        # Total games before PFSP reaches full confidence for a snapshot
        warmup_games = 20
        weights = []
        for name in snapshot_names:
            wr = self._snapshot_ema.get(name, 0.5)
            games = self._snapshot_games.get(name, 0)
            confidence = min(games / warmup_games, 1.0)
            if self.pfsp_mode == "hard":
                pfsp_w = 1.0 - wr
            else:  # variance
                pfsp_w = wr * (1.0 - wr)
            # Blend between uniform (1.0) and PFSP based on observation count
            w = confidence * pfsp_w + (1 - confidence) * 1.0
            weights.append(max(w, 1e-6))

        return weights

    def make_factory(
        self,
        card_names: list[str],
        device: str = "cpu",
        registry=None,
    ) -> Callable[[], Agent]:
        """Return a factory that samples an opponent per call.

        When self-play snapshots exist, `self_play_ratio` fraction of calls
        return a PPO snapshot opponent; the rest sample from the fixed pool.
        Snapshot selection uses PFSP weighting when pfsp_mode is not "uniform".
        """
        entries = list(self.entries)
        snapshots = list(self._snapshots)
        sp_ratio = self.self_play_ratio

        # Pre-extract weights for fixed pool sampling
        names = [n for n, _ in entries]
        weights = [w for _, w in entries]

        # PFSP weights frozen at factory creation time for this batch
        snap_names = [n for n, _, _ in snapshots]
        pfsp_w = self._pfsp_weights(snap_names)

        def factory() -> Agent:
            # Decide whether to use a snapshot or fixed agent
            if snapshots and random.random() < sp_ratio:
                idx = random.choices(range(len(snapshots)), weights=pfsp_w, k=1)[0]
                snap_name, snap_sd, snap_mcfg = snapshots[idx]
                return _make_ppo_opponent(snap_name, snap_sd, card_names, device,
                                          registry=registry,
                                          model_config=snap_mcfg)
            # Weighted sample from fixed pool
            chosen = random.choices(names, weights=weights, k=1)[0]
            return AGENT_REGISTRY[chosen](chosen.capitalize())

        return factory

    def make_factory_for_type(
        self,
        agent_type: str,
        card_names: list[str] | None = None,
        device: str = "cpu",
    ) -> Callable[[], Agent]:
        """Return a factory that always produces the given agent type.

        Used for per-opponent evaluation.
        """
        if agent_type in AGENT_REGISTRY:
            def factory() -> Agent:
                return AGENT_REGISTRY[agent_type](agent_type.capitalize())
            return factory
        else:
            raise ValueError(f"Unknown agent type for eval factory: {agent_type!r}")


def _make_ppo_opponent(
    name: str,
    state_dict: dict,
    card_names: list[str],
    device: str,
    registry=None,
    model_config=None,
) -> Agent:
    """Create a fresh PPOAgent and load a snapshot state_dict.

    Args:
        model_config: ModelConfig used to build the snapshot's architecture.
            Must match the actor_type of the snapshot weights.
    """
    from src.ai.ppo_agent import PPOAgent
    opp = PPOAgent(name, card_names, device=device,
                   main_device=device, simulation_device=device,
                   registry=registry, model_config=model_config)
    opp.model.load_state_dict(state_dict)
    return opp


def _log_spaced_ages(count: int, history: int) -> list[int]:
    """Return ``count`` target ages log-spaced between 0 and ``history``.

    The sequence always includes 0 (newest snapshot) and ``history``
    (oldest available snapshot); intermediate entries are distributed
    evenly in log space so the returned ladder grows geometrically.
    When ``history`` is too small to yield ``count`` distinct rounded
    ages the caller sees duplicates (the geometric-eviction path
    naturally consumes them without double-picking any snapshot).
    """
    if count <= 0:
        return []
    if count == 1 or history <= 0:
        return [0] * count
    log_h = math.log(history)
    ages = [0]
    for i in range(1, count):
        age = int(round(math.exp(log_h * i / (count - 1))))
        ages.append(age)
    return ages


def _parse_opponent_spec(spec: str) -> list[tuple[str, float]]:
    """Parse an opponent specification string into (name, weight) pairs.

    Formats:
        "random"                     → [("random", 1.0)]
        "random,heuristic"           → [("random", 0.5), ("heuristic", 0.5)]
        "random:0.6,heuristic:0.4"   → [("random", 0.6), ("heuristic", 0.4)]
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("Opponent spec cannot be empty")

    entries: list[tuple[str, float]] = []
    has_weights = ":" in spec

    for part in spec.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if ":" in part:
            name, weight_str = part.split(":", 1)
            name = name.strip()
            weight = float(weight_str.strip())
            if weight <= 0:
                raise ValueError(f"Weight must be positive, got {weight} for {name!r}")
        else:
            name = part.strip()
            weight = 0.0  # placeholder for equal distribution
        if name not in VALID_AGENT_NAMES:
            raise ValueError(
                f"Unknown opponent type {name!r}. "
                f"Valid types: {', '.join(sorted(VALID_AGENT_NAMES))}"
            )
        if any(n == name for n, _ in entries):
            raise ValueError(f"Duplicate opponent type: {name!r}")
        entries.append((name, weight))

    if not entries:
        raise ValueError("No valid opponent types in spec")

    # Assign equal weights if none were specified
    if not has_weights:
        equal_w = 1.0 / len(entries)
        entries = [(n, equal_w) for n, _ in entries]

    # Normalize weights to sum to 1.0
    total = sum(w for _, w in entries)
    if total <= 0:
        raise ValueError("Total weight must be positive")
    entries = [(n, w / total) for n, w in entries]

    # Sort for canonical checkpoint representation
    entries.sort(key=lambda e: e[0])
    return entries
