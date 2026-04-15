"""OpponentPool: configurable, weighted opponent sampling for PPO training.

Manages a pool of fixed agent types (random, heuristic, simple) and optional
PPO snapshots for self-play. Supports Prioritized Fictitious Self-Play (PFSP)
for adaptive snapshot weighting based on win rate estimates.
"""
from __future__ import annotations

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
    ):
        from src.config import RunConfig
        _defaults = RunConfig()
        if opponent_spec is None:
            opponent_spec = _defaults.opponents
        if self_play_ratio is None:
            self_play_ratio = _defaults.self_play_ratio
        if pfsp_mode is None:
            pfsp_mode = _defaults.pfsp_mode
        if pfsp_mode not in PFSP_MODES:
            raise ValueError(
                f"Unknown PFSP mode {pfsp_mode!r}. "
                f"Valid modes: {', '.join(sorted(PFSP_MODES))}"
            )
        self.entries: list[tuple[str, float]] = _parse_opponent_spec(opponent_spec)
        self.self_play_ratio = self_play_ratio
        self.snapshot_cap = snapshot_cap
        self.pfsp_mode = pfsp_mode
        self._pfsp_ema_alpha = pfsp_ema_alpha

        # PPO snapshot state_dicts stored as (name, state_dict) pairs
        self._snapshots: list[tuple[str, dict]] = []
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

    def add_snapshot(self, state_dict: dict, name: str) -> None:
        """Add a PPO snapshot for self-play. Evicts oldest when at cap."""
        self._snapshots.append((name, state_dict))
        self._snapshot_ema.setdefault(name, 0.5)
        self._snapshot_games.setdefault(name, 0)
        if len(self._snapshots) > self.snapshot_cap:
            evicted_name = self._snapshots[0][0]
            self._snapshot_ema.pop(evicted_name, None)
            self._snapshot_games.pop(evicted_name, None)
            self._snapshots = self._snapshots[-self.snapshot_cap:]

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
        snapshot_names = {name for name, _ in self._snapshots}
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
        snap_names = [n for n, _ in self._snapshots]
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
        snap_names = [n for n, _ in snapshots]
        pfsp_w = self._pfsp_weights(snap_names)

        def factory() -> Agent:
            # Decide whether to use a snapshot or fixed agent
            if snapshots and random.random() < sp_ratio:
                idx = random.choices(range(len(snapshots)), weights=pfsp_w, k=1)[0]
                snap_name, snap_sd = snapshots[idx]
                return _make_ppo_opponent(snap_name, snap_sd, card_names, device,
                                          registry=registry)
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
) -> Agent:
    """Create a fresh PPOAgent and load a snapshot state_dict."""
    from src.ai.ppo_agent import PPOAgent
    opp = PPOAgent(name, card_names, device=device,
                   main_device=device, simulation_device=device,
                   registry=registry)
    opp.model.load_state_dict(state_dict)
    return opp


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
