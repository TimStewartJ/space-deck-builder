"""OpponentPool: configurable, weighted opponent sampling for PPO training.

Manages a pool of fixed agent types (random, heuristic, simple) and optional
PPO snapshots for self-play. The pool produces a factory callable that samples
a fresh opponent per game.
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


class OpponentPool:
    """Weighted pool of opponent agents for training and evaluation.

    Fixed agents are always available. PPO snapshots are added during
    self-play and stored separately with a configurable pool cap.
    """

    def __init__(
        self,
        opponent_spec: str = "random",
        self_play_ratio: float = 0.5,
        snapshot_cap: int = 10,
    ):
        self.entries: list[tuple[str, float]] = _parse_opponent_spec(opponent_spec)
        self.self_play_ratio = self_play_ratio
        self.snapshot_cap = snapshot_cap

        # PPO snapshot state_dicts stored as (name, state_dict) pairs
        self._snapshots: list[tuple[str, dict]] = []

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
        if len(self._snapshots) > self.snapshot_cap:
            self._snapshots = self._snapshots[-self.snapshot_cap:]

    def make_factory(
        self,
        card_names: list[str],
        device: str = "cpu",
        registry=None,
    ) -> Callable[[], Agent]:
        """Return a factory that samples an opponent per call.

        When self-play snapshots exist, `self_play_ratio` fraction of calls
        return a PPO snapshot opponent; the rest sample from the fixed pool.
        """
        entries = list(self.entries)
        snapshots = list(self._snapshots)
        sp_ratio = self.self_play_ratio

        # Pre-extract weights for fixed pool sampling
        names = [n for n, _ in entries]
        weights = [w for _, w in entries]

        def factory() -> Agent:
            # Decide whether to use a snapshot or fixed agent
            if snapshots and random.random() < sp_ratio:
                snap_name, snap_sd = random.choice(snapshots)
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
