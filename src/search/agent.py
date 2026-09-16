"""Scalar agent that chooses every decision with turn-bounded search."""
from __future__ import annotations

from dataclasses import dataclass

from src.ai.agent import Agent
from src.encoding.action_context import build_action_context
from src.search.turn_search import SearchResult, TurnSearch


@dataclass
class SearchStatistics:
    """Running totals over one agent's decisions."""

    decisions: int = 0
    forced: int = 0
    deviations: int = 0
    rollouts: int = 0
    elapsed_s: float = 0.0

    def record(self, result: SearchResult) -> None:
        self.decisions += 1
        self.forced += int(result.forced)
        self.deviations += int(result.action != result.greedy_action)
        self.rollouts += result.rollouts
        self.elapsed_s += result.elapsed_s

    def to_dict(self) -> dict:
        return {
            "decisions": self.decisions,
            "forced": self.forced,
            "deviations": self.deviations,
            "rollouts": self.rollouts,
            "elapsed_s": self.elapsed_s,
        }


class SearchAgent(Agent):
    """Play the action a :class:`TurnSearch` selects from public information."""

    def __init__(self, name: str, search: TurnSearch):
        super().__init__(name)
        self.search = search
        self.statistics = SearchStatistics()

    def make_decision(self, game_state):
        context = build_action_context(
            game_state,
            game_state.current_player,
            self.search.card_index_map,
            self.search.action_dim,
        )
        result = self.search.choose(game_state, context)
        action = context.resolvers.get(result.action)
        if action is None:
            raise RuntimeError(
                f"search selected illegal action index {result.action}"
            )
        self.statistics.record(result)
        return action
