"""Play many scalar games at once with batched policy and search decisions.

Each seat of a game is one of three controllers:

- :class:`SearchSeat` chooses with :meth:`TurnSearch.choose_many` at its own
  rollout budget;
- :class:`PolicySeat` chooses from the policy at a temperature (0 selects the
  highest-probability legal action);
- :class:`AgentSeat` wraps a scalar :class:`Agent` such as the Heuristic.

The runner keeps up to ``batch_size`` games in flight. Scalar agents act
immediately, then every waiting policy decision is evaluated in one forward
pass until each game waits on search or has ended, and finally every waiting
search decision is searched together. A game's outcome depends only on its
seed, its seats, and the model: sampled policy seats draw from a per-game
random generator, and search results do not depend on batch composition.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch

from src.ai.agent import Agent
from src.encoding.action_context import ActionContext, build_action_context
from src.encoding.action_encoder import END_TURN_INDEX, get_action_space_size
from src.encoding.state_encoder import encode_state
from src.engine.game import Game
from src.ppo.action_selection import validate_temperature
from src.search.agent import SearchStatistics
from src.search.turn_search import PolicyValueEvaluator, TurnSearch

STEP_LIMIT = 20_000


@dataclass(frozen=True)
class SearchSeat:
    budget: int

    def __post_init__(self):
        if self.budget < 1:
            raise ValueError(f"search budget must be positive, got {self.budget}")


@dataclass(frozen=True)
class PolicySeat:
    temperature: float = 0.0

    def __post_init__(self):
        validate_temperature(self.temperature)


@dataclass(frozen=True)
class AgentSeat:
    factory: Callable[[str], Agent]


Seat = SearchSeat | PolicySeat | AgentSeat


@dataclass(frozen=True)
class GameSpec:
    """One game: caller key, deal seed, two named seats, and who moves first."""

    key: Any
    seed: int
    players: tuple[tuple[str, Seat], tuple[str, Seat]]
    first_player: str


@dataclass(frozen=True)
class GameOutcome:
    key: Any
    winner: str | None
    turns: int
    decisions: int
    elapsed_s: float
    search: dict[str, SearchStatistics]


@dataclass
class RunnerTimings:
    """Cumulative wall time and work counters for :meth:`BatchedGameRunner.run`."""

    games: int = 0
    agent_decisions: int = 0
    policy_steps: int = 0
    policy_decisions: int = 0
    search_steps: int = 0
    search_decisions: int = 0
    start_s: float = 0.0
    agents_s: float = 0.0
    policy_prepare_s: float = 0.0
    policy_eval_s: float = 0.0
    policy_apply_s: float = 0.0
    search_prepare_s: float = 0.0
    search_s: float = 0.0
    search_apply_s: float = 0.0
    total_s: float = 0.0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class _ExternalAgent(Agent):
    """Placeholder for seats whose decisions the runner makes."""

    def make_decision(self, game_state):
        raise RuntimeError("batched seats are decided by BatchedGameRunner")


@dataclass
class _Slot:
    spec: GameSpec
    game: Game
    seats: dict[str, Seat]
    rng: np.random.Generator
    started: float
    search: dict[str, SearchStatistics] = field(default_factory=dict)
    steps: int = 0

    @property
    def seat(self) -> Seat | None:
        if self.game.is_game_over:
            return None
        return self.seats[self.game.current_player.name]


class BatchedGameRunner:
    """Drive many games concurrently with batched policy and search decisions."""

    def __init__(
        self,
        *,
        cards: Sequence,
        registry,
        model: torch.nn.Module,
        search: TurnSearch | None,
        device: torch.device | str,
        batch_size: int,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.cards = cards
        self.registry = registry
        self.card_names = registry.card_names
        self.card_index_map = registry.card_index_map
        self.action_dim = get_action_space_size(self.card_names)
        self.search = search
        self.evaluator = (
            search.evaluator if search is not None else PolicyValueEvaluator(model, device)
        )
        self.batch_size = batch_size
        self.timings = RunnerTimings()

    def run(
        self,
        specs: Iterable[GameSpec],
        on_finish: Callable[[GameOutcome], None],
    ) -> None:
        started = time.perf_counter()
        try:
            self._run(specs, on_finish)
        finally:
            self.timings.total_s += time.perf_counter() - started

    def _run(
        self,
        specs: Iterable[GameSpec],
        on_finish: Callable[[GameOutcome], None],
    ) -> None:
        pending = iter(specs)
        exhausted = False
        slots: list[_Slot] = []
        while True:
            phase = time.perf_counter()
            while not exhausted and len(slots) < self.batch_size:
                spec = next(pending, None)
                if spec is None:
                    exhausted = True
                    break
                slots.append(self._start(spec))
            self.timings.start_s += time.perf_counter() - phase
            if not slots:
                return
            self._advance_agents(slots)
            slots = self._finish(slots, on_finish)
            while True:
                policy_slots = [slot for slot in slots if isinstance(slot.seat, PolicySeat)]
                if not policy_slots:
                    break
                self._policy_step(policy_slots)
                self._advance_agents(policy_slots)
                slots = self._finish(slots, on_finish)
            search_slots = [slot for slot in slots if isinstance(slot.seat, SearchSeat)]
            if search_slots:
                self._search_step(search_slots)
            slots = self._finish(slots, on_finish)

    def _start(self, spec: GameSpec) -> _Slot:
        names = [name for name, _ in spec.players]
        if len(set(names)) != 2 or spec.first_player not in names:
            raise ValueError(f"game {spec.key!r} needs two distinct named seats")
        game = Game(
            self.cards,
            card_names=self.card_names,
            card_index_map=self.card_index_map,
        )
        game.rng = random.Random(spec.seed)
        seats = {}
        search = {}
        for name, seat in spec.players:
            if isinstance(seat, SearchSeat) and self.search is None:
                raise ValueError("search seats require a TurnSearch")
            agent = seat.factory(name) if isinstance(seat, AgentSeat) else _ExternalAgent(name)
            game.add_player(name, agent)
            seats[name] = seat
            if isinstance(seat, SearchSeat):
                search[name] = SearchStatistics()
        game.start_game(first_player_name=spec.first_player)
        return _Slot(
            spec=spec,
            game=game,
            seats=seats,
            rng=np.random.default_rng(spec.seed),
            started=time.perf_counter(),
            search=search,
        )

    def _apply(self, slot: _Slot, action) -> None:
        slot.game.apply_decision(action)
        slot.steps += 1
        if slot.steps > STEP_LIMIT:
            raise RuntimeError(f"game {slot.spec.key!r} exceeded {STEP_LIMIT} decisions")

    def _advance_agents(self, slots: Iterable[_Slot]) -> None:
        phase = time.perf_counter()
        for slot in slots:
            while isinstance(slot.seat, AgentSeat):
                game = slot.game
                self._apply(slot, game.current_player.make_decision(game))
                self.timings.agent_decisions += 1
        self.timings.agents_s += time.perf_counter() - phase

    def _contexts(self, slots: Sequence[_Slot]) -> list[ActionContext]:
        return [
            build_action_context(
                slot.game, slot.game.current_player, self.card_index_map, self.action_dim
            )
            for slot in slots
        ]

    def _policy_step(self, slots: Sequence[_Slot]) -> None:
        phase = time.perf_counter()
        contexts = self._contexts(slots)
        observations = np.stack([
            encode_state(
                slot.game,
                is_current_player_training=True,
                cards=self.card_names,
                card_index_map=self.card_index_map,
                can_buy=context.can_buy,
                has_actions=context.has_meaningful,
                return_numpy=True,
            )
            for slot, context in zip(slots, contexts)
        ])
        masks = np.stack([context.mask for context in contexts])
        for row, context in enumerate(contexts):
            if context.has_meaningful:
                masks[row, END_TURN_INDEX] = False
        self.timings.policy_prepare_s += time.perf_counter() - phase
        phase = time.perf_counter()
        logits = self.evaluator.logits(observations, masks)
        self.timings.policy_eval_s += time.perf_counter() - phase
        phase = time.perf_counter()
        for row, (slot, context) in enumerate(zip(slots, contexts)):
            temperature = slot.seat.temperature
            if temperature == 0.0:
                index = int(np.argmax(logits[row]))
            else:
                scaled = logits[row].astype(np.float64) / temperature
                legal = np.isfinite(scaled)
                weights = np.zeros_like(scaled)
                weights[legal] = np.exp(scaled[legal] - scaled[legal].max())
                cumulative = np.cumsum(weights)
                index = int(np.searchsorted(
                    cumulative, slot.rng.random() * cumulative[-1], side="right"
                ))
                index = min(index, len(weights) - 1)
            action = context.resolvers.get(index)
            if action is None:
                raise RuntimeError(f"policy selected illegal action index {index}")
            self._apply(slot, action)
        self.timings.policy_apply_s += time.perf_counter() - phase
        self.timings.policy_steps += 1
        self.timings.policy_decisions += len(slots)

    def _search_step(self, slots: Sequence[_Slot]) -> None:
        phase = time.perf_counter()
        contexts = self._contexts(slots)
        self.timings.search_prepare_s += time.perf_counter() - phase
        phase = time.perf_counter()
        results = self.search.choose_many(
            [slot.game for slot in slots],
            budgets=[slot.seat.budget for slot in slots],
            contexts=contexts,
        )
        self.timings.search_s += time.perf_counter() - phase
        phase = time.perf_counter()
        for slot, context, result in zip(slots, contexts, results):
            action = context.resolvers.get(result.action)
            if action is None:
                raise RuntimeError(
                    f"search selected illegal action index {result.action}"
                )
            slot.search[slot.game.current_player.name].record(result)
            self._apply(slot, action)
        self.timings.search_apply_s += time.perf_counter() - phase
        self.timings.search_steps += 1
        self.timings.search_decisions += len(slots)

    def _finish(
        self,
        slots: list[_Slot],
        on_finish: Callable[[GameOutcome], None],
    ) -> list[_Slot]:
        remaining = []
        for slot in slots:
            if not slot.game.is_game_over:
                remaining.append(slot)
                continue
            on_finish(GameOutcome(
                key=slot.spec.key,
                winner=slot.game.get_winner(),
                turns=int(slot.game.stats.total_turns),
                decisions=slot.steps,
                elapsed_s=time.perf_counter() - slot.started,
                search=dict(slot.search),
            ))
            self.timings.games += 1
        return remaining
