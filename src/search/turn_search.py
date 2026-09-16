"""Turn-bounded, information-safe search guided by a PPO policy and critic.

At each decision the searcher projects the live game into its information set,
converts the canonical public view into a vector population, and compares the
policy's highest-prior legal actions by truncated Monte Carlo rollouts:

1. apply a candidate root action in every row of that candidate;
2. continue the searcher's own turn with the policy's greedy action;
3. stop when the turn passes to the opponent, the game ends, or a step cap is
   reached, and score the leaf from the searcher's perspective. A finished
   game scores its exact outcome. A passed turn is scored with the negated
   critic value of the opponent's first decision, evaluated from the
   opponent's own observation in a sampled world, which is the kind of
   acting-player state the critic was trained on.

During the searcher's turn the opponent takes no decisions, and draws and
market refills come from each row's RNG. Rows therefore differ only in hidden
information that the searcher cannot observe. Root candidates share world ids
(common random numbers), and the rollout budget is allocated by sequential
halving scored with ``logits + sigma(q)`` (Danihelka et al., 2022, evaluated
deterministically without Gumbel noise).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from src.encoding.action_context import build_action_context
from src.encoding.action_encoder import END_TURN_INDEX
from src.engine.game import Game
from src.search.information_set import project_information_set
from src.search.vector_determinization import (
    derive_search_seed,
    known_hand_counts,
    sample_world_hands,
    tile_state,
)
from src.vector_engine.kernels import VectorEngine
from src.vector_engine.policy import suppress_redundant_end_turn
from src.vector_engine.rules import CompiledRules
from src.vector_engine.state import NO_WINNER, from_games


@dataclass(frozen=True)
class TurnSearchConfig:
    """Search budget and selection constants."""

    budget: int = 64
    max_candidates: int = 8
    seed: int = 0
    c_visit: float = 50.0
    c_scale: float = 1.0
    max_turn_steps: int = 256

    def __post_init__(self):
        if self.budget < 1:
            raise ValueError(f"budget must be positive, got {self.budget}")
        if self.max_candidates < 1:
            raise ValueError(
                f"max_candidates must be positive, got {self.max_candidates}"
            )
        if self.max_turn_steps < 1:
            raise ValueError(
                f"max_turn_steps must be positive, got {self.max_turn_steps}"
            )
        if self.c_visit < 0 or self.c_scale < 0:
            raise ValueError("c_visit and c_scale must be non-negative")


@dataclass(frozen=True)
class CandidateStatistics:
    action: int
    logit: float
    rollouts: int
    value: float
    score: float


@dataclass(frozen=True)
class SearchResult:
    action: int
    greedy_action: int
    forced: bool
    rollouts: int
    candidates: tuple[CandidateStatistics, ...]
    fingerprint: str
    elapsed_s: float


class PolicyValueEvaluator:
    """Batched masked-logit and value evaluation for one policy model."""

    def __init__(self, model: torch.nn.Module, device: torch.device | str):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def logits(self, observations: np.ndarray, masks: np.ndarray) -> np.ndarray:
        states = torch.from_numpy(np.ascontiguousarray(observations)).to(self.device)
        legal = torch.from_numpy(np.ascontiguousarray(masks, dtype=np.bool_)).to(self.device)
        logits, _ = self.model(states)
        logits = logits.reshape(states.shape[0], -1).float()
        return logits.masked_fill(~legal, float("-inf")).cpu().numpy()

    @torch.no_grad()
    def values(self, observations: np.ndarray) -> np.ndarray:
        states = torch.from_numpy(np.ascontiguousarray(observations)).to(self.device)
        _, values = self.model(states)
        return values.reshape(-1).float().cpu().numpy()


def greedy_actions(masked_logits: np.ndarray) -> np.ndarray:
    """Highest-probability legal action per row, END_TURN when none is legal."""
    actions = np.argmax(masked_logits, axis=1)
    no_legal = ~np.isfinite(masked_logits).any(axis=1)
    actions[no_legal] = END_TURN_INDEX
    return actions


class TurnSearch:
    """Information-safe turn-bounded search for one policy checkpoint."""

    def __init__(
        self,
        model: torch.nn.Module,
        rules: CompiledRules,
        card_index_map: dict[str, int],
        *,
        device: torch.device | str = "cpu",
        config: TurnSearchConfig | None = None,
    ):
        self.rules = rules
        self.card_index_map = card_index_map
        self.config = config or TurnSearchConfig()
        self.evaluator = PolicyValueEvaluator(model, device)
        self.action_dim = 4 + 10 * rules.num_cards

    def choose(self, game: Game) -> SearchResult:
        started = time.perf_counter()
        config = self.config
        information = project_information_set(game)
        view = information.canonical_game
        viewer = information.viewer_index
        root = from_games([view], self.rules, training_players=[viewer])
        root_engine = VectorEngine(root, self.rules)
        raw_mask = root_engine.build_masks()[0].copy()
        scalar = build_action_context(
            view, view.current_player, self.card_index_map, self.action_dim
        )
        if not np.array_equal(raw_mask, scalar.mask):
            raise RuntimeError(
                "vector root legality differs from the scalar information view"
            )
        mask = suppress_redundant_end_turn(raw_mask[None, :])[0]
        observation = root_engine.materialize_current_player_observations()[:1].copy()
        root_logits = self.evaluator.logits(observation, mask[None, :])[0]
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            raise RuntimeError("search root has no legal action")
        greedy = int(greedy_actions(root_logits[None, :])[0])
        if len(legal) == 1:
            return SearchResult(
                action=int(legal[0]),
                greedy_action=greedy,
                forced=True,
                rollouts=0,
                candidates=(),
                fingerprint=information.fingerprint,
                elapsed_s=time.perf_counter() - started,
            )

        order = sorted(legal.tolist(), key=lambda a: (-root_logits[a], a))
        survivors = order[: min(config.max_candidates, config.budget, len(order))]
        base_seed = derive_search_seed(config.seed, information.fingerprint)
        known_hand = known_hand_counts(information, 1 - viewer, self.rules.num_cards)

        totals = {action: 0.0 for action in survivors}
        counts = {action: 0 for action in survivors}
        used = 0
        world_offset = 0
        rounds = max(1, math.ceil(math.log2(len(survivors))))
        for round_index in range(rounds):
            remaining_rounds = rounds - round_index
            remaining = config.budget - used
            per_candidate = remaining // (remaining_rounds * len(survivors))
            if per_candidate == 0:
                if remaining < len(survivors):
                    break
                per_candidate = 1
            values = self._rollouts(
                root, viewer, known_hand, base_seed,
                survivors, world_offset, per_candidate,
            )
            for index, action in enumerate(survivors):
                totals[action] += float(values[index].sum())
                counts[action] += per_candidate
            used += per_candidate * len(survivors)
            world_offset += per_candidate
            if round_index < rounds - 1 and len(survivors) > 1:
                scores = self._scores(survivors, root_logits, totals, counts)
                ranked = sorted(survivors, key=lambda a: (-scores[a], a))
                survivors = ranked[: max(1, math.ceil(len(survivors) / 2))]

        evaluated = [action for action in order if counts.get(action, 0) > 0]
        scores = self._scores(evaluated, root_logits, totals, counts)
        finalists = [action for action in survivors if counts[action] > 0]
        chosen = min(finalists, key=lambda a: (-scores[a], a))
        candidates = tuple(
            CandidateStatistics(
                action=int(action),
                logit=float(root_logits[action]),
                rollouts=int(counts[action]),
                value=float(totals[action] / counts[action]),
                score=float(scores[action]),
            )
            for action in evaluated
        )
        return SearchResult(
            action=int(chosen),
            greedy_action=greedy,
            forced=False,
            rollouts=used,
            candidates=candidates,
            fingerprint=information.fingerprint,
            elapsed_s=time.perf_counter() - started,
        )

    def _scores(self, actions, root_logits, totals, counts) -> dict[int, float]:
        max_visits = max(counts[action] for action in actions)
        scale = (self.config.c_visit + max_visits) * self.config.c_scale
        return {
            action: float(root_logits[action])
            + scale * ((totals[action] / counts[action]) + 1.0) / 2.0
            for action in actions
        }

    def _rollouts(
        self, root, viewer, known_hand, base_seed, actions, world_offset, worlds,
    ) -> np.ndarray:
        """Return searcher returns shaped ``[len(actions), worlds]``."""
        state = tile_state(root, len(actions) * worlds)
        world_ids = np.tile(
            np.arange(world_offset, world_offset + worlds, dtype=np.int64),
            len(actions),
        )
        sample_world_hands(
            state, viewer=viewer, known_hand=known_hand,
            worlds=world_ids, base_seed=base_seed,
        )
        engine = VectorEngine(state, self.rules)
        rows = state.games
        root_actions = np.repeat(np.asarray(actions, dtype=np.int32), worlds)
        engine.step(root_actions)

        returns = np.zeros(rows, dtype=np.float64)
        active = np.ones(rows, dtype=np.bool_)
        leaf_rows: list[np.ndarray] = []
        leaf_observations: list[np.ndarray] = []
        step_actions = np.full(rows, END_TURN_INDEX, dtype=np.int32)
        for _ in range(self.config.max_turn_steps):
            finished = active & state.done
            if finished.any():
                winners = state.winner[finished]
                returns[finished] = np.where(
                    winners == viewer, 1.0,
                    np.where(winners == NO_WINNER, 0.0, -1.0),
                )
                active &= ~finished
            masks = engine.build_masks()
            passed = active & (state.current_player != viewer)
            if passed.any():
                passed_rows = np.flatnonzero(passed)
                leaf_rows.append(passed_rows)
                leaf_observations.append(
                    engine.materialize_current_player_observations(passed_rows).copy()
                )
                state.done[passed_rows] = True
                active &= ~passed
            if not active.any():
                break
            active_rows = np.flatnonzero(active)
            observations = engine.materialize_current_player_observations(active_rows)
            effective = suppress_redundant_end_turn(masks[active_rows])
            logits = self.evaluator.logits(observations, effective)
            step_actions[:] = END_TURN_INDEX
            step_actions[active_rows] = greedy_actions(logits)
            engine.step(step_actions)
        else:
            if active.any():
                capped = np.flatnonzero(active)
                engine.build_masks()
                returns[capped] = self.evaluator.values(
                    engine.materialize_current_player_observations(capped).copy()
                )
        if leaf_rows:
            rows_index = np.concatenate(leaf_rows)
            opponent_values = self.evaluator.values(np.concatenate(leaf_observations))
            returns[rows_index] = -opponent_values
        return returns.reshape(len(actions), worlds)
