"""Turn-bounded, information-safe search guided by a PPO policy and critic.

At each decision the searcher writes the acting player's vector root from
public counts (:mod:`src.search.vector_root`) and compares the policy's
highest-prior legal actions by truncated Monte Carlo rollouts:

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

:meth:`TurnSearch.choose_many` searches many decisions together. Each
decision keeps the schedule, world ids, and seeds it would have on its own,
while the rows of every decision in a halving round share one vector
population, so a rollout step costs one policy evaluation for all of them.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from src.encoding.action_context import ActionContext, build_action_context
from src.encoding.action_encoder import END_TURN_INDEX
from src.engine.game import Game
from src.search.vector_determinization import deal_world_hands, derive_search_seed
from src.search.vector_root import SearchRoot, SearchRootBuilder
from src.vector_engine.kernels import VectorEngine
from src.vector_engine.policy import suppress_redundant_end_turn
from src.vector_engine.rules import CompiledRules
from src.vector_engine.state import NO_WINNER, VectorState, take_rows

# Identifies how search samples worlds and seeds decisions. Results recorded
# under a different version play different search worlds on the same deals
# and must not be mixed.
SEARCH_IMPLEMENTATION = "batched-count-roots-v1"


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
class SearchTargets:
    """Training targets for one searched decision.

    ``observation`` and ``mask`` are the acting player's policy input and legal
    actions with production END_TURN suppression. ``policy`` is the completed-Q
    improved policy ``softmax(logits + sigma(completedQ))`` over legal actions
    (Danihelka et al., 2022): evaluated candidates use their rollout mean,
    unevaluated actions use the mixed value ``v_mix`` of the critic root value
    and the prior-weighted candidate values, and ``sigma`` is the same
    transform search uses to select actions. A forced decision targets its only
    legal action. ``root_value`` is the critic's estimate at the root.
    """

    observation: np.ndarray
    mask: np.ndarray
    policy: np.ndarray
    root_value: float


@dataclass(frozen=True)
class SearchResult:
    """One searched decision.

    ``fingerprint`` is the public root digest that seeds the search.
    ``elapsed_s`` attributes the wall time of the batch that searched this
    decision: root work is shared equally by the batch's decisions, and
    rollout work is shared in proportion to rollouts. ``targets`` is present
    when :meth:`TurnSearch.choose_many` is asked for training targets.
    """

    action: int
    greedy_action: int
    forced: bool
    rollouts: int
    candidates: tuple[CandidateStatistics, ...]
    fingerprint: str
    elapsed_s: float
    targets: SearchTargets | None = None


@dataclass
class SearchTimings:
    """Cumulative wall time and work counters across :meth:`choose_many` calls."""

    calls: int = 0
    decisions: int = 0
    searched: int = 0
    rounds: int = 0
    rollout_steps: int = 0
    rollout_rows: int = 0
    roots_s: float = 0.0
    root_policy_s: float = 0.0
    populate_s: float = 0.0
    engine_s: float = 0.0
    policy_s: float = 0.0
    values_s: float = 0.0
    total_s: float = 0.0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class PolicyValueEvaluator:
    """Batched masked-logit, greedy-action, and value evaluation for one model."""

    def __init__(self, model: torch.nn.Module, device: torch.device | str):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        tensor = next(
            itertools.chain(self.model.parameters(), self.model.buffers()), None
        )
        self.dtype = (
            tensor.dtype
            if tensor is not None and tensor.is_floating_point()
            else torch.float32
        )
        self._policy_logits = getattr(self.model, "policy_logits", None)

    def _states(self, observations: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(observations)).to(
            self.device, dtype=self.dtype
        )

    def _legal(self, masks: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(masks, dtype=np.bool_)).to(
            self.device
        )

    def _actor(self, states: torch.Tensor) -> torch.Tensor:
        if self._policy_logits is not None:
            logits = self._policy_logits(states)
        else:
            logits, _ = self.model(states)
        return logits.reshape(states.shape[0], -1)

    @torch.no_grad()
    def logits(self, observations: np.ndarray, masks: np.ndarray) -> np.ndarray:
        logits = self._actor(self._states(observations))
        return logits.masked_fill(~self._legal(masks), float("-inf")).cpu().numpy()

    @torch.no_grad()
    def greedy(self, observations: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Highest-probability legal action per row, END_TURN when none is legal."""
        legal = self._legal(masks)
        logits = self._actor(self._states(observations)).masked_fill(
            ~legal, float("-inf")
        )
        actions = logits.argmax(dim=1)
        actions = torch.where(
            legal.any(dim=1), actions, torch.full_like(actions, END_TURN_INDEX)
        )
        return actions.cpu().numpy().astype(np.int32)

    @torch.no_grad()
    def values(self, observations: np.ndarray) -> np.ndarray:
        _, values = self.model(self._states(observations))
        return values.reshape(-1).double().cpu().numpy()


def greedy_actions(masked_logits: np.ndarray) -> np.ndarray:
    """Highest-probability legal action per row, END_TURN when none is legal."""
    actions = np.argmax(masked_logits, axis=1)
    no_legal = ~np.isfinite(masked_logits).any(axis=1)
    actions[no_legal] = END_TURN_INDEX
    return actions


@dataclass
class _Plan:
    """Sequential-halving state for one searched decision."""

    decision: int
    budget: int
    greedy: int
    order: list[int]
    survivors: list[int]
    base_seed: int
    rounds: int
    totals: dict[int, float] = field(default_factory=dict)
    counts: dict[int, int] = field(default_factory=dict)
    used: int = 0
    world_offset: int = 0
    round_index: int = 0
    finished: bool = False


@dataclass(frozen=True)
class _Block:
    """Rows one decision contributes to a halving round's population."""

    plan: _Plan
    actions: tuple[int, ...]
    per_candidate: int
    world_offset: int

    @property
    def rows(self) -> int:
        return len(self.actions) * self.per_candidate


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
        self.roots = SearchRootBuilder(rules)
        self.timings = SearchTimings()

    def choose(
        self, game: Game, context: ActionContext | None = None
    ) -> SearchResult:
        """Search one decision; identical to a one-game :meth:`choose_many`."""
        contexts = None if context is None else [context]
        return self.choose_many([game], contexts=contexts)[0]

    def choose_many(
        self,
        games: Sequence[Game],
        *,
        budgets: Sequence[int] | None = None,
        contexts: Sequence[ActionContext] | None = None,
        return_targets: bool = False,
    ) -> list[SearchResult]:
        """Search the current decision of every game together.

        *budgets* gives each decision its own rollout budget (default: the
        configured budget). *contexts* are the scalar action contexts of the
        games' current decisions; they are computed when omitted, and each
        must match the vector root's legal actions. With *return_targets*,
        every result carries :class:`SearchTargets`; search decisions are the
        same either way.
        """
        started = time.perf_counter()
        count = len(games)
        if count == 0:
            return []
        config = self.config
        if budgets is None:
            budgets = [config.budget] * count
        budgets = [int(budget) for budget in budgets]
        if len(budgets) != count or any(budget < 1 for budget in budgets):
            raise ValueError("one positive budget is required per game")
        if contexts is None:
            contexts = [
                build_action_context(
                    game, game.current_player, self.card_index_map, self.action_dim
                )
                for game in games
            ]
        elif len(contexts) != count:
            raise ValueError("one action context is required per game")

        phase = time.perf_counter()
        roots_state, roots = self.roots.build(games)
        self.timings.roots_s += time.perf_counter() - phase

        phase = time.perf_counter()
        engine = VectorEngine(roots_state, self.rules)
        raw_masks = engine.build_masks().copy()
        for decision, context in enumerate(contexts):
            if not np.array_equal(raw_masks[decision], context.mask):
                raise RuntimeError(
                    "vector root legality differs from the scalar decision"
                )
        masks = suppress_redundant_end_turn(raw_masks)
        observations = engine.materialize_current_player_observations().copy()
        root_logits = self.evaluator.logits(observations, masks)
        root_values = (
            self.evaluator.values(observations) if return_targets else None
        )
        self.timings.root_policy_s += time.perf_counter() - phase
        root_elapsed = time.perf_counter() - started

        forced: dict[int, tuple[int, int]] = {}
        plans: list[_Plan] = []
        for decision in range(count):
            legal = np.flatnonzero(masks[decision])
            if len(legal) == 0:
                raise RuntimeError("search root has no legal action")
            greedy = int(greedy_actions(root_logits[decision : decision + 1])[0])
            if len(legal) == 1:
                forced[decision] = (int(legal[0]), greedy)
                continue
            order = sorted(
                legal.tolist(), key=lambda a, row=root_logits[decision]: (-row[a], a)
            )
            survivors = order[: min(config.max_candidates, budgets[decision], len(order))]
            plans.append(
                _Plan(
                    decision=decision,
                    budget=budgets[decision],
                    greedy=greedy,
                    order=order,
                    survivors=survivors,
                    base_seed=derive_search_seed(config.seed, roots[decision].digest),
                    rounds=max(1, math.ceil(math.log2(len(survivors)))),
                    totals={action: 0.0 for action in survivors},
                    counts={action: 0 for action in survivors},
                )
            )

        active = list(plans)
        while active:
            blocks = []
            for plan in active:
                remaining_rounds = plan.rounds - plan.round_index
                remaining = plan.budget - plan.used
                per_candidate = remaining // (remaining_rounds * len(plan.survivors))
                if per_candidate == 0:
                    if remaining < len(plan.survivors):
                        plan.finished = True
                        continue
                    per_candidate = 1
                blocks.append(
                    _Block(plan, tuple(plan.survivors), per_candidate, plan.world_offset)
                )
            if not blocks:
                break
            for block, values in zip(blocks, self._rollouts(roots_state, roots, blocks)):
                plan = block.plan
                for index, action in enumerate(block.actions):
                    plan.totals[action] += float(values[index].sum())
                    plan.counts[action] += block.per_candidate
                plan.used += block.rows
                plan.world_offset += block.per_candidate
                if plan.round_index < plan.rounds - 1 and len(plan.survivors) > 1:
                    scores = self._scores(
                        plan.survivors, root_logits[plan.decision],
                        plan.totals, plan.counts,
                    )
                    ranked = sorted(plan.survivors, key=lambda a: (-scores[a], a))
                    plan.survivors = ranked[: max(1, math.ceil(len(plan.survivors) / 2))]
                plan.round_index += 1
                if plan.round_index >= plan.rounds:
                    plan.finished = True
            active = [plan for plan in active if not plan.finished]

        elapsed = time.perf_counter() - started
        # Root work is shared equally; rollout work is attributed in proportion
        # to each decision's rollouts, so decisions with different budgets in
        # one call report comparable latencies.
        root_share = root_elapsed / count
        total_rollouts = sum(plan.used for plan in plans)
        rollout_elapsed = max(0.0, elapsed - root_elapsed)
        results: list[SearchResult | None] = [None] * count
        for decision, (action, greedy) in forced.items():
            targets = None
            if return_targets:
                policy = np.zeros(self.action_dim, dtype=np.float32)
                policy[action] = 1.0
                targets = SearchTargets(
                    observation=observations[decision],
                    mask=masks[decision],
                    policy=policy,
                    root_value=float(root_values[decision]),
                )
            results[decision] = SearchResult(
                action=action,
                greedy_action=greedy,
                forced=True,
                rollouts=0,
                candidates=(),
                fingerprint=roots[decision].digest,
                elapsed_s=root_share,
                targets=targets,
            )
        for plan in plans:
            row = root_logits[plan.decision]
            evaluated = [action for action in plan.order if plan.counts.get(action, 0) > 0]
            scores = self._scores(evaluated, row, plan.totals, plan.counts)
            finalists = [action for action in plan.survivors if plan.counts[action] > 0]
            chosen = min(finalists, key=lambda a: (-scores[a], a))
            targets = None
            if return_targets:
                targets = SearchTargets(
                    observation=observations[plan.decision],
                    mask=masks[plan.decision],
                    policy=self.improved_policy(
                        row, masks[plan.decision], plan.totals, plan.counts,
                        float(root_values[plan.decision]),
                    ),
                    root_value=float(root_values[plan.decision]),
                )
            results[plan.decision] = SearchResult(
                action=int(chosen),
                greedy_action=plan.greedy,
                forced=False,
                rollouts=plan.used,
                candidates=tuple(
                    CandidateStatistics(
                        action=int(action),
                        logit=float(row[action]),
                        rollouts=int(plan.counts[action]),
                        value=float(plan.totals[action] / plan.counts[action]),
                        score=float(scores[action]),
                    )
                    for action in evaluated
                ),
                fingerprint=roots[plan.decision].digest,
                elapsed_s=root_share + (
                    rollout_elapsed * plan.used / total_rollouts
                    if total_rollouts else 0.0
                ),
                targets=targets,
            )
        self.timings.calls += 1
        self.timings.decisions += count
        self.timings.searched += len(plans)
        self.timings.total_s += elapsed
        return results

    def improved_policy(
        self,
        logits: np.ndarray,
        mask: np.ndarray,
        totals: dict[int, float],
        counts: dict[int, int],
        root_value: float,
    ) -> np.ndarray:
        """Completed-Q improved policy over the legal actions of one decision."""
        legal = np.asarray(mask, dtype=np.bool_)
        visited = [action for action, visits in counts.items() if visits > 0]
        if not legal.any() or not visited:
            raise ValueError("an improved policy needs legal and evaluated actions")
        logits = np.asarray(logits, dtype=np.float64)
        prior = np.zeros(len(logits), dtype=np.float64)
        finite = logits[legal]
        prior[legal] = np.exp(finite - finite.max())
        prior /= prior.sum()
        q = {action: totals[action] / counts[action] for action in visited}
        visits = float(sum(counts[action] for action in visited))
        visited_prior = float(sum(prior[action] for action in visited))
        weighted_q = float(sum(prior[action] * q[action] for action in visited))
        if visited_prior > 0.0:
            mixed = (root_value + visits * weighted_q / visited_prior) / (1.0 + visits)
        else:
            mixed = (root_value + sum(q.values()) / len(q) * visits) / (1.0 + visits)
        completed = np.full(len(logits), mixed, dtype=np.float64)
        for action in visited:
            completed[action] = q[action]
        scale = (self.config.c_visit + max(counts[action] for action in visited)) * self.config.c_scale
        improved = np.where(legal, logits + scale * (completed + 1.0) / 2.0, -np.inf)
        improved -= improved[legal].max()
        policy = np.where(legal, np.exp(improved), 0.0)
        return (policy / policy.sum()).astype(np.float32)

    def _scores(self, actions, root_logits, totals, counts) -> dict[int, float]:
        max_visits = max(counts[action] for action in actions)
        scale = (self.config.c_visit + max_visits) * self.config.c_scale
        return {
            action: float(root_logits[action])
            + scale * ((totals[action] / counts[action]) + 1.0) / 2.0
            for action in actions
        }

    def _rollouts(
        self,
        roots_state: VectorState,
        roots: Sequence[SearchRoot],
        blocks: Sequence[_Block],
    ) -> list[np.ndarray]:
        """Return searcher returns shaped ``[len(actions), per_candidate]`` per block."""
        timings = self.timings
        phase = time.perf_counter()
        count = len(blocks)
        sizes = np.asarray([block.rows for block in blocks], dtype=np.int64)
        per_candidate = np.asarray([block.per_candidate for block in blocks], dtype=np.int64)
        offsets = np.asarray([block.world_offset for block in blocks], dtype=np.int64)
        decisions = np.asarray([block.plan.decision for block in blocks], dtype=np.int64)
        stops = np.cumsum(sizes)
        starts = stops - sizes
        width = max(len(block.actions) for block in blocks)
        candidates = np.full((count, width), END_TURN_INDEX, dtype=np.int32)
        for index, block in enumerate(blocks):
            candidates[index, : len(block.actions)] = block.actions
        block_of_row = np.repeat(np.arange(count), sizes)
        local = np.arange(int(stops[-1])) - starts[block_of_row]
        root_actions = candidates[block_of_row, local // per_candidate[block_of_row]]
        world_ids = offsets[block_of_row] + local % per_candidate[block_of_row]
        state = take_rows(roots_state, decisions[block_of_row])
        viewers = np.asarray([roots[decision].viewer for decision in decisions], dtype=np.int8)
        deal_world_hands(
            state,
            starts=starts,
            stops=stops,
            viewers=viewers,
            known_hands=np.stack([roots[decision].known_hand for decision in decisions]),
            base_seeds=[block.plan.base_seed for block in blocks],
            worlds=world_ids,
        )
        viewers = viewers[block_of_row]
        engine = VectorEngine(state, self.rules)
        timings.populate_s += time.perf_counter() - phase

        phase = time.perf_counter()
        loop_policy_s = 0.0
        engine.step(root_actions)
        rows = state.games
        returns = np.zeros(rows, dtype=np.float64)
        active = np.ones(rows, dtype=np.bool_)
        leaf_rows: list[np.ndarray] = []
        leaf_observations: list[np.ndarray] = []
        step_actions = np.full(rows, END_TURN_INDEX, dtype=np.int32)
        # The final pass only classifies rows after the last allowed step, so
        # a row that finishes or passes on that step is scored like any other.
        for step_index in range(self.config.max_turn_steps + 1):
            finished = active & state.done
            if finished.any():
                winners = state.winner[finished]
                returns[finished] = np.where(
                    winners == viewers[finished], 1.0,
                    np.where(winners == NO_WINNER, 0.0, -1.0),
                )
                active &= ~finished
            masks = engine.build_masks()
            passed = active & (state.current_player != viewers)
            if passed.any():
                passed_rows = np.flatnonzero(passed)
                leaf_rows.append(passed_rows)
                leaf_observations.append(
                    engine.materialize_current_player_observations(passed_rows).copy()
                )
                state.done[passed_rows] = True
                active &= ~passed
            if not active.any() or step_index == self.config.max_turn_steps:
                break
            active_rows = np.flatnonzero(active)
            observations = engine.materialize_current_player_observations(active_rows)
            effective = suppress_redundant_end_turn(masks[active_rows])
            policy_started = time.perf_counter()
            greedy = self.evaluator.greedy(observations, effective)
            loop_policy_s += time.perf_counter() - policy_started
            step_actions[:] = END_TURN_INDEX
            step_actions[active_rows] = greedy
            engine.step(step_actions)
            timings.rollout_steps += 1
            timings.rollout_rows += len(active_rows)
        capped = np.flatnonzero(active)
        engine_elapsed = time.perf_counter() - phase

        phase = time.perf_counter()
        if len(capped):
            # Rows still inside the searcher's turn at the step cap are scored
            # from the searcher's own perspective.
            returns[capped] = self.evaluator.values(
                engine.materialize_current_player_observations(capped).copy()
            )
        if leaf_rows:
            rows_index = np.concatenate(leaf_rows)
            opponent_values = self.evaluator.values(np.concatenate(leaf_observations))
            returns[rows_index] = -opponent_values
        values_elapsed = time.perf_counter() - phase
        timings.values_s += values_elapsed
        timings.policy_s += loop_policy_s
        timings.engine_s += engine_elapsed - loop_policy_s
        timings.rounds += 1

        out = []
        start = 0
        for block in blocks:
            out.append(
                returns[start : start + block.rows].reshape(
                    len(block.actions), block.per_candidate
                )
            )
            start += block.rows
        return out
