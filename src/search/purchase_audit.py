"""Purchase audit: how much win probability a checkpoint's purchases give away.

The protocol is fixed by docs/decisions/0013-strength-yardstick.md.

1. **Population.** Greedy self-play games of the checkpoint on deals
   ``game_seed(seed, index)``, seat A first on even indices. A decision is
   eligible when the acting player has no pending choice, a purchase is
   legal, the greedy move is a purchase or END_TURN, and there are at least
   two candidates: every legal purchase and END_TURN when it is legal after
   the production suppression.
2. **Sampling.** One decision per game, uniform among that game's eligible
   decisions of both seats. A first pass plays every game to the end and
   counts them; the target comes from a hash of the master seed and game
   index. A second pass replays the same games in the same batches and copies
   each game at its target, which a fingerprint of the observation and legal
   actions confirms; a mismatch, or a missing decision, skips the game. The
   audit uses the first ``decisions`` games, in index order, that yield one.
3. **Candidates** add the one-turn search's choice (:class:`TurnSearch`).
4. **Rollouts.** Every candidate gets ``rollouts`` complete games in the
   vector engine with both seats greedy, from the acting player's
   information set. Rollout ``r`` re-deals the opponent's hidden hand and
   seeds the engine with id ``r``, the same for every candidate. Rollouts in
   the first half form half A and the rest half B.
5. **Measures** (points of win probability): the confident-override gain,
   clear mistakes, and the search's choice; see :func:`decision_measures`.

Finished decisions are appended to ``<output>.journal.jsonl`` so an
interrupted audit resumes.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass

import numpy as np

AUDIT_IMPLEMENTATION = "purchase-audit-v1"
TURN_BINS = ((0, 3), (4, 9), (10, 15), (16, None))
SEAT_NAMES = ("A", "B")
ROLLOUT_STEP_LIMIT = 20_000
GAME_STEP_LIMIT = 20_000


@dataclass(frozen=True)
class AuditConfig:
    model_path: str
    decisions: int = 1000
    rollouts: int = 256
    seed: int = 20260930
    search_budget: int = 64
    search_candidates: int = 8
    search_seed: int = 0
    override_se: float = 2.0
    clear_se: float = 3.0
    rows_per_chunk: int = 150_000
    search_batch: int = 256
    numba_threads: int = 8
    device: str = "cuda"

    def __post_init__(self):
        if self.decisions < 1:
            raise ValueError(f"decisions must be positive, got {self.decisions}")
        if self.rollouts < 2 or self.rollouts % 2:
            raise ValueError(f"rollouts must be even and at least 2, got {self.rollouts}")
        for name in ("search_budget", "search_candidates", "rows_per_chunk", "search_batch", "numba_threads"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")


def target_index(seed: int, game: int, count: int) -> int:
    """1-based eligible decision drawn uniformly for *game*, from a seeded hash."""
    digest = hashlib.sha256(f"purchase-audit:{int(seed)}:{int(game)}".encode("ascii")).digest()
    return 1 + int.from_bytes(digest[:8], "little") % count


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------

def decision_measures(record: dict, *, override_se: float = 2.0, clear_se: float = 3.0,
                      score_on_selection_half: bool = False) -> dict:
    """Per-decision gains of a record, in win probability (not points).

    - ``override``: on one half pick the candidate with the most wins (the
      earlier candidate on ties); if it is not the checkpoint's move and leads
      it by at least ``override_se`` standard errors, its gain is scored on the
      other half, else 0; averaged over both directions. Scoring on the
      selecting half instead (``score_on_selection_half``) shows the selection
      bias the split removes.
    - ``clear``: the largest gain over all rollouts among alternatives at
      least ``clear_se`` paired standard errors better than the move, or None.
    - ``search``: the search's choice minus the move, over all rollouts.
    """
    wins_a = np.asarray(record["wins_a"], dtype=np.float64)
    wins_b = np.asarray(record["wins_b"], dtype=np.float64)
    half = record["rollouts"] // 2
    greedy = record["greedy"]
    rates_a = wins_a / half
    rates_b = wins_b / half
    gains = []
    for selection, evaluation in ((rates_a, rates_b), (rates_b, rates_a)):
        choice = int(np.argmax(selection))
        if choice == greedy:
            gains.append(0.0)
            continue
        se = math.sqrt(max(
            1e-9,
            selection[choice] * (1 - selection[choice]) / half
            + selection[greedy] * (1 - selection[greedy]) / half,
        ))
        if selection[choice] - selection[greedy] < override_se * se:
            gains.append(0.0)
            continue
        scored = selection if score_on_selection_half else evaluation
        gains.append(float(scored[choice] - scored[greedy]))
    totals = wins_a + wins_b
    rollouts = record["rollouts"]
    best_clear = None
    for candidate, paired_sd in enumerate(record["paired_sd"]):
        if candidate == greedy:
            continue
        difference = (totals[candidate] - totals[greedy]) / rollouts
        se = paired_sd / math.sqrt(rollouts)
        if difference > 0 and (se == 0 or difference / se >= clear_se):
            best_clear = difference if best_clear is None else max(best_clear, difference)
    return {
        "override": float(np.mean(gains)),
        "clear": best_clear,
        "search": float((totals[record["search"]] - totals[greedy]) / rollouts),
        "search_differs": record["search"] != greedy,
    }


def _mean_se(values) -> tuple[float | None, float | None]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return None, None
    if len(values) == 1:
        return float(values.mean()), None
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(len(values)))


def summarize_records(records: list[dict], *, override_se: float = 2.0, clear_se: float = 3.0) -> dict:
    """Headline and secondary measures in points, overall and by turn counter."""

    def block(rows):
        measures = [decision_measures(row, override_se=override_se, clear_se=clear_se) for row in rows]
        override_mean, override_se_ = _mean_se([m["override"] for m in measures])
        search_mean, search_se = _mean_se([m["search"] for m in measures])
        clear = [m["clear"] for m in measures if m["clear"] is not None]
        differs = [m["search"] for m in measures if m["search_differs"]]
        to_points = lambda value: None if value is None else 100.0 * value
        return {
            "decisions": len(rows),
            "override_gain_points": to_points(override_mean),
            "override_gain_se_points": to_points(override_se_),
            "override_used_share": (
                float(np.mean([m["override"] != 0.0 for m in measures])) if measures else None
            ),
            "clear_mistake_share": len(clear) / len(rows) if rows else None,
            "clear_mistakes": len(clear),
            "clear_mistake_mean_gain_points": to_points(float(np.mean(clear))) if clear else None,
            "search_gain_points": to_points(search_mean),
            "search_gain_se_points": to_points(search_se),
            "search_differs_share": len(differs) / len(rows) if rows else None,
            "search_gain_where_differs_points": to_points(float(np.mean(differs))) if differs else None,
        }

    by_turn = {}
    for low, high in TURN_BINS:
        label = f"{low}-{high}" if high is not None else f"{low}+"
        rows = [row for row in records if row["turn"] >= low and (high is None or row["turn"] <= high)]
        by_turn[label] = block(rows)
    return {"all": block(records), "by_turn_counter": by_turn}


# ---------------------------------------------------------------------------
# Population: greedy self-play and decision sampling
# ---------------------------------------------------------------------------

class _ExternalAgent:
    def __init__(self, name):
        self.name = name

    def make_decision(self, game_state):
        raise RuntimeError("decisions are made by the purchase audit")


@dataclass
class _Slot:
    index: int
    game: object
    count: int = 0
    fingerprints: list | None = None
    snapshot: dict | None = None
    steps: int = 0


def _new_game(cards, registry, seed: int, index: int):
    import random

    from src.engine.game import Game
    from src.search.uplift import game_seed

    game = Game(cards, card_names=registry.card_names, card_index_map=registry.card_index_map)
    game.rng = random.Random(game_seed(seed, index))
    for name in SEAT_NAMES:
        game.add_player(name, _ExternalAgent(name))
    game.start_game(first_player_name=SEAT_NAMES[0] if index % 2 == 0 else SEAT_NAMES[1])
    return game


def play_greedy_selfplay(evaluator, cards, registry, indices, seed: int, targets: dict | None = None,
                         log_fn=None) -> dict[int, _Slot]:
    """Play greedy self-play games together and find their eligible decisions.

    Without *targets* every eligible decision's fingerprint is recorded. With
    *targets* (game index to 1-based eligible decision), each game is copied
    at that decision. Games are always played to the end, so both passes
    evaluate identical batches.
    """
    from src.encoding.action_context import build_action_context
    from src.encoding.action_encoder import END_TURN_INDEX, get_action_space_size
    from src.encoding.state_encoder import encode_state

    names = registry.card_names
    card_index_map = registry.card_index_map
    action_dim = get_action_space_size(names)
    num_cards = len(names)
    buy_offset = 3 + num_cards
    slots = {
        index: _Slot(index=index, game=_new_game(cards, registry, seed, index),
                     fingerprints=[] if targets is None else None)
        for index in indices
    }
    active = list(slots.values())
    iterations = 0
    while active:
        iterations += 1
        contexts = [
            build_action_context(slot.game, slot.game.current_player, card_index_map, action_dim)
            for slot in active
        ]
        observations = np.stack([
            encode_state(
                slot.game, True, cards=names, card_index_map=card_index_map,
                can_buy=context.can_buy, has_actions=context.has_meaningful, return_numpy=True,
            )
            for slot, context in zip(active, contexts)
        ])
        masks = np.stack([context.mask for context in contexts])
        for row, context in enumerate(contexts):
            if context.has_meaningful:
                masks[row, END_TURN_INDEX] = False
        logits = evaluator.logits(observations, masks)
        remaining = []
        for row, (slot, context) in enumerate(zip(active, contexts)):
            game = slot.game
            player = game.current_player
            action = int(np.argmax(logits[row]))
            legal = np.flatnonzero(masks[row])
            purchases = [int(a) for a in legal if buy_offset <= a < buy_offset + num_cards]
            candidates = purchases + ([END_TURN_INDEX] if masks[row, END_TURN_INDEX] else [])
            if (
                player.get_current_pending_set() is None
                and purchases
                and (buy_offset <= action < buy_offset + num_cards or action == END_TURN_INDEX)
                and len(candidates) >= 2
            ):
                slot.count += 1
                fingerprint = hashlib.sha1(
                    observations[row].tobytes() + masks[row].tobytes()
                ).hexdigest()
                if targets is None:
                    slot.fingerprints.append(fingerprint)
                elif targets.get(slot.index) == slot.count:
                    row_logits = logits[row].astype(np.float64)
                    finite = np.isfinite(row_logits)
                    probabilities = np.zeros_like(row_logits)
                    probabilities[finite] = np.exp(row_logits[finite] - row_logits[finite].max())
                    probabilities /= probabilities.sum()
                    memo = {id(p.agent): p.agent for p in game.players}
                    opponent = game.get_opponent(player)
                    slot.snapshot = {
                        "game": copy.deepcopy(game, memo),
                        "fingerprint": fingerprint,
                        "greedy": action,
                        "candidates": candidates,
                        "probabilities": probabilities,
                        "turn": int(game.stats.total_turns),
                        "trade": int(player.trade),
                        "health": int(player.health),
                        "opponent_health": int(opponent.health),
                        "first": game.first_player_name == player.name,
                    }
            game.apply_decision(context.resolvers[action])
            slot.steps += 1
            if slot.steps > GAME_STEP_LIMIT:
                raise RuntimeError(f"game {slot.index} exceeded {GAME_STEP_LIMIT} decisions")
            if not game.is_game_over:
                remaining.append(slot)
        active = remaining
        if log_fn and iterations % 100 == 0:
            log_fn(f"    step {iterations}: {len(active)} games still running")
    return slots


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------

def rollout_wins(roots_state, roots, decisions: list[int], candidate_lists: list[list[int]],
                 rollouts: int, evaluator, rules, seed: int) -> tuple[list[np.ndarray], list[int]]:
    """Complete greedy games for every candidate of the given decisions.

    Returns, per decision, a boolean array shaped ``[candidates, rollouts]``
    (True when the acting player won) and the number of its games still
    running at the step limit.
    """
    from src.encoding.action_encoder import END_TURN_INDEX
    from src.search.vector_determinization import deal_world_hands, derive_search_seed
    from src.vector_engine.kernels import VectorEngine
    from src.vector_engine.policy import suppress_redundant_end_turn
    from src.vector_engine.state import take_rows

    sizes = np.asarray([len(candidate_lists[d]) * rollouts for d in decisions], dtype=np.int64)
    stops = np.cumsum(sizes)
    starts = stops - sizes
    blocks = len(decisions)
    block_of_row = np.repeat(np.arange(blocks), sizes)
    local = np.arange(int(stops[-1])) - starts[block_of_row]
    worlds = local % rollouts
    root_actions = np.empty(int(stops[-1]), dtype=np.int32)
    for block, decision in enumerate(decisions):
        root_actions[starts[block]:stops[block]] = np.repeat(
            np.asarray(candidate_lists[decision], dtype=np.int32), rollouts
        )
    state = take_rows(roots_state, np.asarray(decisions, dtype=np.int64)[block_of_row])
    viewers = np.asarray([roots[d].viewer for d in decisions], dtype=np.int8)
    deal_world_hands(
        state,
        starts=starts,
        stops=stops,
        viewers=viewers,
        known_hands=np.stack([roots[d].known_hand for d in decisions]),
        base_seeds=[derive_search_seed(seed, roots[d].digest) for d in decisions],
        worlds=worlds,
    )
    viewer_of_row = viewers[block_of_row]
    engine = VectorEngine(state, rules)
    engine.step(root_actions)
    actions = np.full(state.games, END_TURN_INDEX, dtype=np.int32)
    for _ in range(ROLLOUT_STEP_LIMIT):
        active = np.flatnonzero(~state.done)
        if len(active) == 0:
            break
        masks = engine.build_masks()
        observations = engine.materialize_current_player_observations(active)
        greedy = evaluator.greedy(observations, suppress_redundant_end_turn(masks[active]))
        actions[:] = END_TURN_INDEX
        actions[active] = greedy
        engine.step(actions)
    running = ~state.done
    wins = state.winner == viewer_of_row
    return [
        wins[starts[block]:stops[block]].reshape(len(candidate_lists[decision]), rollouts)
        for block, decision in enumerate(decisions)
    ], [int(running[starts[block]:stops[block]].sum()) for block in range(blocks)]


def describe_action(action: int, names: list[str]) -> str:
    from src.encoding.action_encoder import END_TURN_INDEX, get_action_space_layout

    if action == END_TURN_INDEX:
        return "PASS"
    layout = get_action_space_layout(len(names))
    for group, offset, count in layout["card_groups"]:
        if offset <= action < offset + count:
            card = names[action - offset]
            return card if group == "buy_card" else f"{group}:{card}"
    for group, index in layout["global_indices"].items():
        if action == index:
            return group
    return f"action{action}"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: str, payload: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    os.replace(temporary, path)


def sample_decisions(config: AuditConfig, evaluator, cards, registry, log_fn=print) -> tuple[list[dict], dict]:
    """The audit's decisions in game-index order, with skip counts."""
    counts = {"games": 0, "no_eligible_decision": 0, "diverged": 0}
    decisions: list[dict] = []
    next_index = 0
    while len(decisions) < config.decisions:
        needed = config.decisions - len(decisions)
        indices = list(range(next_index, next_index + needed + max(8, needed // 50)))
        next_index = indices[-1] + 1
        log_fn(f"  pass 1: games {indices[0]}-{indices[-1]}")
        first = play_greedy_selfplay(evaluator, cards, registry, indices, config.seed, log_fn=log_fn)
        targets = {
            index: target_index(config.seed, index, slot.count)
            for index, slot in first.items() if slot.count > 0
        }
        log_fn(f"  pass 2: replaying {len(targets)} games to their target decisions")
        second = play_greedy_selfplay(evaluator, cards, registry, indices, config.seed, targets, log_fn=log_fn)
        for index in indices:
            if len(decisions) == config.decisions:
                break
            counts["games"] += 1
            if index not in targets:
                counts["no_eligible_decision"] += 1
                continue
            snapshot = second[index].snapshot
            expected = first[index].fingerprints[targets[index] - 1]
            if snapshot is None or snapshot["fingerprint"] != expected:
                counts["diverged"] += 1
                continue
            snapshot["index"] = index
            snapshot["target"] = targets[index]
            snapshot["eligible"] = first[index].count
            decisions.append(snapshot)
    return decisions, counts


def run_audit(config: AuditConfig, output_json: str, log_fn=print) -> dict:
    """Sample decisions, roll out every candidate, and write records and summary."""
    import torch

    from src.config import DataConfig, DeviceConfig
    from src.encoding.action_context import build_action_context
    from src.encoding.action_encoder import get_action_space_size
    from src.search.turn_search import PolicyValueEvaluator, TurnSearch, TurnSearchConfig
    from src.search.uplift import _load_model
    from src.search.vector_root import SearchRootBuilder
    from src.utils.logger import set_disabled
    from src.vector_engine.kernels import configure_numba_threads
    from src.vector_engine.rules import compile_rules

    started = time.perf_counter()
    set_disabled(True)
    configure_numba_threads(config.numba_threads)
    device = DeviceConfig.resolve(config.device)
    data = DataConfig()
    cards = data.load_cards()
    registry = data.build_registry(cards)
    rules = compile_rules(cards, registry)
    model = _load_model(config.model_path, registry, device)
    evaluator = PolicyValueEvaluator(model, device)
    names = registry.card_names
    action_dim = get_action_space_size(names)

    identity = {
        key: value for key, value in asdict(config).items()
        if key not in ("model_path", "numba_threads", "device")
    }
    identity["model_sha256"] = _file_sha256(config.model_path)
    identity["implementation"] = AUDIT_IMPLEMENTATION
    journal = f"{output_json}.journal.jsonl"
    records: list[dict] = []
    if os.path.exists(journal):
        with open(journal, encoding="utf-8") as fh:
            lines = [json.loads(line) for line in fh if line.strip()]
        if not lines or lines[0].get("identity") != identity:
            raise ValueError(f"existing journal {journal} was written for a different configuration")
        records = lines[1:]
    else:
        os.makedirs(os.path.dirname(os.path.abspath(journal)), exist_ok=True)
        with open(journal, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"identity": identity}) + "\n")
    done = {record["game"] for record in records}
    log_fn(
        f"purchase audit: {config.model_path} decisions={config.decisions} rollouts={config.rollouts} "
        f"seed={config.seed} journaled={len(done)}"
    )

    decisions, counts = sample_decisions(config, evaluator, cards, registry, log_fn=log_fn)
    log_fn(
        f"  sampled {len(decisions)} decisions from {counts['games']} games "
        f"({counts['no_eligible_decision']} without one, {counts['diverged']} diverged) "
        f"in {time.perf_counter() - started:.0f}s"
    )

    search = TurnSearch(
        model, rules, registry.card_index_map, device=device,
        config=TurnSearchConfig(
            budget=config.search_budget, max_candidates=config.search_candidates,
            seed=config.search_seed,
        ),
    )
    counts["search_greedy_mismatch"] = 0
    pending = []
    for start in range(0, len(decisions), config.search_batch):
        batch = decisions[start:start + config.search_batch]
        games = [decision["game"] for decision in batch]
        contexts = [build_action_context(game, game.current_player, registry.card_index_map, action_dim) for game in games]
        results = search.choose_many(games, contexts=contexts)
        for decision, result in zip(batch, results):
            if result.greedy_action != decision["greedy"]:
                counts["search_greedy_mismatch"] += 1
                continue
            candidates = list(decision["candidates"])
            if int(result.action) not in candidates:
                candidates.append(int(result.action))
            decision["candidates"] = candidates
            decision["search"] = candidates.index(int(result.action))
            if decision["index"] not in done:
                pending.append(decision)
    log_fn(
        f"  search choices done ({counts['search_greedy_mismatch']} greedy mismatches); "
        f"{len(pending)} decisions to roll out"
    )

    if pending:
        roots_state, roots = SearchRootBuilder(rules).build([decision["game"] for decision in pending])
        candidate_lists = [decision["candidates"] for decision in pending]
        chunks, current, rows = [], [], 0
        for position, candidates in enumerate(candidate_lists):
            need = len(candidates) * config.rollouts
            if current and rows + need > config.rows_per_chunk:
                chunks.append(current)
                current, rows = [], 0
            current.append(position)
            rows += need
        if current:
            chunks.append(current)
        half = config.rollouts // 2
        for number, chunk in enumerate(chunks, 1):
            chunk_started = time.perf_counter()
            tables, chunk_unfinished = rollout_wins(
                roots_state, roots, chunk, candidate_lists, config.rollouts, evaluator, rules, config.seed,
            )
            new_records = []
            for position, table, still_running in zip(chunk, tables, chunk_unfinished):
                decision = pending[position]
                greedy = decision["candidates"].index(decision["greedy"])
                wins = table.astype(np.int8)
                new_records.append({
                    "game": decision["index"],
                    "target": decision["target"],
                    "eligible": decision["eligible"],
                    "turn": decision["turn"],
                    "trade": decision["trade"],
                    "health": decision["health"],
                    "opponent_health": decision["opponent_health"],
                    "first": decision["first"],
                    "candidates": [describe_action(action, names) for action in decision["candidates"]],
                    "actions": [int(action) for action in decision["candidates"]],
                    "prior": [float(decision["probabilities"][action]) for action in decision["candidates"]],
                    "greedy": greedy,
                    "search": decision["search"],
                    "rollouts": config.rollouts,
                    "wins_a": table[:, :half].sum(axis=1).astype(int).tolist(),
                    "wins_b": table[:, half:].sum(axis=1).astype(int).tolist(),
                    "paired_sd": [float((wins[c] - wins[greedy]).std()) for c in range(len(wins))],
                    "unfinished": still_running,
                })
            with open(journal, "a", encoding="utf-8") as fh:
                for record in new_records:
                    fh.write(json.dumps(record) + "\n")
            records.extend(new_records)
            log_fn(
                f"  chunk {number}/{len(chunks)}: {len(chunk)} decisions, "
                f"unfinished {sum(chunk_unfinished)}, {time.perf_counter() - chunk_started:.0f}s"
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    wanted = {decision["index"] for decision in decisions if "search" in decision}
    final = sorted((record for record in records if record["game"] in wanted), key=lambda r: r["game"])
    counts["unfinished_rollouts"] = sum(record["unfinished"] for record in final)
    summary = summarize_records(final, override_se=config.override_se, clear_se=config.clear_se)
    payload = {
        "protocol": "docs/decisions/0013-strength-yardstick.md",
        "config": asdict(config),
        "identity": identity,
        "counts": counts,
        "summary": summary,
        "elapsed_s": time.perf_counter() - started,
        "records": final,
    }
    _write_json(output_json, payload)
    headline = summary["all"]
    log_fn(
        f"  override gain {headline['override_gain_points']:+.3f} points (SE {headline['override_gain_se_points']:.3f}); "
        f"clear mistakes {headline['clear_mistakes']}/{headline['decisions']}; "
        f"search gain {headline['search_gain_points']:+.3f} (SE {headline['search_gain_se_points']:.3f}); "
        f"{time.perf_counter() - started:.0f}s"
    )
    return payload
