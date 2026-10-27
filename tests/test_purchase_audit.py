"""Purchase audit: sampling, rollouts, measures, and resume."""
from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

from src.config import DataConfig
from src.encoding.action_encoder import END_TURN_INDEX
from src.search import purchase_audit
from src.search.purchase_audit import (
    AuditConfig,
    decision_measures,
    play_greedy_selfplay,
    rollout_wins,
    run_audit,
    sample_decisions,
    summarize_records,
    target_index,
)
from src.search.turn_search import PolicyValueEvaluator, TurnSearch, TurnSearchConfig
from src.search.uplift import _load_model
from src.search.vector_root import SearchRootBuilder
from src.utils.logger import set_disabled
from src.vector_engine.rules import compile_rules

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT = str(REPO / "results" / "optimized_selfplay_baseline" / "checkpoints" / "selfplay_seed0.pth")


@pytest.fixture(scope="module")
def setup():
    set_disabled(True)
    data = DataConfig()
    cards = data.load_cards()
    registry = data.build_registry(cards)
    rules = compile_rules(cards, registry)
    model = _load_model(CHECKPOINT, registry, "cpu")
    return cards, registry, rules, model, PolicyValueEvaluator(model, "cpu")


@pytest.fixture(scope="module")
def search(setup):
    _, registry, rules, model, _ = setup
    turn_search = TurnSearch(
        model, rules, registry.card_index_map, device="cpu",
        config=TurnSearchConfig(budget=64, max_candidates=8, seed=0),
    )
    return lambda games, contexts: [int(result.action) for result in turn_search.choose_many(games, contexts=contexts)]


@pytest.fixture(scope="module")
def sampled(setup, search):
    cards, registry, _, _, evaluator = setup
    config = AuditConfig(model_path=CHECKPOINT, decisions=3, rollouts=8, device="cpu")
    decisions, counts = sample_decisions(config, evaluator, cards, registry, search, log_fn=lambda message: None)
    return decisions, counts


def test_target_index_covers_every_eligible_decision_uniformly():
    draws = 20000
    counts = np.zeros(5, dtype=int)
    for game in range(draws):
        target = target_index(20260930, game, 5)
        assert 1 <= target <= 5
        counts[target - 1] += 1
    expected = draws / 5
    # 18.47 is the chi-square 0.999 quantile with four degrees of freedom.
    assert ((counts - expected) ** 2 / expected).sum() < 18.47
    assert target_index(7, 3, 1) == 1
    assert target_index(7, 3, 9) == target_index(7, 3, 9)


def _synthetic_records(decisions=2000, candidates=4, rollouts=256, seed=0):
    rng = np.random.default_rng(seed)
    half = rollouts // 2
    return [
        {
            "turn": int(rng.integers(0, 25)),
            "rollouts": rollouts,
            "greedy": 0,
            "search": 1,
            "wins_a": rng.binomial(half, 0.5, candidates).tolist(),
            "wins_b": rng.binomial(half, 0.5, candidates).tolist(),
            "paired_sd": [0.0] + [math.sqrt(0.5)] * (candidates - 1),
        }
        for _ in range(decisions)
    ]


def test_split_half_override_is_unbiased_without_true_differences():
    records = _synthetic_records()
    summary = summarize_records(records)["all"]
    same_half = np.mean([
        decision_measures(record, score_on_selection_half=True)["override"] for record in records
    ])

    assert abs(summary["override_gain_points"]) < 0.15
    assert 100.0 * same_half > 0.5
    assert summary["decisions"] == 2000
    assert set(summarize_records(records)["by_turn_counter"]) == {"0-3", "4-9", "10-15", "16+"}


def test_decision_measures_find_clear_mistakes_and_search_gains():
    record = {
        "turn": 2,
        "rollouts": 256,
        "greedy": 0,
        "search": 2,
        "wins_a": [50, 100, 60],
        "wins_b": [50, 100, 60],
        "paired_sd": [0.0, 0.5, 0.5],
    }
    measures = decision_measures(record)

    assert measures["override"] == pytest.approx(50 / 128)
    assert measures["clear"] == pytest.approx(100 / 256)
    assert measures["search"] == pytest.approx(20 / 256)
    assert measures["search_differs"] is True

    tied = dict(record, wins_a=[50, 50, 50], wins_b=[50, 50, 50])
    assert decision_measures(tied)["override"] == 0.0
    assert decision_measures(tied)["clear"] is None


def test_sampled_decisions_are_eligible_purchases(setup, sampled):
    _, registry, _, _, _ = setup
    decisions, counts = sampled
    buy_offset = 3 + len(registry.card_names)

    assert len(decisions) == 3
    assert counts["games"] >= 3 and counts["diverged"] == 0
    assert [decision["index"] for decision in decisions] == sorted(decision["index"] for decision in decisions)
    for decision in decisions:
        game = decision["game"]
        assert not game.is_game_over
        assert game.current_player.get_current_pending_set() is None
        if len(decision["candidates"]) < 2:
            assert decision["first_pass_search"] not in (None, *decision["candidates"])
        else:
            assert decision["first_pass_search"] is None
        assert decision["greedy"] in decision["candidates"]
        assert any(buy_offset <= action < buy_offset + len(registry.card_names) for action in decision["candidates"])
        assert 1 <= decision["target"] <= decision["eligible"]


def test_single_purchase_decisions_are_eligible_when_the_search_differs(setup):
    cards, registry, _, _, evaluator = setup
    num_cards = len(registry.card_names)
    buy_offset = 3 + num_cards
    asked = []

    def agreeing(games, contexts):
        actions = []
        for game, context in zip(games, contexts):
            legal = np.flatnonzero(context.mask)
            purchases = [int(a) for a in legal if buy_offset <= a < buy_offset + num_cards]
            assert len(purchases) == 1 and not game.is_game_over
            asked.append(purchases[0])
            actions.append(purchases[0])
        return actions

    def differing(games, contexts):
        return [END_TURN_INDEX] * len(games)

    indices = list(range(12))
    agree = play_greedy_selfplay(evaluator, cards, registry, indices, 20260930, search=agreeing)
    differ = play_greedy_selfplay(evaluator, cards, registry, indices, 20260930, search=differing)

    assert asked
    assert sum(slot.count for slot in differ.values()) == sum(slot.count for slot in agree.values()) + len(asked)
    extra = []
    for index in indices:
        assert all(action is None for action in agree[index].search_actions)
        kept = set(agree[index].eligible_steps)
        assert kept <= set(differ[index].eligible_steps)
        for position, step in enumerate(differ[index].eligible_steps):
            if step not in kept:
                assert differ[index].search_actions[position] == END_TURN_INDEX
                extra.append((index, position))
    assert len(extra) == len(asked)

    index, position = extra[0]
    replay = play_greedy_selfplay(
        evaluator, cards, registry, indices, 20260930, {index: differ[index].eligible_steps[position]},
    )
    snapshot = replay[index].snapshot
    assert snapshot["fingerprint"] == differ[index].fingerprints[position]
    assert len(snapshot["candidates"]) == 1
    assert snapshot["greedy"] == snapshot["candidates"][0]

    with pytest.raises(ValueError, match="one-turn search"):
        play_greedy_selfplay(evaluator, cards, registry, indices[:1], 20260930)


def test_rollouts_finish_and_repeat_a_duplicated_candidate(setup, sampled):
    _, _, rules, _, evaluator = setup
    decisions, _ = sampled
    roots_state, roots = SearchRootBuilder(rules).build([d["game"] for d in decisions])
    lists = [[d["greedy"], d["greedy"], *[a for a in d["candidates"] if a != d["greedy"]][:1]] for d in decisions]

    tables, unfinished = rollout_wins(roots_state, roots, [0, 1, 2], lists, 8, evaluator, rules, seed=11)

    assert unfinished == [0, 0, 0]
    for table, candidates in zip(tables, lists):
        assert table.shape == (len(candidates), 8)
        assert np.array_equal(table[0], table[1])


def _hidden_variant(game, seed):
    memo = {id(player.agent): player.agent for player in game.players}
    variant = copy.deepcopy(game, memo)
    rng = random.Random(seed)
    viewer_index = variant.players.index(variant.current_player)
    viewer = variant.players[viewer_index]
    top = len(viewer.ordered_top)
    unknown = viewer.deck[:-top] if top else list(viewer.deck)
    rng.shuffle(unknown)
    viewer.deck = [*unknown, *viewer.ordered_top]
    opponent = variant.players[1 - viewer_index]
    known = {id(card) for card in opponent.public_known_hand}
    unknown_hand = [card for card in opponent.hand if id(card) not in known]
    top = len(opponent.ordered_top)
    pool = [*unknown_hand, *(opponent.deck[:-top] if top else opponent.deck)]
    rng.shuffle(pool)
    opponent.hand = [*opponent.public_known_hand, *pool[: len(unknown_hand)]]
    opponent.deck = [*pool[len(unknown_hand):], *opponent.ordered_top]
    rng.shuffle(variant.trade_deck)
    return variant


def test_rollouts_ignore_the_true_hidden_arrangement(setup, sampled):
    _, _, rules, _, evaluator = setup
    decisions, _ = sampled
    games = [d["game"] for d in decisions]
    variants = [_hidden_variant(game, seed) for seed, game in enumerate(games)]
    lists = [d["candidates"] for d in decisions]

    true_state, true_roots = SearchRootBuilder(rules).build(games)
    variant_state, variant_roots = SearchRootBuilder(rules).build(variants)
    expected, _ = rollout_wins(true_state, true_roots, [0, 1, 2], lists, 8, evaluator, rules, seed=13)
    actual, _ = rollout_wins(variant_state, variant_roots, [0, 1, 2], lists, 8, evaluator, rules, seed=13)

    for left, right in zip(expected, actual):
        assert np.array_equal(left, right)


def test_run_audit_writes_records_and_resumes(tmp_path, monkeypatch):
    config = AuditConfig(
        model_path=CHECKPOINT, decisions=2, rollouts=4, device="cpu",
        numba_threads=2, rows_per_chunk=12,
    )
    output = tmp_path / "audit.json"

    payload = run_audit(config, str(output), log_fn=lambda message: None)

    assert len(payload["records"]) == 2
    assert payload["counts"]["unfinished_rollouts"] == 0
    assert payload["identity"]["implementation"] == "purchase-audit-v2"
    assert payload["identity"]["device"] == "cpu"
    assert {"commit", "dirty"} <= set(payload["identity"])
    for record in payload["records"]:
        assert len(record["wins_a"]) == len(record["candidates"]) == len(record["paired_sd"])
        assert all(0 <= wins <= 2 for wins in record["wins_a"] + record["wins_b"])
        assert 0 <= record["greedy"] < len(record["candidates"])
        assert 0 <= record["search"] < len(record["candidates"])
    assert json.loads(output.read_text(encoding="utf-8"))["records"] == payload["records"]

    def fail(*args, **kwargs):
        raise AssertionError("journaled decisions were rolled out again")

    monkeypatch.setattr(purchase_audit, "rollout_wins", fail)
    resumed = run_audit(config, str(output), log_fn=lambda message: None)
    assert resumed["records"] == payload["records"]

    changed = AuditConfig(
        model_path=CHECKPOINT, decisions=2, rollouts=6, device="cpu",
        numba_threads=2, rows_per_chunk=12,
    )
    with pytest.raises(ValueError, match="different configuration"):
        run_audit(changed, str(output), log_fn=lambda message: None)
