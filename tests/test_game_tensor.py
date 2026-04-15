"""Round-trip tests for game_tensor: to_tensor → from_tensor equivalence."""
import pytest
import random
import torch

from src.config import DataConfig, GameConfig
from src.cards.loader import load_trade_deck_cards
from src.cards.registry import build_registry
from src.engine.game import Game
from src.engine.player import Player
from src.ai.random_agent import RandomAgent
from src.ai.agent import Agent
from src.encoding.game_tensor import (
    game_to_tensor, tensor_to_game, TENSOR_SIZE,
    _pack_effect_bits, _unpack_effect_bits,
)


@pytest.fixture(scope="module")
def game_resources():
    """Load cards and build registry once for all tests."""
    data_cfg = DataConfig()
    cards = data_cfg.load_cards()
    registry = data_cfg.build_registry(cards)
    return cards, registry


def _make_game(cards, registry):
    """Create a fresh 2-player game with random agents."""
    game = Game(
        cards=cards,
        card_names=registry.card_names,
        card_index_map=registry.card_index_map,
    )
    game.add_player("P1", RandomAgent("P1"))
    game.add_player("P2", RandomAgent("P2"))
    game.start_game()
    return game


def _assert_games_equivalent(original: Game, restored: Game):
    """Assert two games have equivalent state."""
    # Game-level
    orig_idx = original.players.index(original.current_player)
    rest_idx = restored.players.index(restored.current_player)
    assert orig_idx == rest_idx, f"current_player mismatch: {orig_idx} vs {rest_idx}"
    assert original.is_game_over == restored.is_game_over
    assert original.stats.total_turns == restored.stats.total_turns

    # Explorer pile
    assert len(original.explorer_pile) == len(restored.explorer_pile)

    # Trade row
    assert len(original.trade_row) == len(restored.trade_row)
    for i in range(len(original.trade_row)):
        assert original.trade_row[i].index == restored.trade_row[i].index, \
            f"trade_row[{i}] mismatch"

    # Trade deck
    assert len(original.trade_deck) == len(restored.trade_deck)
    for i in range(len(original.trade_deck)):
        assert original.trade_deck[i].index == restored.trade_deck[i].index, \
            f"trade_deck[{i}] mismatch"

    # Players
    for pi in range(2):
        op = original.players[pi]
        rp = restored.players[pi]

        # Scalars
        assert op.health == rp.health, f"P{pi} health: {op.health} vs {rp.health}"
        assert op.trade == rp.trade, f"P{pi} trade: {op.trade} vs {rp.trade}"
        assert op.combat == rp.combat, f"P{pi} combat: {op.combat} vs {rp.combat}"
        assert op.cards_drawn == rp.cards_drawn, f"P{pi} cards_drawn mismatch"

        # Zones
        _assert_zone_eq(op.hand, rp.hand, f"P{pi}.hand")
        _assert_zone_eq(op.deck, rp.deck, f"P{pi}.deck")
        _assert_zone_eq(op.discard_pile, rp.discard_pile, f"P{pi}.discard")
        _assert_zone_eq(op.bases, rp.bases, f"P{pi}.bases")

        # Played cards (compare non-base + bases, since bases are aliased)
        orig_played_indices = [c.index for c in op.played_cards]
        rest_played_indices = [c.index for c in rp.played_cards]
        assert sorted(orig_played_indices) == sorted(rest_played_indices), \
            f"P{pi}.played_cards mismatch: {orig_played_indices} vs {rest_played_indices}"

        # Effect applied state for bases
        for i in range(min(len(op.bases), len(rp.bases))):
            orig_bits = _pack_effect_bits(op.bases[i])
            rest_bits = _pack_effect_bits(rp.bases[i])
            if orig_bits != rest_bits:
                assert False, (
                    f"P{pi}.bases[{i}] ({op.bases[i].name}) "
                    f"effect bits differ: {orig_bits:#x} vs {rest_bits:#x}"
                )

        # Effect applied state for non-base played cards
        orig_non_base = [c for c in op.played_cards if c not in op.bases]
        rest_non_base = [c for c in rp.played_cards if c not in rp.bases]
        for i in range(min(len(orig_non_base), len(rest_non_base))):
            orig_bits = _pack_effect_bits(orig_non_base[i])
            rest_bits = _pack_effect_bits(rest_non_base[i])
            if orig_bits != rest_bits:
                assert False, (
                    f"P{pi}.played[{i}] ({orig_non_base[i].name}) "
                    f"effect bits differ: {orig_bits:#x} vs {rest_bits:#x}"
                )

        # Pending actions
        orig_pending = op.get_current_pending_set()
        rest_pending = rp.get_current_pending_set()
        if orig_pending is None:
            assert rest_pending is None, f"P{pi} pending should be None"
        else:
            assert rest_pending is not None, f"P{pi} pending should exist"
            assert orig_pending.decisions_left == rest_pending.decisions_left
            assert orig_pending.mandatory == rest_pending.mandatory
            assert orig_pending.resolved_count == rest_pending.resolved_count
            assert orig_pending.on_complete_draw == rest_pending.on_complete_draw
            assert len(orig_pending.actions) == len(rest_pending.actions), \
                f"P{pi} pending action count: {len(orig_pending.actions)} vs {len(rest_pending.actions)}"


def _assert_zone_eq(orig_cards, rest_cards, label):
    """Assert two card zones have the same cards in the same order."""
    assert len(orig_cards) == len(rest_cards), \
        f"{label} length: {len(orig_cards)} vs {len(rest_cards)}"
    for i in range(len(orig_cards)):
        assert orig_cards[i].index == rest_cards[i].index, \
            f"{label}[{i}]: {orig_cards[i].name}(idx={orig_cards[i].index}) " \
            f"vs {rest_cards[i].name}(idx={rest_cards[i].index})"


# ── Tests ───────────────────────────────────────────────────────────


class TestTensorSize:
    def test_tensor_size_is_stable(self):
        """TENSOR_SIZE should be deterministic."""
        assert TENSOR_SIZE > 0
        assert isinstance(TENSOR_SIZE, int)


class TestEffectBitPacking:
    def test_round_trip_no_applied(self):
        from src.cards.effects import Effect, CardEffectType
        from src.cards.card import Card
        card = Card("Test", 0, 0, [
            Effect(CardEffectType.COMBAT, 3),
            Effect(CardEffectType.TRADE, 2),
        ])
        bits = _pack_effect_bits(card)
        assert bits == 0
        card2 = card.clone()
        _unpack_effect_bits(card2, bits)
        assert not card2.effects[0].applied
        assert not card2.effects[1].applied

    def test_round_trip_some_applied(self):
        from src.cards.effects import Effect, CardEffectType
        from src.cards.card import Card
        card = Card("Test", 0, 0, [
            Effect(CardEffectType.COMBAT, 3),
            Effect(CardEffectType.TRADE, 2),
            Effect(CardEffectType.DRAW, 1),
        ])
        card.effects[0].applied = True
        card.effects[2].applied = True
        bits = _pack_effect_bits(card)
        assert bits == 0b101

        card2 = card.clone()
        _unpack_effect_bits(card2, bits)
        assert card2.effects[0].applied is True
        assert card2.effects[1].applied is False
        assert card2.effects[2].applied is True

    def test_round_trip_nested_effects(self):
        from src.cards.effects import Effect, CardEffectType
        from src.cards.card import Card
        child1 = Effect(CardEffectType.COMBAT, 2)
        child2 = Effect(CardEffectType.TRADE, 3)
        parent = Effect(CardEffectType.PARENT, is_or_effect=True, child_effects=[child1, child2])
        card = Card("Test", 0, 0, [parent])

        child1.applied = True
        bits = _pack_effect_bits(card)

        card2 = card.clone()
        _unpack_effect_bits(card2, bits)
        # DFS order: parent(bit0), child1(bit1), child2(bit2)
        assert card2.effects[0].applied is False  # parent
        assert card2.effects[0].child_effects[0].applied is True  # child1
        assert card2.effects[0].child_effects[1].applied is False  # child2


class TestRoundTrip:
    def test_at_game_start(self, game_resources):
        cards, registry = game_resources
        game = _make_game(cards, registry)

        tensor = game_to_tensor(game)
        assert tensor.shape == (TENSOR_SIZE,)
        assert tensor.dtype == torch.int32

        restored = tensor_to_game(tensor, registry, cards)
        _assert_games_equivalent(game, restored)

    def test_after_several_steps(self, game_resources):
        cards, registry = game_resources
        for n_steps in [5, 10, 20, 50]:
            game = _make_game(cards, registry)
            for _ in range(n_steps):
                if game.is_game_over:
                    break
                game.step()
            if game.is_game_over:
                continue

            tensor = game_to_tensor(game)
            restored = tensor_to_game(tensor, registry, cards)
            _assert_games_equivalent(game, restored)

    def test_many_random_states(self, game_resources):
        """Round-trip 100 random game states."""
        cards, registry = game_resources
        random.seed(42)
        checks = 0
        for _ in range(100):
            game = _make_game(cards, registry)
            steps = random.randint(0, 200)
            for _ in range(steps):
                if game.is_game_over:
                    break
                game.step()
            if game.is_game_over:
                continue

            tensor = game_to_tensor(game)
            restored = tensor_to_game(tensor, registry, cards)
            _assert_games_equivalent(game, restored)
            checks += 1

        assert checks > 30, f"Only verified {checks} states — need more variety"

    def test_game_over_state(self, game_resources):
        """Round-trip a completed game."""
        cards, registry = game_resources
        game = _make_game(cards, registry)
        # Play to completion
        for _ in range(2000):
            if game.is_game_over:
                break
            game.step()
        assert game.is_game_over

        tensor = game_to_tensor(game)
        restored = tensor_to_game(tensor, registry, cards)
        _assert_games_equivalent(game, restored)

    def test_idempotent(self, game_resources):
        """Double round-trip should produce identical tensors."""
        cards, registry = game_resources
        game = _make_game(cards, registry)
        for _ in range(20):
            if game.is_game_over:
                break
            game.step()

        t1 = game_to_tensor(game)
        restored = tensor_to_game(t1, registry, cards)
        t2 = game_to_tensor(restored)
        assert torch.equal(t1, t2), "Double round-trip produced different tensors"
