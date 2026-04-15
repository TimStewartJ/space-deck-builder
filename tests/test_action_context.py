"""Tests for action_context: parity with legacy get_available_actions + encode_action path."""
import pytest
import numpy as np
from src.config import DataConfig, GameConfig
from src.cards.loader import load_trade_deck_cards
from src.engine.game import Game
from src.engine.actions import get_available_actions, ActionType
from src.encoding.action_encoder import encode_action, get_action_space_size
from src.encoding.action_context import build_action_context
from src.encoding.state_encoder import build_card_index_map
from src.ai.random_agent import RandomAgent


@pytest.fixture(scope="module")
def game_setup():
    """Load cards and build shared state for all tests."""
    data_cfg = DataConfig()
    cards = data_cfg.load_cards()
    registry = data_cfg.build_registry(cards)
    card_names = registry.card_names
    card_index_map = registry.card_index_map
    action_dim = get_action_space_size(card_names)
    return cards, card_names, card_index_map, action_dim


def _play_random_game_to_step(cards, n_steps=0, card_names=None, card_index_map=None):
    """Create a game and advance it n_steps with random actions."""
    game = Game(cards, card_names=card_names, card_index_map=card_index_map)
    game.add_player("P1", RandomAgent("P1"))
    game.add_player("P2", RandomAgent("P2"))
    game.start_game()
    for _ in range(n_steps):
        if game.is_game_over:
            break
        game.step()
    return game


class TestMaskParity:
    """Verify that build_action_context produces the exact same mask
    as the legacy get_available_actions → encode_action path."""

    def _compare_paths(self, game, player, card_names, card_index_map, action_dim):
        """Run both paths and compare masks and resolvers."""
        # Legacy path
        available = get_available_actions(game, player)
        legacy_encoded = [encode_action(a, cards=card_names, card_index_map=card_index_map)
                          for a in available]
        legacy_mask = np.zeros(action_dim, dtype=bool)
        for idx in legacy_encoded:
            legacy_mask[idx] = True

        legacy_meaningful = any(
            a.type in (ActionType.ATTACK_PLAYER, ActionType.PLAY_CARD)
            for a in available
        )
        legacy_can_buy = any(a.type == ActionType.BUY_CARD for a in available)

        # New path
        ctx = build_action_context(game, player, card_index_map, action_dim)

        # Compare masks
        np.testing.assert_array_equal(
            ctx.mask, legacy_mask,
            err_msg="Mask mismatch between legacy and action context paths"
        )
        assert ctx.has_meaningful == legacy_meaningful, \
            f"has_meaningful mismatch: ctx={ctx.has_meaningful}, legacy={legacy_meaningful}"
        assert ctx.can_buy == legacy_can_buy, \
            f"can_buy mismatch: ctx={ctx.can_buy}, legacy={legacy_can_buy}"

        # Verify every masked index has a resolver
        for idx in range(action_dim):
            if ctx.mask[idx]:
                assert idx in ctx.resolvers, f"Missing resolver for masked index {idx}"

        # Verify resolver action types match legacy for each encoded index
        legacy_first_action = {}
        for a, enc in zip(available, legacy_encoded):
            if enc not in legacy_first_action:
                legacy_first_action[enc] = a
        for idx, action in ctx.resolvers.items():
            if idx in legacy_first_action:
                legacy_action = legacy_first_action[idx]
                assert action.type == legacy_action.type, \
                    f"Action type mismatch at index {idx}: ctx={action.type}, legacy={legacy_action.type}"

    def test_parity_at_game_start(self, game_setup):
        cards, card_names, card_index_map, action_dim = game_setup
        game = _play_random_game_to_step(cards, card_names=card_names, card_index_map=card_index_map, n_steps=0)
        player = game.current_player
        self._compare_paths(game, player, card_names, card_index_map, action_dim)

    def test_parity_after_several_steps(self, game_setup):
        cards, card_names, card_index_map, action_dim = game_setup
        for step_count in [5, 10, 20, 50]:
            game = _play_random_game_to_step(cards, card_names=card_names, card_index_map=card_index_map, n_steps=step_count)
            if not game.is_game_over:
                player = game.current_player
                self._compare_paths(game, player, card_names, card_index_map, action_dim)

    def test_parity_across_many_random_states(self, game_setup):
        """Run 50 random games to various points and verify parity at each."""
        cards, card_names, card_index_map, action_dim = game_setup
        import random
        random.seed(42)
        mismatches = 0
        checks = 0
        for _ in range(50):
            steps = random.randint(0, 100)
            game = _play_random_game_to_step(cards, card_names=card_names, card_index_map=card_index_map, n_steps=steps)
            if not game.is_game_over:
                player = game.current_player
                self._compare_paths(game, player, card_names, card_index_map, action_dim)
                checks += 1
        assert checks > 20, f"Only verified {checks} states — need more variety"

    def test_parity_with_pending_actions(self, game_setup):
        """Verify parity when pending action sets are active."""
        cards, card_names, card_index_map, action_dim = game_setup
        import random
        random.seed(123)
        pending_checks = 0
        for _ in range(200):
            steps = random.randint(5, 150)
            game = _play_random_game_to_step(cards, card_names=card_names, card_index_map=card_index_map, n_steps=steps)
            if game.is_game_over:
                continue
            player = game.current_player
            if player.get_current_pending_set() is not None:
                self._compare_paths(game, player, card_names, card_index_map, action_dim)
                pending_checks += 1
        # We should hit at least some pending states in 200 random games
        assert pending_checks > 0, "No pending action states encountered"

    def test_non_current_player_gets_empty_mask(self, game_setup):
        cards, card_names, card_index_map, action_dim = game_setup
        game = _play_random_game_to_step(cards, card_names=card_names, card_index_map=card_index_map, n_steps=0)
        opponent = game.get_opponent(game.current_player)
        ctx = build_action_context(game, opponent, card_index_map, action_dim)
        assert not ctx.mask.any()
        assert len(ctx.resolvers) == 0


class TestActionContextFlags:
    def test_end_turn_always_present(self, game_setup):
        """END_TURN (index 1) should always be in the mask for normal turns."""
        cards, card_names, card_index_map, action_dim = game_setup
        game = _play_random_game_to_step(cards, card_names=card_names, card_index_map=card_index_map, n_steps=0)
        player = game.current_player
        if player.get_current_pending_set() is None:
            ctx = build_action_context(game, player, card_index_map, action_dim)
            assert ctx.mask[1], "END_TURN should always be legal"

    def test_mask_buffer_reuse(self, game_setup):
        """Using a pre-allocated mask buffer should produce same result."""
        cards, card_names, card_index_map, action_dim = game_setup
        game = _play_random_game_to_step(cards, card_names=card_names, card_index_map=card_index_map, n_steps=10)
        if game.is_game_over:
            return
        player = game.current_player

        ctx1 = build_action_context(game, player, card_index_map, action_dim)
        buf = np.zeros(action_dim, dtype=bool)
        ctx2 = build_action_context(game, player, card_index_map, action_dim, mask_buf=buf)
        np.testing.assert_array_equal(ctx1.mask, ctx2.mask)
