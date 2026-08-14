"""Tests for state encoder hidden information handling.

Verifies that the opponent's hand and deck are encoded as a single
'unseen' zone, while the training player retains full zone visibility.
"""
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock

from src.cards.card import Card
from src.cards.effects import Effect, CardEffectType
from src.encoding.state_encoder import (
    encode_state, encode_player_into, encode_opponent_into,
    encode_card_presence, get_state_size, build_card_index_map,
)
from src.encoding.state_utils import unpack_state
from src.encoding.action_encoder import get_action_space_size


def _make_card(name: str, index: int = 0) -> Card:
    return Card(name=name, index=index, cost=1, effects=[])


# Shared fixtures
CARD_NAMES = ["Scout", "Viper", "Explorer", "Blob Fighter", "Trade Pod"]
CARD_INDEX_MAP = build_card_index_map(CARD_NAMES)
NUM_CARDS = len(CARD_NAMES)


def _make_player(
    hand_names=None, deck_names=None, discard_names=None, base_names=None,
    played_names=None,
    trade=0, combat=0, health=50,
):
    """Build a mock Player with the specified card zones."""
    p = MagicMock()
    p.trade = trade
    p.combat = combat
    p.health = health
    p.hand = [_make_card(n, CARD_INDEX_MAP[n]) for n in (hand_names or [])]
    p.deck = [_make_card(n, CARD_INDEX_MAP[n]) for n in (deck_names or [])]
    p.discard_pile = [_make_card(n, CARD_INDEX_MAP[n]) for n in (discard_names or [])]
    p.bases = [_make_card(n, CARD_INDEX_MAP[n]) for n in (base_names or [])]
    p.played_cards = [_make_card(n, CARD_INDEX_MAP[n]) for n in (played_names or [])]
    return p


class TestGetStateSize:
    def test_asymmetric_size(self):
        """Opponent encoding is smaller (4 zones + 6 scalars vs 5 zones + 5 scalars)."""
        size = get_state_size(CARD_NAMES)
        n = NUM_CARDS
        expected = (
            4                   # flags
            + n                 # trade row
            + (5 + n * 5)       # training player: 5 scalars + 5 zones
            + (6 + n * 4)       # opponent: 6 scalars + 4 zones
        )
        assert size == expected

    def test_opponent_smaller_than_training_player(self):
        """Hidden information means the opponent gets one fewer zone."""
        n = NUM_CARDS
        training_player = 5 + n * 5
        opponent = 6 + n * 4
        assert opponent < training_player
        # one fewer zone, one extra scalar (hand_size)
        assert opponent == training_player - n + 1


class TestEncodeOpponentInto:
    def test_hand_deck_merged(self):
        """Hand and deck cards appear in the same 'unseen' zone region."""
        player = _make_player(
            hand_names=["Scout"],
            deck_names=["Viper"],
        )
        out = np.zeros(6 + NUM_CARDS * 4, dtype=np.float32)
        encode_opponent_into(player, NUM_CARDS, CARD_INDEX_MAP, out, 0)

        unseen_start = 6
        unseen_end = 6 + NUM_CARDS
        unseen = out[unseen_start:unseen_end]

        # Both Scout (idx 0) and Viper (idx 1) should be present
        assert unseen[CARD_INDEX_MAP["Scout"]] > 0
        assert unseen[CARD_INDEX_MAP["Viper"]] > 0
        # Others should be zero
        assert unseen[CARD_INDEX_MAP["Explorer"]] == 0

    def test_split_invariance(self):
        """Same total cards, different hand/deck split → identical unseen vector."""
        player_a = _make_player(hand_names=["Scout", "Viper"], deck_names=["Explorer"])
        player_b = _make_player(hand_names=["Explorer"], deck_names=["Scout", "Viper"])

        size = 6 + NUM_CARDS * 4
        out_a = np.zeros(size, dtype=np.float32)
        out_b = np.zeros(size, dtype=np.float32)

        encode_opponent_into(player_a, NUM_CARDS, CARD_INDEX_MAP, out_a, 0)
        encode_opponent_into(player_b, NUM_CARDS, CARD_INDEX_MAP, out_b, 0)

        unseen_start = 6
        unseen_end = 6 + NUM_CARDS
        np.testing.assert_array_equal(
            out_a[unseen_start:unseen_end],
            out_b[unseen_start:unseen_end],
        )

    def test_hand_size_differs(self):
        """Even with same unseen contents, hand_size scalar differs by split."""
        player_a = _make_player(hand_names=["Scout", "Viper"], deck_names=["Explorer"])
        player_b = _make_player(hand_names=["Explorer"], deck_names=["Scout", "Viper"])

        size = 6 + NUM_CARDS * 4
        out_a = np.zeros(size, dtype=np.float32)
        out_b = np.zeros(size, dtype=np.float32)

        encode_opponent_into(player_a, NUM_CARDS, CARD_INDEX_MAP, out_a, 0)
        encode_opponent_into(player_b, NUM_CARDS, CARD_INDEX_MAP, out_b, 0)

        # hand_size is at index 4 (trade, combat, health, deck_size, hand_size, discard_size)
        assert out_a[4] == pytest.approx(2 / 10.0)   # 2 cards in hand
        assert out_b[4] == pytest.approx(1 / 10.0)   # 1 card in hand

    def test_scalars_encoded(self):
        """All 6 opponent scalars are correctly written."""
        player = _make_player(
            trade=10, combat=5, health=30,
            hand_names=["Scout", "Viper"],
            deck_names=["Explorer"],
            discard_names=["Blob Fighter"],
        )
        out = np.zeros(6 + NUM_CARDS * 4, dtype=np.float32)
        encode_opponent_into(player, NUM_CARDS, CARD_INDEX_MAP, out, 0)

        assert out[0] == pytest.approx(10 / 100.0)   # trade
        assert out[1] == pytest.approx(5 / 100.0)    # combat
        assert out[2] == pytest.approx(30 / 100.0)   # health
        assert out[3] == pytest.approx(1 / 40.0)     # deck_size (1 card in deck)
        assert out[4] == pytest.approx(2 / 10.0)     # hand_size (2 cards in hand)
        assert out[5] == pytest.approx(1 / 40.0)     # discard_size


class TestEncodeDecode:
    """Training player retains separate hand/deck; opponent gets merged unseen."""

    def test_training_player_has_separate_hand_deck(self):
        """unpack_state should produce separate train_hand and train_deck."""
        keys = self._unpack_keys()
        assert 'train_hand' in keys
        assert 'train_deck' in keys

    def test_opponent_has_unseen_not_hand_deck(self):
        """unpack_state should produce opp_unseen, not opp_hand or opp_deck."""
        keys = self._unpack_keys()
        assert 'opp_unseen' in keys
        assert 'opp_hand' not in keys
        assert 'opp_deck' not in keys

    def test_opponent_res_is_6_wide(self):
        """Opponent resources should be 6-wide (includes hand_size)."""
        pieces = self._unpack_pieces()
        assert pieces['opp_res'].shape[-1] == 6

    def test_training_res_is_5_wide(self):
        """Training player resources should be 5-wide."""
        pieces = self._unpack_pieces()
        assert pieces['train_res'].shape[-1] == 5

    def test_unpack_consumes_full_vector(self):
        """unpack_state assertion should pass — no leftover elements."""
        state_size = get_state_size(CARD_NAMES)
        x = torch.zeros(state_size)
        # Should not raise
        unpack_state(x, NUM_CARDS, get_action_space_size(CARD_NAMES))

    def test_unpack_rejects_wrong_size(self):
        """unpack_state should fail on a vector sized for a different layout."""
        wrong_size = get_state_size(CARD_NAMES) - NUM_CARDS
        x = torch.zeros(wrong_size)
        with pytest.raises(AssertionError, match="State layout mismatch"):
            unpack_state(x, NUM_CARDS, get_action_space_size(CARD_NAMES))

    def test_both_players_have_played_zone(self):
        """Cards played this turn must be observable for both players."""
        keys = self._unpack_keys()
        assert 'train_played' in keys
        assert 'opp_played' in keys

    def _unpack_keys(self):
        state_size = get_state_size(CARD_NAMES)
        x = torch.zeros(state_size)
        pieces, _ = unpack_state(x, NUM_CARDS, get_action_space_size(CARD_NAMES))
        return set(pieces.keys())

    def _unpack_pieces(self):
        state_size = get_state_size(CARD_NAMES)
        x = torch.zeros(state_size)
        pieces, _ = unpack_state(x, NUM_CARDS, get_action_space_size(CARD_NAMES))
        return pieces


class TestPlayedCardsVisibility:
    """Cards played this turn must be observable.

    Ally abilities and the set of legal actions are derived from
    ``played_cards + bases``. A ship played this turn leaves ``hand`` and
    lives only in ``played_cards`` until end of turn, so omitting that zone
    makes the policy blind to the state its own action mask depends on.
    """

    def test_training_player_played_zone_is_populated(self):
        player = _make_player(hand_names=["Scout"], played_names=["Blob Fighter"])
        out = np.zeros(5 + NUM_CARDS * 5, dtype=np.float32)
        encode_player_into(player, NUM_CARDS, CARD_INDEX_MAP, out, 0)

        played = out[5 + NUM_CARDS: 5 + NUM_CARDS * 2]
        assert played[CARD_INDEX_MAP["Blob Fighter"]] > 0
        assert played[CARD_INDEX_MAP["Scout"]] == 0

    def test_opponent_played_zone_is_populated(self):
        player = _make_player(hand_names=["Scout"], played_names=["Trade Pod"])
        out = np.zeros(6 + NUM_CARDS * 4, dtype=np.float32)
        encode_opponent_into(player, NUM_CARDS, CARD_INDEX_MAP, out, 0)

        played = out[6 + NUM_CARDS: 6 + NUM_CARDS * 2]
        assert played[CARD_INDEX_MAP["Trade Pod"]] > 0

    def test_states_differing_only_by_played_cards_are_distinguishable(self):
        """The regression this zone exists to prevent."""
        size = 5 + NUM_CARDS * 5
        empty = _make_player(hand_names=["Scout"])
        with_played = _make_player(hand_names=["Scout"], played_names=["Blob Fighter"])

        out_a = np.zeros(size, dtype=np.float32)
        out_b = np.zeros(size, dtype=np.float32)
        encode_player_into(empty, NUM_CARDS, CARD_INDEX_MAP, out_a, 0)
        encode_player_into(with_played, NUM_CARDS, CARD_INDEX_MAP, out_b, 0)

        assert not np.array_equal(out_a, out_b)

    def test_played_base_appears_in_both_played_and_bases(self):
        """A base played this turn is distinguishable from an established one."""
        base = _make_card("Trade Pod", CARD_INDEX_MAP["Trade Pod"])
        player = _make_player()
        player.played_cards = [base]
        player.bases = [base]

        out = np.zeros(5 + NUM_CARDS * 5, dtype=np.float32)
        encode_player_into(player, NUM_CARDS, CARD_INDEX_MAP, out, 0)

        played = out[5 + NUM_CARDS: 5 + NUM_CARDS * 2]
        bases = out[5 + NUM_CARDS * 4: 5 + NUM_CARDS * 5]
        idx = CARD_INDEX_MAP["Trade Pod"]
        assert played[idx] > 0
        assert bases[idx] > 0


class TestPPOActorCriticSmoke:
    """Smoke test that the full forward pass works with the new layout."""

    def test_single_state_forward(self):
        from src.ppo.ppo_actor_critic import PPOActorCritic

        action_dim = get_action_space_size(CARD_NAMES)
        state_dim = get_state_size(CARD_NAMES)
        model = PPOActorCritic(state_dim, action_dim, NUM_CARDS)

        x = torch.randn(state_dim)
        logits, value = model(x)
        assert logits.shape == (action_dim,)
        assert value.shape == ()

    def test_batch_forward(self):
        from src.ppo.ppo_actor_critic import PPOActorCritic

        action_dim = get_action_space_size(CARD_NAMES)
        state_dim = get_state_size(CARD_NAMES)
        model = PPOActorCritic(state_dim, action_dim, NUM_CARDS)

        batch_size = 4
        x = torch.randn(batch_size, state_dim)
        logits, value = model(x)
        assert logits.shape == (batch_size, action_dim)
        assert value.shape == (batch_size,)
