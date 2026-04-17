import torch
import numpy as np
from typing import TYPE_CHECKING

from src.engine.actions import ActionType, get_available_actions
from src.encoding.action_encoder import encode_action, get_action_space_size

if TYPE_CHECKING:
    from src.engine.game import Game
    from src.cards.card import Card
    from src.engine.player import Player
    from src.engine.actions import Action


def build_card_index_map(cards: list[str]) -> dict[str, int]:
    """Build O(1) card name → index lookup table."""
    return {name: i for i, name in enumerate(cards)}


def get_state_size(cards: list[str]) -> int:
    """Get the size of the state vector.

    Training player and opponent have asymmetric encodings because
    the opponent's hand and deck are hidden information — they are
    merged into a single 'unseen' zone.
    """
    n = len(cards)
    training_player_size = (
        5 +     # Scalars: trade, combat, health, deck_size, discard_size
        n * 4   # Zones: hand, discard, deck, bases
    )
    opponent_size = (
        6 +     # Scalars: trade, combat, health, deck_size, hand_size, discard_size
        n * 3   # Zones: unseen (hand+deck), discard, bases
    )
    return (
        4 +     # Flags: is_training_player, is_first_player, can_buy, has_actions
        n +     # Trade row
        training_player_size +
        opponent_size
    )

def encode_card_presence(cards_list: list['Card'], num_cards: int, card_index_map: dict[str, int] = None, out: np.ndarray = None, offset: int = 0) -> np.ndarray:
    """Encode card presence using direct card.index for O(1) lookups.
    
    Falls back to card_index_map dict lookup when card.index is out of bounds
    (e.g. starter cards created without a canonical index map).
    """
    card_presence_worth = 0.125

    if out is None:
        out = np.zeros(num_cards, dtype=np.float32)
        offset = 0

    for card in cards_list:
        idx = card.index
        if not (0 <= idx < num_cards):
            # Fallback for cards without a valid encoding index
            if card_index_map is not None:
                idx = card_index_map.get(card.name, -1)
            else:
                continue
        if 0 <= idx < num_cards:
            out[offset + idx] = min(out[offset + idx] + card_presence_worth, 1.0)
    return out

def encode_player_into(player: 'Player', num_cards: int, card_index_map: dict[str, int], out: np.ndarray, offset: int) -> int:
    """Encode player info directly into pre-allocated numpy array. Returns new offset."""
    # Player resources
    out[offset] = player.trade / 100.0
    out[offset + 1] = player.combat / 100.0
    out[offset + 2] = player.health / 100.0
    out[offset + 3] = len(player.deck) / 40.0
    out[offset + 4] = len(player.discard_pile) / 40.0
    offset += 5

    # Hand, discard, deck, bases — each is num_cards wide
    encode_card_presence(player.hand, num_cards, out=out, offset=offset)
    offset += num_cards
    encode_card_presence(player.discard_pile, num_cards, out=out, offset=offset)
    offset += num_cards
    encode_card_presence(player.deck, num_cards, out=out, offset=offset)
    offset += num_cards
    encode_card_presence(player.bases, num_cards, out=out, offset=offset)
    offset += num_cards

    return offset

def encode_opponent_into(player: 'Player', num_cards: int, card_index_map: dict[str, int], out: np.ndarray, offset: int) -> int:
    """Encode opponent info with hidden-information-aware zones.

    Unlike encode_player_into(), the opponent's hand and deck are merged
    into a single 'unseen' zone since the active player cannot observe
    which cards the opponent drew vs. which remain in their deck.
    Hand size is encoded as an explicit scalar since it is observable.
    """
    # Opponent resources (6 scalars)
    out[offset] = player.trade / 100.0
    out[offset + 1] = player.combat / 100.0
    out[offset + 2] = player.health / 100.0
    out[offset + 3] = len(player.deck) / 40.0
    out[offset + 4] = len(player.hand) / 10.0
    out[offset + 5] = len(player.discard_pile) / 40.0
    offset += 6

    # Unseen zone: encode hand and deck separately into the same output
    # region to avoid creating a concatenated list
    encode_card_presence(player.hand, num_cards, out=out, offset=offset)
    encode_card_presence(player.deck, num_cards, out=out, offset=offset)
    offset += num_cards
    encode_card_presence(player.discard_pile, num_cards, out=out, offset=offset)
    offset += num_cards
    encode_card_presence(player.bases, num_cards, out=out, offset=offset)
    offset += num_cards

    return offset

def encode_state(game_state: 'Game', is_current_player_training: bool, cards: list[str], available_actions: list['Action'] = None, card_index_map: dict[str, int] = None, state_buf: np.ndarray = None, can_buy: bool = None, has_actions: bool = None, return_numpy: bool = False):
    """Convert game state to fixed-length state vector.

    If state_buf is provided, it is zeroed and filled in place. When
    return_numpy=True the filled buffer (or a freshly allocated numpy
    array if state_buf was None) is returned without torch wrapping —
    hot inference paths use this to skip the CPU→torch→numpy round-trip
    when the downstream consumer is IPC/numpy. Default behavior returns
    a fresh torch.FloatTensor so existing callers remain unchanged.
    Accepts pre-computed can_buy/has_actions flags to avoid scanning
    available_actions. Falls back to scanning if flags are not provided.
    """
    num_cards = len(cards)

    if state_buf is not None:
        state_buf[:] = 0
        state = state_buf
    else:
        state_size = get_state_size(cards)
        state = np.zeros(state_size, dtype=np.float32)

    training_player = game_state.current_player if is_current_player_training else game_state.get_opponent(game_state.current_player)
    if training_player is None:
        raise ValueError("Training player not found in game state.")

    opponent = game_state.get_opponent(training_player)
    if opponent is None:
        raise ValueError("Opponent not found in game state.")

    offset = 0

    # Flags — use pre-computed booleans if provided, else fall back to scanning
    state[offset] = 1.0 if is_current_player_training else 0.0
    offset += 1
    state[offset] = 1.0 if game_state.first_player_name == training_player.name else 0.0
    offset += 1
    if can_buy is not None:
        state[offset] = 1.0 if can_buy else 0.0
    else:
        state[offset] = 1.0 if (available_actions and any(a.type == ActionType.BUY_CARD for a in available_actions)) else 0.0
    offset += 1
    if has_actions is not None:
        state[offset] = 1.0 if has_actions else 0.0
    else:
        state[offset] = 1.0 if (available_actions and any(a.type in (ActionType.ATTACK_PLAYER, ActionType.PLAY_CARD) for a in available_actions)) else 0.0
    offset += 1

    # Trade row
    encode_card_presence(game_state.trade_row, num_cards, out=state, offset=offset)
    offset += num_cards

    # Training player
    offset = encode_player_into(training_player, num_cards, card_index_map, state, offset)

    # Opponent (asymmetric encoding — hand and deck merged into unseen zone)
    offset = encode_opponent_into(opponent, num_cards, card_index_map, state, offset)

    # Return the raw numpy buffer for hot paths that feed IPC/numpy. When
    # state_buf was supplied the caller is responsible for not mutating it
    # before they're done with the returned array (or copying it). When
    # state_buf is None the freshly-allocated array is returned directly.
    if return_numpy:
        return state

    # Return a copy so the buffer can be safely reused
    return torch.from_numpy(state.copy()) if state_buf is not None else torch.from_numpy(state)