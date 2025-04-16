import torch
from typing import TYPE_CHECKING

from src.engine.actions import get_available_actions
from src.nn.action_encoder import encode_action, get_action_space_size

if TYPE_CHECKING:
    from src.engine.game import Game
    from src.cards.card import Card
    from src.engine.player import Player
    from src.engine.actions import Action

def get_state_size(cards: list[str]) -> int:
    """Get the size of the state vector"""
    cards_length = len(cards)
    player_encoding_size = (
        5 + # Player resources (trade, combat, health, deck size, discard pile size)
        cards_length * 4 # Player hand, discard pile, deck, bases
    )
    return (
        1 + # is_training_player flag
        1 + # is_first_player flag
        cards_length + # Trade row
        player_encoding_size * 2 + # Players
        get_action_space_size(cards) # Available actions
    )

def encode_card_presence(cards_list: list['Card'], cards: list[str]) -> list[float]:
    """Encode card presence in a list of cards"""

    card_presence_worth = 0.25

    card_presence = [0.0] * len(cards)
    for card in cards_list:
        if card.name in cards:
            card_index = card.index
            card_presence[card_index] += card_presence_worth  # Increment the count for this card
            card_presence[card_index] = min(card_presence[card_index], 1.0)  # Cap at 1.0
    return card_presence

def encode_player(player: 'Player', cards: list[str]) -> list[float]:
    """Convert player information to a fixed-length vector"""
    # Player resources (trade, combat, health, deck size, discard pile size)
    player_resources = [
        player.trade / 100.0,  # Normalize values
        player.combat / 100.0,
        player.health / 100.0,
        len(player.deck) / 40.0,
        len(player.discard_pile) / 40.0
    ]

    # Encode hand cards
    hand_encoding = encode_card_presence(player.hand, cards)
    
    # Encode discard pile cards
    discard_encoding = encode_card_presence(player.discard_pile, cards)

    # Encode draw deck cards
    deck_encoding = encode_card_presence(player.deck, cards)

    # Encode bases as well
    bases_encoding = encode_card_presence(player.bases, cards)

    return player_resources + hand_encoding + discard_encoding + deck_encoding + bases_encoding

def encode_state(game_state: 'Game', is_current_player_training: bool, cards: list[str], available_actions: list['Action'] | None = None) -> torch.FloatTensor:
    """Convert variable-length game state to fixed-length tensor"""
    state = []

    training_player = game_state.current_player if is_current_player_training else game_state.get_opponent(game_state.current_player)
    if training_player is None:
        raise ValueError("Training player not found in game state.")
    
    opponent = game_state.get_opponent(training_player)
    if opponent is None:
        raise ValueError("Opponent not found in game state.")
    
    # Encode if the current player is the training player
    is_training_player = 1.0 if is_current_player_training else 0.0
    state.append(is_training_player)

    # Encode if the first player is the training player
    is_first_player = 1.0 if game_state.first_player_name == training_player.name else 0.0
    state.append(is_first_player)
    
    # Encode trade row
    trade_row_encoding = encode_card_presence(game_state.trade_row, cards)
    state.extend(trade_row_encoding)

    # Encode training player
    state.extend(encode_player(training_player, cards=cards))

    # Encode opponent player
    state.extend(encode_player(opponent, cards=cards))

    # Encode available actions for the training player in 1 hot format
    if available_actions is None:
        available_actions = get_available_actions(game_state, training_player)
    action_encoding = [0.0] * get_action_space_size(cards)
    for action in available_actions:
        action_index = encode_action(action, cards=cards)
        if 0 <= action_index < len(action_encoding):
            action_encoding[action_index] = 1.0
    state.extend(action_encoding)

    return torch.FloatTensor(state)