import torch
from typing import TYPE_CHECKING

from src.engine.actions import ActionType, get_available_actions
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
        1 + # can_buy_anything flag
        1 + # has_available_actions flag
        cards_length + # Trade row
        player_encoding_size * 2 # Players
    )

def encode_card_presence(cards_list: list['Card'], cards: list[str]) -> list[float]:
    """Encode card presence in a list of cards"""

    card_presence_worth = 0.0125

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

def encode_state(game_state: 'Game', is_current_player_training: bool, cards: list[str], available_actions: list['Action']) -> torch.FloatTensor:
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

    # Encode if there are non-end turn and non-scrap actions available
    can_buy_anything = 1.0 if any(
            action.type == ActionType.BUY_CARD
            for action in available_actions
        ) else 0.0
    state.append(can_buy_anything)

    has_available_actions = 1.0 if any(
            action.type == ActionType.ATTACK_PLAYER or action.type == ActionType.PLAY_CARD
            for action in available_actions
        ) else 0.0
    state.append(has_available_actions)
    
    # Encode trade row
    trade_row_encoding = encode_card_presence(game_state.trade_row, cards)
    state.extend(trade_row_encoding)

    # Encode training player
    state.extend(encode_player(training_player, cards=cards))

    # Encode opponent player
    state.extend(encode_player(opponent, cards=cards))

    return torch.FloatTensor(state)