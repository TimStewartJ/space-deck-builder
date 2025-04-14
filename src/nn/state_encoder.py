import torch
from typing import TYPE_CHECKING

from src.engine.actions import get_available_actions
from src.nn.action_encoder import encode_action, get_action_space_size

if TYPE_CHECKING:
    from src.engine.game import Game
    from src.cards.card import Card
    from src.engine.player import Player

# Constants for encoding
MAX_HAND = 20
MAX_TRADE_ROW = 5
MAX_BASES = 10
CARD_ENCODING_SIZE = 19 # Determined by the encode_card function structure
PLAYER_ENCODING_SIZE = (
    5 + # Player resources (trade, combat, health, deck size, discard pile size)
    MAX_HAND * CARD_ENCODING_SIZE + # Player hand
    MAX_BASES * CARD_ENCODING_SIZE # Player bases
)
STATE_SIZE = (
    1 + # is_training_player flag
    1 + # is_first_player flag
    PLAYER_ENCODING_SIZE + # Player resources
    PLAYER_ENCODING_SIZE + # Opponent resources
    MAX_TRADE_ROW * CARD_ENCODING_SIZE # Trade row
)

def get_state_size(cards: list[str]) -> int:
    """Get the size of the state vector"""
    return STATE_SIZE + get_action_space_size(cards)

def encode_card(card: 'Card') -> list[float]:
    """Convert a card to a fixed-length embedding vector"""
    # imports
    from src.cards.effects import CardEffectType

    # Card type one-hot encoding (ship, base, outpost)
    card_type = [0.0, 0.0, 0.0]  # [ship, base, outpost]
    if card.card_type == "ship":
        card_type[0] = 1.0
    elif card.card_type == "base":
        card_type[1] = 1.0
        if card.is_outpost():
            card_type[2] = 1.0

    # Faction one-hot encoding
    faction = [0.0, 0.0, 0.0, 0.0, 0.0]  # [trade_federation, blob, machine_cult, star_empire, unaligned]
    if card.faction == "Trade Federation":
        faction[0] = 1.0
    elif card.faction == "Blob":
        faction[1] = 1.0
    elif card.faction == "Machine Cult":
        faction[2] = 1.0
    elif card.faction == "Star Empire":
        faction[3] = 1.0
    else:  # Unaligned
        faction[4] = 1.0

    # Numeric properties
    properties = [
        card.cost / 10.0,  # Normalize cost
        card.defense / 10.0 if card.defense else 0.0,
    ]

    # Effects encoding (simplified: only first effect)
    effects = []
    if card.effects:
        effect = card.effects[0]
        effect_type_encoding = [0.0] * len(CardEffectType)
        try:
            effect_type_encoding[list(CardEffectType).index(effect.effect_type)] = 1.0
        except ValueError:
            # Handle cases where effect_type might not be in CardEffectType enum
            pass # Keep as all zeros
        effects.extend(effect_type_encoding)
        # Normalize effect value if it exists, else 0
        effects.append(effect.value / 10.0 if hasattr(effect, 'value') and effect.value is not None else 0.0)
    else:
        # If no effects, add padding for the effect encoding size (len(CardEffectType) + 1 for value)
        effects.extend([0.0] * (len(CardEffectType) + 1))


    # Combine all features
    card_encoding = card_type + faction + properties + effects

    # Ensure the encoding size matches CARD_ENCODING_SIZE
    if len(card_encoding) != CARD_ENCODING_SIZE:
         # This should ideally not happen if CARD_ENCODING_SIZE is calculated correctly
         # Pad or truncate if necessary, though it indicates a mismatch
         card_encoding = card_encoding[:CARD_ENCODING_SIZE] + [0.0] * (CARD_ENCODING_SIZE - len(card_encoding))

    return card_encoding

def encode_player(player: 'Player') -> list[float]:
    """Convert player information to a fixed-length vector"""
    # Player resources (trade, combat, health, deck size, discard pile size)
    player_resources = [
        player.trade / 100.0,  # Normalize values
        player.combat / 100.0,
        player.health / 100.0,
        len(player.deck) / 40.0,
        len(player.discard_pile) / 40.0
    ]

    # Encode hand cards (variable -> fixed)
    hand_encoding = []
    for card in player.hand[:MAX_HAND]:
        hand_encoding.extend(encode_card(card))
    # Pad to fixed length
    padding_needed = MAX_HAND - len(player.hand)
    hand_encoding.extend([0.0] * (padding_needed * CARD_ENCODING_SIZE))

    # Encode bases as well
    bases_encoding = []
    for base in player.bases[:MAX_BASES]:
        bases_encoding.extend(encode_card(base))
    padding_needed = MAX_BASES - len(player.bases)
    bases_encoding.extend([0.0] * (padding_needed * CARD_ENCODING_SIZE))

    return player_resources + hand_encoding + bases_encoding

def encode_state(game_state: 'Game', is_current_player_training: bool, cards: list[str]) -> torch.FloatTensor:
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

    # Encode training player
    state.extend(encode_player(training_player))

    # Encode opponent player
    state.extend(encode_player(opponent))

    # Encode available actions for the training player in 1 hot format
    available_actions = get_available_actions(game_state, training_player)
    action_encoding = [0.0] * get_action_space_size(cards)
    for action in available_actions:
        action_index = encode_action(action, cards=cards)
        if 0 <= action_index < len(action_encoding):
            action_encoding[action_index] = 1.0
    state.extend(action_encoding)

    # Encode trade row
    trade_row_encoding = []
    for card in game_state.trade_row[:MAX_TRADE_ROW]:
        trade_row_encoding.extend(encode_card(card))
    padding_needed = MAX_TRADE_ROW - len(game_state.trade_row)
    trade_row_encoding.extend([0.0] * (padding_needed * CARD_ENCODING_SIZE))
    state.extend(trade_row_encoding)

    return torch.FloatTensor(state)