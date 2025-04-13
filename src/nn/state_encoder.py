import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.engine.game import Game
    from src.cards.card import Card

# Constants for encoding
MAX_HAND = 20
MAX_TRADE_ROW = 5
MAX_BASES = 10
CARD_ENCODING_SIZE = 19 # Determined by the encode_card function structure
STATE_SIZE = (
    5 + # Player resources
    MAX_HAND * CARD_ENCODING_SIZE +
    MAX_TRADE_ROW * CARD_ENCODING_SIZE +
    MAX_BASES * CARD_ENCODING_SIZE + # Player bases
    MAX_BASES * CARD_ENCODING_SIZE   # Opponent bases
)


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


def encode_state(game_state: 'Game') -> torch.FloatTensor:
    """Convert variable-length game state to fixed-length tensor"""
    state = []

    # Encode player resources (fixed length)
    state.extend([
        game_state.current_player.trade / 100.0,  # Normalize values
        game_state.current_player.combat / 100.0,
        game_state.current_player.health / 100.0,
        len(game_state.current_player.deck) / 40.0,
        len(game_state.current_player.discard_pile) / 40.0
    ])

    # Encode player hand (variable -> fixed)
    hand_encoding = []
    for card in game_state.current_player.hand[:MAX_HAND]:
        hand_encoding.extend(encode_card(card))
    # Pad to fixed length
    padding_needed = MAX_HAND - len(game_state.current_player.hand)
    hand_encoding.extend([0.0] * (padding_needed * CARD_ENCODING_SIZE))
    state.extend(hand_encoding)

    # Encode trade row
    trade_row_encoding = []
    for card in game_state.trade_row[:MAX_TRADE_ROW]:
        trade_row_encoding.extend(encode_card(card))
    padding_needed = MAX_TRADE_ROW - len(game_state.trade_row)
    trade_row_encoding.extend([0.0] * (padding_needed * CARD_ENCODING_SIZE))
    state.extend(trade_row_encoding)

    # Encode player bases
    bases_encoding = []
    for base in game_state.current_player.bases[:MAX_BASES]:
        bases_encoding.extend(encode_card(base))
    padding_needed = MAX_BASES - len(game_state.current_player.bases)
    bases_encoding.extend([0.0] * (padding_needed * CARD_ENCODING_SIZE))
    state.extend(bases_encoding)

    # Encode opponent bases
    opponent = game_state.get_opponent(game_state.current_player)
    opponent_bases_encoding = []
    # Add check for opponent existence
    if opponent:
        for base in opponent.bases[:MAX_BASES]:
            opponent_bases_encoding.extend(encode_card(base))
        padding_needed = MAX_BASES - len(opponent.bases)
        opponent_bases_encoding.extend([0.0] * (padding_needed * CARD_ENCODING_SIZE))
    else:
        # If no opponent, pad with zeros for the entire opponent bases section
        opponent_bases_encoding.extend([0.0] * (MAX_BASES * CARD_ENCODING_SIZE))
    state.extend(opponent_bases_encoding)

    # Ensure final state vector matches STATE_SIZE
    if len(state) != STATE_SIZE:
        # Pad or truncate if necessary, though it indicates a mismatch
        state = state[:STATE_SIZE] + [0.0] * (STATE_SIZE - len(state))

    return torch.FloatTensor(state)