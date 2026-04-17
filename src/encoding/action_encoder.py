from src.cards.card import Card
from src.engine.actions import Action, ActionType, CardSource
from src.utils.logger import log
from src.cards.effects import Effect, CardEffectType

# Canonical action-index constants. Import these instead of hardcoding
# magic numbers when building or inspecting action masks.
INVALID_INDEX = 0
END_TURN_INDEX = 1
SKIP_INDEX = 2

# Number of card-indexed action type groups in the encoding scheme.
NUM_CARD_ACTION_TYPES = 10

# Ordered names for card-indexed action types (matches encoding order).
CARD_ACTION_TYPE_NAMES = [
    'play_card', 'buy_card', 'attack_base', 'destroy_base',
    'effect_nonscrap', 'effect_scrap',
    'scrap_hand', 'scrap_discard', 'scrap_trade',
    'discard',
]


def get_action_space_layout(num_cards: int) -> dict:
    """Return the action space layout as a dict of offsets and indices.

    This is the single source of truth for how action indices map to
    semantic groups. Use this instead of hardcoding offsets.

    Returns a dict with:
        'global_indices': dict mapping global action names to their flat indices
        'card_groups': list of (name, offset, count) for each card-indexed group
        'num_cards': the num_cards used to compute offsets
    """
    off = 3  # after invalid(0), end_turn(1), skip(2)

    card_groups = []
    for name in CARD_ACTION_TYPE_NAMES[:2]:  # play_card, buy_card
        card_groups.append((name, off, num_cards))
        off += num_cards

    # attack_player sits between buy_card and attack_base
    attack_player_idx = off
    off += 1

    for name in CARD_ACTION_TYPE_NAMES[2:]:  # attack_base onward
        card_groups.append((name, off, num_cards))
        off += num_cards

    global_indices = {
        'invalid': INVALID_INDEX,
        'end_turn': END_TURN_INDEX,
        'skip': SKIP_INDEX,
        'attack_player': attack_player_idx,
    }

    return {
        'global_indices': global_indices,
        'card_groups': card_groups,
        'num_cards': num_cards,
    }


def get_action_space_size(cards: list[str]) -> int:
    """Calculate the total size of the action space based on the encoding scheme.

    Args:
        cards: The list of all possible cards in the game.

    Returns:
        The total number of possible encoded actions.
    """
    cards_length = len(cards)
    # Calculate size based on the ranges defined in encode_action
    size = (
        1                 # Invalid Action
        + 1               # END_TURN
        + 1               # SKIP_DECISION
        + cards_length    # PLAY_CARD
        + cards_length    # BUY_CARD
        + 1               # ATTACK_PLAYER
        + cards_length    # ATTACK_BASE
        + cards_length    # DESTROY_BASE
        + 2 * cards_length  # APPLY_EFFECT (card and scrap flag)
        + 3 * cards_length  # SCRAP_CARD (hand, discard, trade)
        + cards_length      # DISCARD_CARDS (target discard)
    )
    return size

def encode_action(action: Action | None, cards: list[str], card_index_map: dict[str, int] = None) -> int:
    """Convert an Action object to a numerical index for neural network processing
    
    Maps different action types to different index ranges.
    Action.card_id is already an integer card index.
    
    Returns an integer representation of the action.
    """

    if action is None:
        return 0

    current_act_index = 1

    if action.type == ActionType.END_TURN:
        return current_act_index
    current_act_index += 1

    if action.type == ActionType.SKIP_DECISION:
        return current_act_index
    current_act_index += 1

    cards_length = len(cards)
    card_index = action.card_id  # Already an int index

    # Encode play card action based on the card
    if action.type == ActionType.PLAY_CARD:
        if card_index is not None:
            return current_act_index + card_index
    current_act_index += cards_length
    
    # Encode buy card action based on the card
    if action.type == ActionType.BUY_CARD:
        if card_index is not None:
            return current_act_index + card_index
    current_act_index += cards_length  # end of BUY_CARD range

    # Encode ATTACK_PLAYER
    if action.type == ActionType.ATTACK_PLAYER:
        return current_act_index
    current_act_index += 1

    # Encode ATTACK_BASE per card index
    if action.type == ActionType.ATTACK_BASE and card_index is not None:
        return current_act_index + card_index
    current_act_index += cards_length

    # Encode DESTROY_BASE per card index
    if action.type == ActionType.DESTROY_BASE and card_index is not None:
        return current_act_index + card_index
    current_act_index += cards_length

    # Encode APPLY_EFFECT: include card index and scrap effect flag
    apply_effect_start = current_act_index
    non_scrap_start = apply_effect_start
    scrap_start = apply_effect_start + cards_length
    if action.type == ActionType.APPLY_EFFECT and card_index is not None:
        # Distinguish scrap effects
        if action.card_effect and action.card_effect.is_scrap_effect:
            return scrap_start + card_index
        else:
            return non_scrap_start + card_index
    current_act_index += 2 * cards_length

    # Encode SCRAP_CARD
    scrap_hand_start_index = current_act_index
    scrap_discard_start_index = scrap_hand_start_index + cards_length
    scrap_trade_start_index = scrap_discard_start_index + cards_length
    
    if action.type == ActionType.SCRAP_CARD:
        if card_index is not None: # Check if card was found
            if action.card_source == CardSource.HAND:
                return scrap_hand_start_index + card_index
            if action.card_source == CardSource.DISCARD:
                return scrap_discard_start_index + card_index
            if action.card_source == CardSource.TRADE:
                return scrap_trade_start_index + card_index
                
    current_act_index += 3 * cards_length # Increment for all three potential scrap sources

    # Encode DISCARD_CARDS: target opponent discard by card index
    discard_start_index = current_act_index
    if action.type == ActionType.DISCARD_CARDS and card_index is not None:
        return discard_start_index + card_index
    current_act_index += cards_length

    log(f"Invalid action: {action} with index {current_act_index}")

    # Default case
    return 0

def decode_action(action_idx: int, card_names: list[str]) -> Action:
    """Decode an action index into a new Action object.

    Returns Actions with integer card_id (card index) instead of string names.
    The card reference (action.card) is NOT set — callers that need it must
    resolve from live game state.
    """
    from src.engine.actions import Action, ActionType, CardSource
    cards_length = len(card_names)
    current_act_index = 1

    # 0: Invalid Action
    if action_idx <= 0:
        return Action(type=ActionType.END_TURN)

    # END_TURN
    if action_idx == current_act_index:
        return Action(type=ActionType.END_TURN)
    current_act_index += 1

    # SKIP_DECISION
    if action_idx == current_act_index:
        return Action(type=ActionType.SKIP_DECISION)
    current_act_index += 1

    # PLAY_CARD
    if current_act_index <= action_idx < current_act_index + cards_length:
        card_idx = action_idx - current_act_index
        if 0 <= card_idx < cards_length:
            return Action(type=ActionType.PLAY_CARD, card_id=card_idx)
    current_act_index += cards_length

    # BUY_CARD
    if current_act_index <= action_idx < current_act_index + cards_length:
        card_idx = action_idx - current_act_index
        if 0 <= card_idx < cards_length:
            return Action(type=ActionType.BUY_CARD, card_id=card_idx)
    current_act_index += cards_length

    # ATTACK_PLAYER
    if action_idx == current_act_index:
        return Action(type=ActionType.ATTACK_PLAYER)
    current_act_index += 1

    # ATTACK_BASE
    if current_act_index <= action_idx < current_act_index + cards_length:
        card_idx = action_idx - current_act_index
        if 0 <= card_idx < cards_length:
            return Action(type=ActionType.ATTACK_BASE, card_id=card_idx, target_id=card_idx)
    current_act_index += cards_length

    # DESTROY_BASE
    if current_act_index <= action_idx < current_act_index + cards_length:
        card_idx = action_idx - current_act_index
        if 0 <= card_idx < cards_length:
            return Action(type=ActionType.DESTROY_BASE, card_id=card_idx, target_id=card_idx)
    current_act_index += cards_length

    # APPLY_EFFECT (non-scrap and scrap)
    apply_effect_start = current_act_index
    # Non-scrap
    if apply_effect_start <= action_idx < apply_effect_start + cards_length:
        card_idx = action_idx - apply_effect_start
        if 0 <= card_idx < cards_length:
            return Action(type=ActionType.APPLY_EFFECT, card_id=card_idx, card_effect=Effect(effect_type=CardEffectType.DRAW, is_scrap_effect=False))
    # Scrap
    if apply_effect_start + cards_length <= action_idx < apply_effect_start + 2 * cards_length:
        card_idx = action_idx - (apply_effect_start + cards_length)
        if 0 <= card_idx < cards_length:
            return Action(type=ActionType.APPLY_EFFECT, card_id=card_idx, card_effect=Effect(effect_type=CardEffectType.SCRAP, is_scrap_effect=True))
    current_act_index += 2 * cards_length

    # SCRAP_CARD (hand, discard, trade)
    scrap_hand_start = current_act_index
    scrap_discard_start = scrap_hand_start + cards_length
    scrap_trade_start = scrap_discard_start + cards_length
    # hand
    if scrap_hand_start <= action_idx < scrap_hand_start + cards_length:
        card_idx = action_idx - scrap_hand_start
        if 0 <= card_idx < cards_length:
            return Action(type=ActionType.SCRAP_CARD, card_id=card_idx, card_source=CardSource.HAND)
    # discard
    if scrap_discard_start <= action_idx < scrap_discard_start + cards_length:
        card_idx = action_idx - scrap_discard_start
        if 0 <= card_idx < cards_length:
            return Action(type=ActionType.SCRAP_CARD, card_id=card_idx, card_source=CardSource.DISCARD)
    # trade
    if scrap_trade_start <= action_idx < scrap_trade_start + cards_length:
        card_idx = action_idx - scrap_trade_start
        if 0 <= card_idx < cards_length:
            return Action(type=ActionType.SCRAP_CARD, card_id=card_idx, card_source=CardSource.TRADE)
    current_act_index += 3 * cards_length

    # DISCARD_CARDS (target discard by card index)
    discard_start_index = current_act_index
    if discard_start_index <= action_idx < discard_start_index + cards_length:
        card_idx = action_idx - discard_start_index
        if 0 <= card_idx < cards_length:
            return Action(type=ActionType.DISCARD_CARDS, card_id=card_idx, card_source=CardSource.OPPONENT)
    current_act_index += cards_length

    # Fallback: return END_TURN
    return Action(type=ActionType.END_TURN)