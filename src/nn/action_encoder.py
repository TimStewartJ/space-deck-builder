from src.cards.card import Card
from src.engine.actions import Action, ActionType
from src.utils.logger import log
from src.cards.effects import Effect, CardEffectType

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

def encode_action(action: Action | None, cards: list[str]) -> int:
    """Convert an Action object to a numerical index for neural network processing
    
    Maps different action types to different index ranges
    
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
    card_index = None
    if action.card_id is not None:
        # Get index of card in all available cards
        card_index = next((i for i, card in enumerate(cards) if card == action.card_id), None)

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
            if "hand" == action.card_source:
                return scrap_hand_start_index + card_index
            if "discard" == action.card_source:
                return scrap_discard_start_index + card_index
            if "trade" == action.card_source:
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
    """Decode an action index into a new Action object, using only the index and card_names.
    Args:
        action_idx: The output index from the neural network
        card_names: List of unique card names (used for index mapping, must match encode_action)
    Returns:
        A new Action object corresponding to the decoded index.
    """
    from src.engine.actions import Action, ActionType
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
            return Action(type=ActionType.PLAY_CARD, card_id=card_names[card_idx])
    current_act_index += cards_length

    # BUY_CARD
    if current_act_index <= action_idx < current_act_index + cards_length:
        card_idx = action_idx - current_act_index
        if 0 <= card_idx < cards_length:
            return Action(type=ActionType.BUY_CARD, card_id=card_names[card_idx])
    current_act_index += cards_length

    # ATTACK_PLAYER
    if action_idx == current_act_index:
        return Action(type=ActionType.ATTACK_PLAYER)
    current_act_index += 1

    # ATTACK_BASE
    if current_act_index <= action_idx < current_act_index + cards_length:
        card_idx = action_idx - current_act_index
        if 0 <= card_idx < cards_length:
            return Action(type=ActionType.ATTACK_BASE, card_id=card_names[card_idx])
    current_act_index += cards_length

    # DESTROY_BASE
    if current_act_index <= action_idx < current_act_index + cards_length:
        card_idx = action_idx - current_act_index
        if 0 <= card_idx < cards_length:
            return Action(type=ActionType.DESTROY_BASE, card_id=card_names[card_idx])
    current_act_index += cards_length

    # APPLY_EFFECT (non-scrap and scrap)
    apply_effect_start = current_act_index
    # Non-scrap
    if apply_effect_start <= action_idx < apply_effect_start + cards_length:
        card_idx = action_idx - apply_effect_start
        if 0 <= card_idx < cards_length:
            return Action(type=ActionType.APPLY_EFFECT, card_id=card_names[card_idx], card_effect=Effect(effect_type=CardEffectType.DRAW, is_scrap_effect=False))
    # Scrap
    if apply_effect_start + cards_length <= action_idx < apply_effect_start + 2 * cards_length:
        card_idx = action_idx - (apply_effect_start + cards_length)
        if 0 <= card_idx < cards_length:
            return Action(type=ActionType.APPLY_EFFECT, card_id=card_names[card_idx], card_effect=Effect(effect_type=CardEffectType.SCRAP, is_scrap_effect=True))
    current_act_index += 2 * cards_length

    # SCRAP_CARD (hand, discard, trade)
    scrap_hand_start = current_act_index
    scrap_discard_start = scrap_hand_start + cards_length
    scrap_trade_start = scrap_discard_start + cards_length
    # hand
    if scrap_hand_start <= action_idx < scrap_hand_start + cards_length:
        card_idx = action_idx - scrap_hand_start
        if 0 <= card_idx < cards_length:
            return Action(type=ActionType.SCRAP_CARD, card_id=card_names[card_idx], card_source="hand")
    # discard
    if scrap_discard_start <= action_idx < scrap_discard_start + cards_length:
        card_idx = action_idx - scrap_discard_start
        if 0 <= card_idx < cards_length:
            return Action(type=ActionType.SCRAP_CARD, card_id=card_names[card_idx], card_source="discard")
    # trade
    if scrap_trade_start <= action_idx < scrap_trade_start + cards_length:
        card_idx = action_idx - scrap_trade_start
        if 0 <= card_idx < cards_length:
            return Action(type=ActionType.SCRAP_CARD, card_id=card_names[card_idx], card_source="trade")
    current_act_index += 3 * cards_length

    # DISCARD_CARDS (target discard by card index)
    discard_start_index = current_act_index
    if discard_start_index <= action_idx < discard_start_index + cards_length:
        card_idx = action_idx - discard_start_index
        if 0 <= card_idx < cards_length:
            return Action(type=ActionType.DISCARD_CARDS, card_id=card_names[card_idx], card_source="opponent")
    current_act_index += cards_length

    # Fallback: return END_TURN
    return Action(type=ActionType.END_TURN)