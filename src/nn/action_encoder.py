from src.cards.card import Card
from src.engine.actions import Action, ActionType
from src.utils.logger import log

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
        + cards_length    # ATTACK_BASE
        + 1               # ATTACK_PLAYER
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

    # Encode ATTACK_BASE per card index
    if action.type == ActionType.ATTACK_BASE and card_index is not None:
        return current_act_index + card_index
    current_act_index += cards_length

    # Encode ATTACK_PLAYER
    if action.type == ActionType.ATTACK_PLAYER:
        return current_act_index
    current_act_index += 1

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

def decode_action(action_idx: int, available_actions: list[Action], cards: list[Card]) -> Action:
    """Convert a neural network action index back to a game Action object
    
    Parameters:
    - action_idx: The output index from the neural network
    - available_actions: List of valid actions for the current game state
    
    Returns the corresponding Action from available_actions that matches
    the action_idx, or falls back to a default action if invalid.
    """
    # Bound check
    if action_idx <= 0 or not available_actions:
        return available_actions[0] if available_actions else Action(type=ActionType.END_TURN)
    
    # Create mappings of available actions by type
    action_by_type = {ActionType.END_TURN: [], ActionType.SKIP_DECISION: [],
                    ActionType.PLAY_CARD: [], ActionType.BUY_CARD: [],
                    ActionType.ATTACK_BASE: [], ActionType.ATTACK_PLAYER: [],
                    ActionType.APPLY_EFFECT: [], ActionType.SCRAP_CARD: [],
                    ActionType.DISCARD_CARDS: [], ActionType.DESTROY_BASE: []}
    
    for action in available_actions:
        if action.type in action_by_type:
            action_by_type[action.type].append(action)
    
    # Handle END_TURN (index 1)
    if action_idx == 1 and action_by_type[ActionType.END_TURN]:
        return action_by_type[ActionType.END_TURN][0]
    
    # Handle SKIP_DECISION (index 2)
    if action_idx == 2 and action_by_type[ActionType.SKIP_DECISION]:
        return action_by_type[ActionType.SKIP_DECISION][0]
    
    current_act_index = 3
    cards_length = len(cards)

    # Handle PLAY_CARD
    if current_act_index <= action_idx < current_act_index + cards_length and action_by_type[ActionType.PLAY_CARD]:
        card_idx = action_idx - current_act_index
        card_name = cards[card_idx].name if card_idx < len(cards) else None
        if card_name:
            for action in action_by_type[ActionType.PLAY_CARD]:
                if action.card_id == card_name:
                    return action
    current_act_index += cards_length

    # Handle BUY_CARD
    if current_act_index <= action_idx < current_act_index + cards_length and action_by_type[ActionType.BUY_CARD]:
        card_idx = action_idx - current_act_index
        card_name = cards[card_idx].name if card_idx < len(cards) else None
        if card_name:
            for action in action_by_type[ActionType.BUY_CARD]:
                if action.card_id == card_name:
                    return action
    current_act_index += cards_length

    # Handle ATTACK_BASE per card
    if current_act_index <= action_idx < current_act_index + cards_length and action_by_type[ActionType.ATTACK_BASE]:
        card_idx = action_idx - current_act_index
        card_name = cards[card_idx].name if card_idx < len(cards) else None
        if card_name:
            for action in action_by_type[ActionType.ATTACK_BASE]:
                if action.card_id == card_name:
                    return action
    current_act_index += cards_length

    # Handle ATTACK_PLAYER
    if action_idx == current_act_index and action_by_type[ActionType.ATTACK_PLAYER]:
        attack_action = action_by_type[ActionType.ATTACK_PLAYER][0]
        if attack_action.target_id is not None:
            return attack_action
    current_act_index += 1

    # Handle DESTROY_BASE per card
    if current_act_index <= action_idx < current_act_index + cards_length and action_by_type[ActionType.DESTROY_BASE]:
        card_idx = action_idx - current_act_index
        card_name = cards[card_idx].name if card_idx < len(cards) else None
        if card_name:
            for action in action_by_type[ActionType.DESTROY_BASE]:
                if action.card_id == card_name:
                    return action
    current_act_index += cards_length
    
    # Handle APPLY_EFFECT (card index + scrap flag)
    apply_effect_start = current_act_index
    if action_by_type[ActionType.APPLY_EFFECT]:
        # Determine if in non-scrap or scrap range
        card_idx = None; is_scrap = None
        # Non-scrap effects range
        if apply_effect_start <= action_idx < apply_effect_start + cards_length:
            card_idx = action_idx - apply_effect_start
            is_scrap = False
        # Scrap effects range
        elif apply_effect_start + cards_length <= action_idx < apply_effect_start + 2 * cards_length:
            card_idx = action_idx - (apply_effect_start + cards_length)
            is_scrap = True
        # Match action by card and scrap flag
        if card_idx is not None and 0 <= card_idx < len(cards):
            card_name = cards[card_idx].name
            for action in action_by_type[ActionType.APPLY_EFFECT]:
                if action.card_id == card_name and action.card_effect and action.card_effect.is_scrap_effect == is_scrap:
                    return action
    current_act_index += 2 * cards_length
  
    # Handle SCRAP_CARD
    scrap_start_index = current_act_index
    card_idx = -1
    expected_source = None

    # Check hand source range
    if scrap_start_index <= action_idx < scrap_start_index + cards_length:
        card_idx = action_idx - scrap_start_index
        expected_source = "hand"
    # Check discard source range
    elif scrap_start_index + cards_length <= action_idx < scrap_start_index + 2 * cards_length:
        card_idx = action_idx - (scrap_start_index + cards_length)
        expected_source = "discard"
    # Check trade source range
    elif scrap_start_index + 2 * cards_length <= action_idx < scrap_start_index + 3 * cards_length:
        card_idx = action_idx - (scrap_start_index + 2 * cards_length)
        expected_source = "trade"

    if card_idx != -1 and expected_source and action_by_type[ActionType.SCRAP_CARD]:
        card_name = cards[card_idx].name if card_idx < len(cards) else None
        if card_name:
            for action in action_by_type[ActionType.SCRAP_CARD]:
                # Check if the action matches the card name and the expected source is allowed for this action
                if action.card_id == card_name and expected_source in action.card_sources:
                    return action
    
    current_act_index += 3 * cards_length # Account for all three potential scrap sources

    # Handle DISCARD_CARDS
    discard_start_index = current_act_index
    if discard_start_index <= action_idx < discard_start_index + cards_length and action_by_type[ActionType.DISCARD_CARDS]:
        card_idx = action_idx - discard_start_index
        card_name = cards[card_idx].name if card_idx < len(cards) else None
        if card_name:
            for action in action_by_type[ActionType.DISCARD_CARDS]:
                if action.card_id == card_name and action.card_source == "opponent":
                    return action
    current_act_index += cards_length

    # Fallback: return first available action
    return available_actions[0]