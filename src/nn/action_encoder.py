from src.engine.actions import Action, ActionType


def encode_action(action: Action) -> int:
    """Convert an Action object to a numerical index for neural network processing
    
    Maps different action types to different index ranges:
    - 0: END_TURN
    - 1: SKIP_DECISION
    - 2-21: PLAY_CARD (indices 0-19 for cards in hand)
    - 22-26: BUY_CARD (indices 0-4 for trade row)
    - 27-36: ATTACK_BASE (indices 0-9 for bases)
    - 37-38: ATTACK_PLAYER (indices 0-1 for players)
    - 39-49: APPLY_EFFECT, SCRAP_CARD, etc.
    
    Returns an integer representation of the action.
    """
    if action.type == ActionType.END_TURN:
        return 0
    
    if action.type == ActionType.SKIP_DECISION:
        return 1
    
    if action.type == ActionType.PLAY_CARD:
        # Find the card index in player's hand
        if hasattr(action, 'card_index'):
            # If index is already computed
            return 2 + action.card_index
        else:
            # Compute card index (max 20 cards in hand)
            for i, card in enumerate(getattr(action, 'player_hand', [])):
                if card.name == action.card_id:
                    return 2 + min(i, 19)
            return 2  # Default to first card if not found
    
    if action.type == ActionType.BUY_CARD:
        # Find the card index in trade row
        if hasattr(action, 'card_index'):
            # If index is already computed
            return 22 + action.card_index
        else:
            # Compute card index (max 5 cards in trade row)
            for i, card in enumerate(getattr(action, 'trade_row', [])):
                if card.name == action.card_id:
                    return 22 + min(i, 4)
            return 22  # Default to first card if not found
    
    if action.type == ActionType.ATTACK_BASE:
        # Encode base index (max 10 bases)
        if hasattr(action, 'base_index'):
            return 27 + min(action.base_index, 9)
        return 27  # Default to first base
    
    if action.type == ActionType.ATTACK_PLAYER:
        # Encode player index (max 2 players)
        if hasattr(action, 'player_index'):
            return 37 + min(action.player_index, 1)
        return 37  # Default to first player
    
    if action.type == ActionType.APPLY_EFFECT:
        # Encode effect actions (max 10 effects)
        if hasattr(action, 'effect_index'):
            return 39 + min(action.effect_index, 9)
        return 39  # Default to first effect
    
    if action.type == ActionType.SCRAP_CARD:
        return 49  # Single index for scrap actions
    
    # Default case
    return 0

def decode_action(action_idx: int, available_actions: list[Action]) -> Action:
    """Convert a neural network action index back to a game Action object
    
    Parameters:
    - action_idx: The output index from the neural network
    - available_actions: List of valid actions for the current game state
    
    Returns the corresponding Action from available_actions that matches
    the action_idx, or falls back to a default action if invalid.
    """
    # Bound check
    if action_idx < 0 or not available_actions:
        return available_actions[0] if available_actions else None
    
    # Create mappings of available actions by type
    action_by_type = {ActionType.END_TURN: [], ActionType.SKIP_DECISION: [],
                    ActionType.PLAY_CARD: [], ActionType.BUY_CARD: [],
                    ActionType.ATTACK_BASE: [], ActionType.ATTACK_PLAYER: [],
                    ActionType.APPLY_EFFECT: [], ActionType.SCRAP_CARD: []}
    
    for action in available_actions:
        if action.type in action_by_type:
            action_by_type[action.type].append(action)
    
    # Handle END_TURN (index 0)
    if action_idx == 0 and action_by_type[ActionType.END_TURN]:
        return action_by_type[ActionType.END_TURN][0]
    
    # Handle SKIP_DECISION (index 1)
    if action_idx == 1 and action_by_type[ActionType.SKIP_DECISION]:
        return action_by_type[ActionType.SKIP_DECISION][0]
    
    # Handle PLAY_CARD (indices 2-21)
    if 2 <= action_idx <= 21 and action_by_type[ActionType.PLAY_CARD]:
        card_idx = action_idx - 2
        if card_idx < len(action_by_type[ActionType.PLAY_CARD]):
            return action_by_type[ActionType.PLAY_CARD][card_idx]
    
    # Handle BUY_CARD (indices 22-26)
    if 22 <= action_idx <= 26 and action_by_type[ActionType.BUY_CARD]:
        card_idx = action_idx - 22
        if card_idx < len(action_by_type[ActionType.BUY_CARD]):
            return action_by_type[ActionType.BUY_CARD][card_idx]
    
    # Handle ATTACK_BASE (indices 27-36)
    if 27 <= action_idx <= 36 and action_by_type[ActionType.ATTACK_BASE]:
        base_idx = action_idx - 27
        if base_idx < len(action_by_type[ActionType.ATTACK_BASE]):
            return action_by_type[ActionType.ATTACK_BASE][base_idx]
    
    # Handle ATTACK_PLAYER (indices 37-38)
    if 37 <= action_idx <= 38 and action_by_type[ActionType.ATTACK_PLAYER]:
        player_idx = action_idx - 37
        if player_idx < len(action_by_type[ActionType.ATTACK_PLAYER]):
            return action_by_type[ActionType.ATTACK_PLAYER][player_idx]
    
    # Handle APPLY_EFFECT (indices 39-48)
    if 39 <= action_idx <= 48 and action_by_type[ActionType.APPLY_EFFECT]:
        effect_idx = action_idx - 39
        if effect_idx < len(action_by_type[ActionType.APPLY_EFFECT]):
            return action_by_type[ActionType.APPLY_EFFECT][effect_idx]
    
    # Handle SCRAP_CARD (index 49)
    if action_idx == 49 and action_by_type[ActionType.SCRAP_CARD]:
        return action_by_type[ActionType.SCRAP_CARD][0]
    
    # Fallback: return first available action
    return available_actions[0]