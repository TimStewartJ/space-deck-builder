from dataclasses import dataclass
from typing import Optional, List, Any
from enum import Enum

class ActionType(Enum):
    PLAY_CARD = "play_card"
    BUY_CARD = "buy_card"
    ATTACK_BASE = "attack_base"
    ATTACK_PLAYER = "attack_player"
    SCRAP_CARD = "scrap_card"
    END_TURN = "end_turn"

@dataclass
class Action:
    type: ActionType
    card_id: Optional[str] = None
    target_id: Optional[str] = None
    source: Optional[str] = None
    additional_params: Optional[dict] = None
    
    def __str__(self):
        """String representation for display in CLI"""
        if self.type == ActionType.PLAY_CARD and self.card_id:
            return f"Play card: {self.card_id}"
        elif self.type == ActionType.BUY_CARD and self.card_id:
            return f"Buy card: {self.card_id}"
        elif self.type == ActionType.ATTACK_BASE:
            return f"Attack base: {self.target_id}"
        elif self.type == ActionType.ATTACK_PLAYER:
            return f"Attack player: {self.target_id}"
        elif self.type == ActionType.SCRAP_CARD:
            return f"Scrap card: {self.card_id}"
        elif self.type == ActionType.END_TURN:
            return "End turn"
        return f"{self.type}"

def get_available_actions(game_state, player):
    """Return list of available actions for a player given the current game state"""
    actions = []
    
    # Add play card actions for each card in hand
    for card in player.hand:
        actions.append(Action(type=ActionType.PLAY_CARD, card_id=card.name))
    
    # Add buy card actions for affordable cards in trade row
    for card in game_state.trade_row:
        if player.trade >= card.cost:
            actions.append(Action(type=ActionType.BUY_CARD, card_id=card.name))
    
    # Add attack actions if player has combat available
    if player.combat > 0:
        for opponent in game_state.players:
            if opponent != player:
                # First target outposts
                for base in [b for b in opponent.bases if b.is_outpost()]:
                    if player.combat >= base.defense:
                        actions.append(Action(type=ActionType.ATTACK_BASE, target_id=base.name))
                
                # If no outposts, can attack other bases or player directly
                if not any(b.is_outpost() for b in opponent.bases):
                    for base in [b for b in opponent.bases if not b.is_outpost()]:
                        if player.combat >= base.defense:
                            actions.append(Action(type=ActionType.ATTACK_BASE, target_id=base.name))
                    actions.append(Action(type=ActionType.ATTACK_PLAYER, target_id=opponent.name))
    
    # Always allow ending turn
    actions.append(Action(type=ActionType.END_TURN))
    
    return actions