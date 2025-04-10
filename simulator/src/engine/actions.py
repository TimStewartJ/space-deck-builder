from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, List, Any
from enum import Enum

if TYPE_CHECKING:
    from src.engine.player import Player
    from src.engine.game import Game

class ActionType(Enum):
    PLAY_CARD = "play_card"
    BUY_CARD = "buy_card"
    ATTACK_BASE = "attack_base"
    ATTACK_PLAYER = "attack_player"
    SCRAP_CARD = "scrap_card"
    END_TURN = "end_turn"
    DISCARD_CARDS = "discard_card"

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
            return f"Scrap card: {self.card_id} from {self.source}"
        elif self.type == ActionType.END_TURN:
            return "End turn"
        return f"{self.type}"

def get_available_actions(game_state: 'Game', player: 'Player') -> List[Action]:
    """Return list of available actions for a player given the current game state"""
    actions = []

    # If there are any pending actions, those are the only actions that a player can do right now
    if len(player.pending_actions) > 0:
        for action in player.pending_actions:
            actions.append(action)
        return actions

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
                # First check for outposts - must be destroyed before attacking other bases or player
                outposts = [b for b in opponent.bases if b.is_outpost()]
                if outposts:
                    # Can only attack outposts
                    for outpost in outposts:
                        if player.combat >= outpost.defense:
                            actions.append(Action(
                                type=ActionType.ATTACK_BASE,
                                target_id=outpost.name,
                                additional_params={"defense": outpost.defense}
                            ))
                else:
                    # If no outposts, can attack other bases or player directly
                    for base in [b for b in opponent.bases if not b.is_outpost()]:
                        if base.defense and player.combat >= base.defense:
                            actions.append(Action(
                                type=ActionType.ATTACK_BASE,
                                target_id=base.name,
                                additional_params={"defense": base.defense}
                            ))
                    # Can attack player directly only if no outposts
                    actions.append(Action(
                        type=ActionType.ATTACK_PLAYER,
                        target_id=opponent.name,
                        additional_params={"damage": player.combat}
                    ))
    
    # Add scrap card actions for played cards
    for card in player.played_cards:
        if card.effects and any("\{Scrap\}" in effect for effect in card.effects):
            # Only allow scrapping cards that have a scrap effect
            actions.append(Action(type=ActionType.SCRAP_CARD, card_id=card.name, source="played"))

    # Always allow ending turn
    actions.append(Action(type=ActionType.END_TURN))
    
    return actions