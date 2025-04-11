from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, List, Any
from enum import Enum

if TYPE_CHECKING:
    from src.engine.player import Player
    from src.engine.game import Game
    from src.cards.card import Card

class ActionType(Enum):
    PLAY_CARD = "play_card"
    APPLY_EFFECT = "apply_effect"
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
    card: Optional['Card'] = None
    target_id: Optional[str] = None
    card_sources: Optional[str] = None
    additional_params: Optional[dict] = None
    
    def __str__(self):
        """String representation for display in CLI"""
        if self.type == ActionType.PLAY_CARD and self.card:
            return f"Play card: {self.card}"
        elif self.type == ActionType.APPLY_EFFECT and self.card_id:
            return f"{self.card_id}: {self.additional_params.get('effect', '')}"
        elif self.type == ActionType.BUY_CARD and self.card_id:
            return f"Buy card: {self.card}"
        elif self.type == ActionType.ATTACK_BASE:
            return f"Attack base: {self.target_id}"
        elif self.type == ActionType.ATTACK_PLAYER:
            return f"Attack player: {self.target_id}"
        elif self.type == ActionType.SCRAP_CARD:
            return f"Scrap card: {self.card_id} from {self.card_sources}"
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
        actions.append(Action(type=ActionType.PLAY_CARD, card=card, card_id=card.name))
    
    # Add buy card actions for affordable cards in trade row
    for card in game_state.trade_row:
        if player.trade >= card.cost:
            actions.append(Action(type=ActionType.BUY_CARD, card=card, card_id=card.name))

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
                                target_id=outpost.name
                            ))
                else:
                    # If no outposts, can attack other bases or player directly
                    for base in [b for b in opponent.bases if not b.is_outpost()]:
                        if base.defense and player.combat >= base.defense:
                            actions.append(Action(
                                type=ActionType.ATTACK_BASE,
                                target_id=base.name
                            ))
                    # Can attack player directly only if no outposts
                    actions.append(Action(
                        type=ActionType.ATTACK_PLAYER,
                        target_id=opponent.name
                    ))
    
    # Add unused actions from played cards
    for card in player.played_cards:
        for effect in card.effects:
            if not effect.applied:
                # Ensure faction allies are valid
                if effect.faction_requirement:
                    faction_count = player.get_faction_ally_count(effect.faction_requirement)
                    if faction_count > effect.faction_requirement_count:
                        actions.append(Action(type=ActionType.APPLY_EFFECT, card_id=card.name, additional_params={"effect": effect}))
                    continue
                # If no faction requirement, add effect directly
                actions.append(Action(type=ActionType.APPLY_EFFECT, card_id=card.name, additional_params={"effect": effect}))

    # Always allow ending turn
    actions.append(Action(type=ActionType.END_TURN))
    
    return actions