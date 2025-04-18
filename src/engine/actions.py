from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, List, Any
from enum import Enum

if TYPE_CHECKING:
    from src.engine.player import Player
    from src.engine.game import Game
    from src.cards.card import Card
    from src.cards.effects import Effect

class ActionType(Enum):
    PLAY_CARD = "play_card"
    APPLY_EFFECT = "apply_effect"
    BUY_CARD = "buy_card"
    ATTACK_BASE = "attack_base"
    ATTACK_PLAYER = "attack_player"
    SCRAP_CARD = "scrap_card"
    END_TURN = "end_turn"
    DISCARD_CARDS = "discard_card"
    DESTROY_BASE = "destroy_base"
    SKIP_DECISION = "skip_decision"

@dataclass
class Action:
    type: ActionType
    card_id: Optional[str] = None
    card: Optional['Card'] = None
    target_id: Optional[str] = None
    card_source: Optional[str] = None
    card_effect: Optional['Effect'] = None
    additional_params: Optional[dict] = None
    
    def __str__(self):
        """String representation for display in CLI"""
        if self.type == ActionType.PLAY_CARD and self.card:
            return f"Play card: {self.card}"
        elif self.type == ActionType.APPLY_EFFECT and self.card_id:
            return f"{self.card_id}: {self.card_effect}"
        elif self.type == ActionType.BUY_CARD and self.card_id:
            return f"Buy card: {self.card}"
        elif self.type == ActionType.ATTACK_BASE:
            return f"Attack base: {self.target_id}"
        elif self.type == ActionType.ATTACK_PLAYER:
            return f"Attack player: {self.target_id}"
        elif self.type == ActionType.SCRAP_CARD:
            return f"Scrap card: {self.card_id} from {self.card_source}"
        elif self.type == ActionType.DISCARD_CARDS:
            return f"Discard cards: {self.card_id} from {self.card_source}"
        elif self.type == ActionType.DESTROY_BASE:
            return f"Destroy base: {self.target_id}"
        elif self.type == ActionType.END_TURN:
            return "End turn"
        return f"{self.type}"

@dataclass
class PendingActionSet:
    actions: List[Action]
    decisions_left: int
    mandatory: bool

def get_available_actions(game_state: 'Game', player: 'Player') -> List[Action]:
    """Return list of available actions for a player given the current game state"""
    actions = []

    # If it's not the player's turn, return empty list
    if game_state.current_player != player:
        return actions

    # If there are pending action sets, handle current set
    pending_set: Optional[PendingActionSet] = player.get_current_pending_set()
    if pending_set:
        for act in pending_set.actions:
            actions.append(act)
        # If the pending actions are optional, add a skip action
        if not pending_set.mandatory:
            actions.append(Action(type=ActionType.SKIP_DECISION))
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
                        if outpost.defense and player.combat >= outpost.defense:
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
            if not effect.all_child_effects_used():
                # Ensure faction allies are valid
                if effect.faction_requirement:
                    faction_count = player.get_faction_ally_count(effect.faction_requirement)
                    if faction_count > effect.faction_requirement_count:
                        actions.append(Action(type=ActionType.APPLY_EFFECT, card_id=card.name, card_effect=effect))
                elif effect.is_or_effect and effect.child_effects and not effect.any_child_effects_used():
                    # If it's an OR effect that hasn't been used yet, add all child effects
                    for child_effect in effect.child_effects:
                        actions.append(Action(type=ActionType.APPLY_EFFECT, card_id=card.name, card_effect=child_effect))
                else:
                    actions.append(Action(type=ActionType.APPLY_EFFECT, card_id=card.name, card_effect=effect))

    # Always allow ending turn
    actions.append(Action(type=ActionType.END_TURN))
    
    return actions