from enum import Enum
import re
from typing import TYPE_CHECKING, List, Optional
from dataclasses import dataclass

if TYPE_CHECKING:
    from src.engine.player import Player
    from src.engine.game import Game

class CardEffectType(Enum):
    COMBAT = "combat"
    TRADE = "trade"
    DRAW = "draw"
    HEAL = "heal"
    SCRAP = "scrap"  # For effects that allow scrapping other cards
    PARENT = "parent"  # For effects that are parent effects
    TARGET_DISCARD = "target_discard"  # For effects that make the target player discard cards
    DESTROY_BASE = "destroy_base"  # For effects that destroy bases
    COMPLEX = "complex"  # For complex effects that require special handling

class CardTargetType(Enum):
    HAND = "hand"
    DISCARD = "discard"
    TRADE = "trade"

@dataclass
class Effect:
    effect_type: CardEffectType
    value: int = 0
    text: str = ""
    faction_requirement: Optional[str] = None
    is_scrap_effect: bool = False
    is_ally_effect: bool = False
    is_or_effect: bool = False
    faction_requirement_count: int = 0
    child_effects: Optional[List['Effect']] = None
    card_targets: Optional[List[str]] = None
    
    def __init__(self, effect_type: CardEffectType, value: int = 0, text: str = "", 
                 faction_requirement: Optional[str] = None, is_scrap_effect: bool = False,
                 is_ally_effect: bool = False, faction_requirement_count: int = 0, is_or_effect: bool = False, 
                 child_effects: Optional[List['Effect']] = None, card_targets: Optional[List[str]] = None):
        self.effect_type = effect_type
        self.value = value
        self.text = text
        self.faction_requirement = faction_requirement
        self.is_scrap_effect = is_scrap_effect
        self.is_ally_effect = is_ally_effect
        self.is_or_effect = is_or_effect
        self.faction_requirement_count = faction_requirement_count if faction_requirement_count > 0 else (1 if faction_requirement else 0)
        self.applied = False
        self.child_effects = child_effects
        self.card_targets = card_targets

    def any_child_effects_used(self):
        """Check if any child effects have been used"""
        if self.child_effects:
            for effect in self.child_effects:
                if effect.applied:
                    return True
        return False
    
    def all_child_effects_used(self):
        """Check if all child effects have been used"""
        if self.child_effects:
            for effect in self.child_effects:
                if not effect.all_child_effects_used():
                    return False
            return True
        return self.applied
    
    def apply(self, game: 'Game', player: 'Player', card=None):
        from src.engine.actions import Action, ActionType
        # If it has already been applied, do nothing
        if self.applied:
            return
        
        # Check if the player meets the faction requirement
        if self.faction_requirement and player.get_faction_ally_count(self.faction_requirement) <= self.faction_requirement_count:
            return
            
        if self.effect_type == CardEffectType.COMBAT:
            player.combat += self.value
        elif self.effect_type == CardEffectType.TRADE:
            player.trade += self.value
            game.stats.record_trade(player.name, self.value)
        elif self.effect_type == CardEffectType.DRAW:
            for _ in range(self.value):
                player.draw_card()
                game.stats.record_card_draw(player.name)
        elif self.effect_type == CardEffectType.HEAL:
            player.health += self.value
            game.stats.record_authority_gain(player.name, self.value)
        elif self.effect_type == CardEffectType.SCRAP:
            pending_actions = []
            if self.card_targets and "discard" in self.card_targets:
                for target in player.discard_pile:
                    pending_actions.append(Action(
                        ActionType.SCRAP_CARD,
                        card_id=target.name,
                        card_source="discard"
                    ))
            if self.card_targets and "hand" in self.card_targets:
                for target in player.hand:
                    pending_actions.append(Action(
                        ActionType.SCRAP_CARD,
                        card_id=target.name,
                        card_source="hand"
                    ))
            if self.card_targets and "trade" in self.card_targets:
                for target in game.trade_row:
                    pending_actions.append(Action(
                        ActionType.SCRAP_CARD,
                        card_id=target.name,
                        card_source="trade"
                    ))
            if pending_actions:
                player.add_pending_actions(pending_actions, self.value, False)
        elif self.effect_type == CardEffectType.TARGET_DISCARD:
            # Create an action for the target player to discard a card
            if self.card_targets and "opponent" in self.card_targets:
                # Assuming the opponent is the next player in the game
                opponent = game.get_opponent(player)
                if opponent is None:
                    return
                # Create an action for each card in the opponent's hand
                pending_actions = []
                for target in opponent.hand:
                    action = Action(
                        ActionType.DISCARD_CARDS,
                        card_id=target.name,
                        card_source="opponent"
                    )
                    pending_actions.append(action)
                opponent.add_pending_actions(pending_actions, self.value, True)
        elif self.effect_type == CardEffectType.DESTROY_BASE:
            opponent = game.get_opponent(player)
            if opponent is None:
                return
            pending_actions = []
            outposts = [b for b in opponent.bases if b.is_outpost()]
            if outposts:
                for outpost in outposts:
                    pending_actions.append(Action(
                        ActionType.DESTROY_BASE,
                        target_id=outpost.name,
                        card_id=outpost.name,
                    ))
            else:
                for base in opponent.bases:
                    pending_actions.append(Action(
                        ActionType.DESTROY_BASE,
                        target_id=base.name,
                        card_id=base.name,
                    ))
            if pending_actions:
                opponent.add_pending_actions(pending_actions, self.value, False)
                
        elif self.effect_type == CardEffectType.PARENT:
            # Apply child effects if this card has any
            if self.child_effects:
                for effect in self.child_effects:
                    effect.apply(game, player, card)
        elif self.effect_type == CardEffectType.COMPLEX:
            self.handle_complex_effect(game, player, card)
        
        # parent effects should only be marked as applied if it is an OR effect
        if self.effect_type != CardEffectType.PARENT:
            self.applied = True

        # if the effect is a scrap effect, remove the card from the game
        if self.is_scrap_effect and card:
            game.stats.record_card_scrap(player.name, "card")
            # find the card in the player's played cards by name and remove it
            for c in player.played_cards:
                if c.name == card.name:
                    player.played_cards.remove(c)
                    break

    def handle_complex_effect(self, game: 'Game', player: 'Player', card):
        # Handle conditional card draw
        draw_match = re.search(r"Draw a card for each (\w+) card", self.text)
        if draw_match:
            faction = draw_match.group(1).lower()
            count = sum(1 for c in player.played_cards if c.faction and c.faction.lower() == faction)
            for _ in range(count):
                player.draw_card()

    def reset(self):
        """Reset the effect's applied status at the end of turn"""
        self.applied = False
        
    def __str__(self):
        base = f"{self.effect_type.name.capitalize()}: "
        if self.is_scrap_effect:
            base = f"(Scrap required): {base}"
        if self.child_effects:
            base += "[ Child Effects: " + (" OR " if self.is_or_effect else ", ").join(str(effect) for effect in self.child_effects) + " ] "
            return base
        base += f"{self.value}" if self.value else self.text
        base += f" from {self.card_targets}" if self.card_targets else ""
        if self.is_ally_effect and self.faction_requirement:
            base = f"{self.faction_requirement} Ally: {base}"
        return base