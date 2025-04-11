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
    faction_requirement_count: int = 0
    child_effects: Optional[List['Effect']] = None
    card_targets: Optional[List[str]] = None
    
    def __init__(self, effect_type: CardEffectType, value: int = 0, text: str = "", 
                 faction_requirement: Optional[str] = None, is_scrap_effect: bool = False,
                 is_ally_effect: bool = False, faction_requirement_count: int = 0, child_effects: Optional[List['Effect']] = None,
                 card_targets: Optional[List[str]] = None):
        self.effect_type = effect_type
        self.value = value
        self.text = text
        self.faction_requirement = faction_requirement
        self.is_scrap_effect = is_scrap_effect
        self.is_ally_effect = is_ally_effect
        self.faction_requirement_count = faction_requirement_count if faction_requirement_count > 0 else (1 if faction_requirement else 0)
        self.applied = False
        self.child_effects = child_effects
        self.card_targets = card_targets
    
    def apply(self, game: 'Game', player: 'Player', card=None):
        if self.applied:
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
            from src.engine.actions import Action, ActionType
            # Create an action for every card in discard pile
            if self.card_targets and "discard" in self.card_targets:
                discard_targets = player.discard_pile
                for target in discard_targets:
                    action = Action(
                        ActionType.SCRAP_CARD,
                        card_id=target,
                        source=["discard"]
                    )
                    player.pending_actions.append(action)
                    game.stats.record_card_scrap(player.name, "discard")
            # Create an action for every card in hand
            if self.card_targets and "hand" in self.card_targets:
                hand_targets = player.hand
                for target in hand_targets:
                    action = Action(
                        ActionType.SCRAP_CARD,
                        card_id=target,
                        source=["hand"]
                    )
                    player.pending_actions.append(action)
                    game.stats.record_card_scrap(player.name, "hand")
            # Create an action for every card in trade row
            if self.card_targets and "trade" in self.card_targets:
                trade_targets = game.trade_row
                for target in trade_targets:
                    action = Action(
                        ActionType.SCRAP_CARD,
                        card_id=target,
                        source=["trade"]
                    )
                    player.pending_actions.append(action)
                    game.stats.record_card_scrap(player.name, "trade")
        elif self.effect_type == CardEffectType.COMPLEX:
            self.handle_complex_effect(game, player, card)
        
        self.applied = True

        # if the effect is a scrap effect, remove the card from the game
        if self.is_scrap_effect and card:
            # find the card in the player's played cards by name and remove it
            for c in player.played_cards:
                if c.name == card.name:
                    player.played_cards.remove(c)
                    break

    def handle_complex_effect(self, game: 'Game', player: 'Player', card):
        if self.child_effects:
            for effect in self.child_effects:
                effect.apply(game, player, card)
            return

        # Handle conditional card draw
        draw_match = re.search(r"Draw a card for each (\w+) card", self.text)
        if draw_match:
            faction = draw_match.group(1).lower()
            count = sum(1 for c in player.played_cards if c.faction.lower() == faction)
            for _ in range(count):
                player.draw_card()

    def reset(self):
        """Reset the effect's applied status at the end of turn"""
        self.applied = False
        
    def __str__(self):
        base = f"{self.effect_type.name.capitalize()}: "
        if self.child_effects:
            base += " | Child Effects: " + ", ".join(str(effect) for effect in self.child_effects)
            return base
        base += f"{self.value}" if self.value else self.text
        base += f" from {self.card_targets}" if self.card_targets else ""
        if self.is_scrap_effect:
            base = f"Scrap: {base}"
        if self.is_ally_effect and self.faction_requirement:
            base = f"{self.faction_requirement} Ally: {base}"
        return base