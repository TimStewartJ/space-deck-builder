import re
from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass

if TYPE_CHECKING:
    from src.engine.player import Player

@dataclass
class Effect:
    effect_type: str
    value: int = 0
    text: str = ""
    faction_requirement: Optional[str] = None
    is_scrap_effect: bool = False
    is_ally_effect: bool = False
    faction_requirement_count: int = 0
    
    def __init__(self, effect_type: str, value: int = 0, text: str = "", 
                 faction_requirement: Optional[str] = None, is_scrap_effect: bool = False,
                 is_ally_effect: bool = False, faction_requirement_count: int = 0):
        self.effect_type = effect_type
        self.value = value
        self.text = text
        self.faction_requirement = faction_requirement
        self.is_scrap_effect = is_scrap_effect
        self.is_ally_effect = is_ally_effect
        self.faction_requirement_count = faction_requirement_count if faction_requirement_count > 0 else (1 if faction_requirement else 0)
        self.applied = False
    
    def apply(self, player: 'Player', card=None):
        if self.applied:
            return
            
        if self.effect_type == "combat":
            player.combat += self.value
        elif self.effect_type == "trade":
            player.trade += self.value
        elif self.effect_type == "draw":
            for _ in range(self.value):
                player.draw_card()
        elif self.effect_type == "heal":
            player.health += self.value
        elif self.effect_type == "complex":
            self.handle_complex_effect(player, card)
        
        self.applied = True

    def handle_complex_effect(self, player: 'Player', card):
        # Handle conditional card draw
        draw_match = re.search(r"Draw a card for each (\w+) card", self.text)
        if draw_match:
            faction = draw_match.group(1).lower()
            count = sum(1 for c in player.played_cards if c.faction.lower() == faction)
            for _ in range(count):
                player.draw_card()
        
        """Create appropriate actions for effects requiring player decisions"""
        from src.engine.actions import Action, ActionType
        
        if "scrap a card in your hand or discard pile":
            discard_targets = [c.name for c in player.discard_pile]
            hand_targets = [c.name for c in player.hand]

            for target in discard_targets:
                action = Action(
                    ActionType.SCRAP_CARD,
                    card_id=target,
                    source=["discard"]
                )
                player.pending_actions.append(action)

            for target in hand_targets:
                action = Action(
                    ActionType.SCRAP_CARD,
                    card_id=target,
                    source=["hand"]
                )
                player.pending_actions.append(action)
    
    def reset(self):
        """Reset the effect's applied status at the end of turn"""
        self.applied = False
        
    def __str__(self):
        base = f"{self.effect_type.capitalize()}: "
        base += f"{self.value}" if self.value else self.text
        if self.is_scrap_effect:
            base = f"Scrap: {base}"
        if self.is_ally_effect and self.faction_requirement:
            base = f"{self.faction_requirement} Ally: {base}"
        return base