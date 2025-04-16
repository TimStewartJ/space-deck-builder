from typing import List, Optional
from src.cards.effects import Effect


class Card:
    def __init__(self, name, index, cost, effects: List[Effect], card_type="ship", defense=None, faction=None, set=None):
        self.name: str = name
        self.index: int = index
        self.cost: int = cost
        self.effects: List[Effect] = effects
        self.card_type = card_type  # "ship", "base", or "outpost"
        self.defense: Optional[int] = defense  # Only used for bases and outposts
        self.faction = faction  # Can be None (unaligned), a string, or a list of factions
        self.set: str | None = set  # The set the card comes from (e.g. "Core Set", "Colony Wars", etc.)
        
    def is_base(self):
        return self.card_type in ["base", "outpost"]
        
    def is_outpost(self):
        return self.card_type == "outpost"
            
    def reset_effects(self):
        """Reset all effects at end of turn"""
        for effect in self.effects:
            effect.reset()

    def __str__(self):
        info = [f"{self.set} {self.name} ({self.cost} cost)"]
        if self.faction:
            faction_str = self.faction if isinstance(self.faction, str) else "/".join(self.faction)
            info.append(f"Faction: {faction_str}")
        if self.defense is not None:
            info.append(f"Defense: {self.defense}")
        info.append(f"Type: {self.card_type}")
        effects_str = ", ".join(str(effect) for effect in self.effects)
        if effects_str:
            info.append(f"Effects: {effects_str}")
        return " | ".join(info)