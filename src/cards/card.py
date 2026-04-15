from typing import List, Optional
from src.cards.effects import Effect


class Card:
    def __init__(self, name, index, cost, effects: List[Effect], card_type="ship",
                 defense=None, faction=None, set=None, ally_factions=None):
        self.name: str = name
        self.index: int = index
        self.cost: int = cost
        self.effects: List[Effect] = effects
        self.card_type = card_type  # "ship", "base", or "outpost"
        self.defense: Optional[int] = defense  # Only used for bases and outposts
        self.faction = faction  # Can be None (unaligned), a string, or a list of factions
        self.set: str | None = set  # The set the card comes from (e.g. "Core Set", "Colony Wars", etc.)
        # Override for ally-counting: None=use faction, ["*"]=all factions, ["X","Y"]=specific
        self.ally_factions: list[str] | None = ally_factions
        
    def is_base(self):
        return self.card_type in ["base", "outpost"]
        
    def is_outpost(self):
        return self.card_type == "outpost"
            
    def clone(self) -> 'Card':
        """Create a lightweight copy with fresh effect state.
        
        Shares immutable fields (name, cost, etc.) by reference but creates
        new Effect objects so mutable applied-state is independent per game.
        """
        return Card(
            name=self.name,
            index=self.index,
            cost=self.cost,
            effects=[e.clone() for e in self.effects],
            card_type=self.card_type,
            defense=self.defense,
            faction=list(self.faction) if isinstance(self.faction, list) else self.faction,
            set=self.set,
            ally_factions=list(self.ally_factions) if self.ally_factions else None,
        )

    def reset_effects(self):
        """Reset all effects at end of turn"""
        for effect in self.effects:
            effect.reset()

    def __str__(self):
        info = [f"{self.set} {self.name} ({self.cost} cost)"]
        if self.faction:
            faction_str = self.faction if isinstance(self.faction, str) else "/".join(self.faction)
            info.append(f"Faction: {faction_str}")
        if self.ally_factions:
            ally_str = "all" if "*" in self.ally_factions else "/".join(self.ally_factions)
            info.append(f"Ally for: {ally_str}")
        if self.defense is not None:
            info.append(f"Defense: {self.defense}")
        info.append(f"Type: {self.card_type}")
        effects_str = ", ".join(str(effect) for effect in self.effects)
        if effects_str:
            info.append(f"Effects: {effects_str}")
        return " | ".join(info)