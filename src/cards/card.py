from enum import Enum
from typing import List, Optional
from src.cards.effects import Effect
from src.cards.factions import Faction, faction_display, parse_faction


class CardType(Enum):
    SHIP = "ship"
    BASE = "base"
    OUTPOST = "outpost"


class Card:
    def __init__(self, name, index, cost, effects: List[Effect], card_type=CardType.SHIP,
                 defense=None, faction=None, set=None, ally_factions=None):
        self.name: str = name
        self.index: int = index
        self.cost: int = cost
        self.effects: List[Effect] = effects
        # Accept string for backward compatibility, normalize to CardType enum
        if isinstance(card_type, str):
            card_type = CardType(card_type)
        self.card_type: CardType = card_type
        self.defense: Optional[int] = defense  # Only used for bases and outposts
        # Faction bitmask: Faction.NONE for unaligned, single bit for one faction,
        # OR'd bits for multi-faction cards. Accepts string/list for backward compat.
        if faction is None:
            self.faction: Faction = Faction.NONE
        elif isinstance(faction, Faction):
            self.faction = faction
        elif isinstance(faction, list):
            result = Faction.NONE
            for f in faction:
                result |= f if isinstance(f, Faction) else parse_faction(f)
            self.faction = result
        elif isinstance(faction, str):
            self.faction = parse_faction(faction)
        else:
            self.faction = Faction(faction)
        self.set: str | None = set  # The set the card comes from (e.g. "Core Set", "Colony Wars", etc.)
        # Override for ally-counting: Faction.NONE=use card faction, Faction.ALL=all factions
        # Accepts list of strings or "*" for backward compat.
        if ally_factions is None:
            self.ally_factions: Faction = Faction.NONE
        elif isinstance(ally_factions, Faction):
            self.ally_factions = ally_factions
        elif isinstance(ally_factions, list):
            if "*" in ally_factions:
                self.ally_factions = Faction.ALL
            else:
                result = Faction.NONE
                for f in ally_factions:
                    result |= f if isinstance(f, Faction) else parse_faction(f)
                self.ally_factions = result
        elif isinstance(ally_factions, str):
            self.ally_factions = parse_faction(ally_factions)
        else:
            self.ally_factions = Faction(ally_factions)
        
    def is_base(self):
        return self.card_type in (CardType.BASE, CardType.OUTPOST)
        
    def is_outpost(self):
        return self.card_type == CardType.OUTPOST
            
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
            faction=self.faction,
            set=self.set,
            ally_factions=self.ally_factions,
        )

    def reset_effects(self):
        """Reset all effects at end of turn"""
        for effect in self.effects:
            effect.reset()

    def __str__(self):
        info = [f"{self.set} {self.name} ({self.cost} cost)"]
        if self.faction:
            info.append(f"Faction: {faction_display(self.faction)}")
        if self.ally_factions:
            info.append(f"Ally for: {faction_display(self.ally_factions)}")
        if self.defense is not None:
            info.append(f"Defense: {self.defense}")
        info.append(f"Type: {self.card_type.value}")
        effects_str = ", ".join(str(effect) for effect in self.effects)
        if effects_str:
            info.append(f"Effects: {effects_str}")
        return " | ".join(info)