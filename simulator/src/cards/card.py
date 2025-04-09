class Card:
    def __init__(self, name, cost, effects, card_type="ship", defense=None, faction=None):
        self.name = name
        self.cost = cost
        self.effects = effects
        self.card_type = card_type  # "ship", "base", or "outpost"
        self.defense = defense  # Only used for bases and outposts
        self.faction = faction  # Can be None (unaligned), a string, or a list of factions
        
    def is_base(self):
        return self.card_type in ["base", "outpost"]
        
    def is_outpost(self):
        return self.card_type == "outpost"

    def apply_effects(self, game_state):
        for effect in self.effects:
            effect.apply(game_state)

    def __str__(self):
        info = [f"{self.name} ({self.cost} cost)"]
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