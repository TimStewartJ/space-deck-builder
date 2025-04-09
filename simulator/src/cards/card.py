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