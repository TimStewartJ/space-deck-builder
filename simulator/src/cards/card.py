class Card:
    def __init__(self, name, cost, effects):
        self.name = name
        self.cost = cost
        self.effects = effects

    def apply_effects(self, game_state):
        for effect in self.effects:
            effect.apply(game_state)