# factions.py

class Faction:
    def __init__(self, name, color, abilities):
        self.name = name
        self.color = color
        self.abilities = abilities

    def get_info(self):
        return {
            "name": self.name,
            "color": self.color,
            "abilities": self.abilities
        }

# Define factions
trade_federation = Faction("Trade Federation", "blue", ["gain_trade", "heal"])
star_empire = Faction("Star Empire", "yellow", ["draw_card", "cycle"])
blob = Faction("Blob", "green", ["gain_combat", "destroy"])
machine_cult = Faction("Machine Cult", "red", ["scrap", "gain_trade"])

# List of all factions
factions = [trade_federation, star_empire, blob, machine_cult]

def get_faction_by_name(name):
    for faction in factions:
        if faction.name.lower() == name.lower():
            return faction
    return None