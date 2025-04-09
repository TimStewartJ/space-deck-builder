class GameState:
    def __init__(self, players, cards_in_play):
        self.players = players
        self.cards_in_play = cards_in_play

    def update_display(self):
        # Logic to update the UI display with the current game state
        pass

    def update_players(self, players):
        self.players = players

    def update_cards_in_play(self, cards_in_play):
        self.cards_in_play = cards_in_play

    def get_game_info(self):
        return {
            "players": self.players,
            "cards_in_play": self.cards_in_play
        }