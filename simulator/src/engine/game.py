class Game:
    def __init__(self, cards=None):
        self.players = []
        self.current_turn = 0
        self.is_game_over = False
        self.trade_deck = cards if cards else []
        self.trade_row = []
        self.is_running = False

    def start_game(self):
        if not self.trade_deck:
            raise ValueError("Cannot start game without cards")
        self.setup_players()
        self.setup_trade_row()
        self.is_game_over = False
        self.is_running = True
        self.next_turn()

    def setup_trade_row(self):
        self.shuffle_trade_deck()
        # Fill the trade row with 5 cards
        while len(self.trade_row) < 5 and self.trade_deck:
            self.trade_row.append(self.trade_deck.pop())

    def shuffle_trade_deck(self):
        import random
        random.shuffle(self.trade_deck)

    def setup_players(self):
        for player in self.players:
            # Initialize each player's starting deck with 8 Scouts and 2 Vipers
            starting_deck = self.create_starting_deck()
            player.deck.extend(starting_deck)
            player.shuffle_deck()
            # Draw initial hand
            for _ in range(5):
                player.draw_card()

    def create_starting_deck(self):
        # Create a deck of 8 Scouts and 2 Vipers
        from src.cards.card import Card
        starting_deck = []
        # Add 8 Scouts
        for _ in range(8):
            starting_deck.append(Card("Scout", 0, ["Gain 1 Trade"], "ship"))
        # Add 2 Vipers
        for _ in range(2):
            starting_deck.append(Card("Viper", 0, ["Gain 1 Combat"], "ship"))
        return starting_deck

    def next_turn(self):
        if not self.is_game_over:
            current_player = self.players[self.current_turn]
            # Logic for the current player's turn
            self.current_turn = (self.current_turn + 1) % len(self.players)

    def end_game(self):
        self.is_game_over = True
        self.is_running = False
        # Logic to determine the winner and end the game
        pass

    def add_player(self):
        if len(self.players) < 4:  # Maximum 4 players
            from src.engine.player import Player
            player_number = len(self.players) + 1
            player = Player(f"Player {player_number}")
            self.players.append(player)
            return player
        else:
            raise ValueError("Maximum number of players reached")