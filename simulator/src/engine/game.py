from typing import List
from src.engine.player import Player
from src.engine.actions import ActionType, Action, get_available_actions

class Game:
    def __init__(self, cards=None):
        self.players: List[Player] = []
        self.current_turn = 0
        self.is_game_over = False
        self.trade_deck = cards if cards else []
        self.trade_row = []
        self.is_running = False
        self.current_player: Player = None

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
        if self.is_game_over:
            return
        
        self.current_player = self.players[self.current_turn]
        self.current_player.reset_resources()
        
        # Process player turn until they choose to end it
        self.process_player_turn()
        
        # End turn and move to next player
        self.current_player.end_turn()
        self.current_turn = (self.current_turn + 1) % len(self.players)
        
    def process_player_turn(self):
        turn_ended = False
        
        while not turn_ended:
            # Get player decision (through UI or AI)
            action = self.current_player.make_decision(self)
            
            if action:
                turn_ended = self.execute_action(action)
    
    def execute_action(self, action):
        """Execute a player's action and update game state"""
        if action.type == ActionType.END_TURN:
            return True  # End the turn
            
        elif action.type == ActionType.PLAY_CARD:
            # Find the card in player's hand
            for card in self.current_player.hand:
                if card.name == action.card_id:
                    self.current_player.play_card(card)
                    self.apply_card_effects(card)
                    break
                    
        elif action.type == ActionType.BUY_CARD:
            # Find the card in trade row
            for i, card in enumerate(self.trade_row):
                if card.name == action.card_id and self.current_player.trade >= card.cost:
                    self.current_player.trade -= card.cost
                    self.current_player.discard_pile.append(card)
                    self.trade_row.pop(i)
                    # Replace card in trade row
                    if self.trade_deck:
                        self.trade_row.append(self.trade_deck.pop())
                    break
                    
        # ... similar logic for other action types ...
        
        return False  # Turn continues
    
    def apply_card_effects(self, card):
        """Apply the effects of a played card"""
        for effect in card.effects:
            # Parse effect string and apply it
            if "Gain 1 Trade" in effect:
                self.current_player.trade += 1
            elif "Gain 2 Trade" in effect:
                self.current_player.trade += 2
            elif "Gain 1 Combat" in effect:
                self.current_player.combat += 1
            # Add more effect parsing as needed

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