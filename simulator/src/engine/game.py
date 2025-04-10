from typing import List
from src.engine.card_effects import CardEffects
from src.cards.card import Card
from src.engine.player import Player
from src.engine.actions import ActionType, Action, get_available_actions

class Game:
    def __init__(self, cards=None, verbose=False):
        self.players: List[Player] = []
        self.current_turn = 0
        self.is_game_over = False
        self.trade_deck = cards if cards else []
        self.trade_row = []
        self.is_running = False
        self.current_player: Player = None
        self.verbose = verbose
        self.card_effects = CardEffects()

    def log(self, message):
        if self.verbose:
            print(f"[Game] {message}")

    def start_game(self):
        if not self.trade_deck:
            raise ValueError("Cannot start game without cards")
        self.setup_players()
        self.setup_trade_row()
        self.is_game_over = False
        self.is_running = True
        self.current_player = self.players[self.current_turn]

    def setup_trade_row(self):
        self.shuffle_trade_deck()
        # Fill the trade row with 5 cards
        while len(self.trade_row) < 5 and self.trade_deck:
            card = self.trade_deck.pop()
            self.trade_row.append(card)
            self.log(f"Added {card.name} to trade row")

    def shuffle_trade_deck(self):
        import random
        random.shuffle(self.trade_deck)

    def setup_players(self):
        # Shuffle players to randomize turn order
        import random
        random.shuffle(self.players)

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
            starting_deck.append(Card("Scout", 0, ["{Gain 1 Trade}"], "ship"))
        # Add 2 Vipers
        for _ in range(2):
            starting_deck.append(Card("Viper", 0, ["{Gain 1 Combat}"], "ship"))
        return starting_deck

    def next_turn(self):
        if self.is_game_over:
            return
        
        self.current_player = self.players[self.current_turn]
        self.current_player.reset_resources()
        
        self.log(f"Starting turn for {self.current_player.name}")
        
        # Process player turn until they choose to end it
        self.process_player_turn()
        
        # End turn and move to next player
        self.current_player.end_turn()
        self.log(f"Ended turn for {self.current_player.name}")
        self.current_turn = (self.current_turn + 1) % len(self.players)
        
    def process_player_turn(self):
        turn_ended = False
        
        while not turn_ended:
            # Get player decision (through UI or AI)
            action = self.current_player.make_decision(self)
            
            if action:
                turn_ended = self.execute_action(action)
    
    def execute_action(self, action: Action):
        """Execute a player's action and update game state"""
        self.log(f"{self.current_player.name} executing action: {action}")

        # if there are any pending actions left, clear them since we just did one
        self.current_player.pending_actions.clear()
        
        if action.type == ActionType.END_TURN:
            return True  # End the turn
            
        elif action.type == ActionType.PLAY_CARD:
            # Find the card in player's hand
            for card in self.current_player.hand:
                if card.name == action.card_id:
                    self.current_player.play_card(card)
                    self.log(f"{self.current_player.name} played {card.name}")
                    self.card_effects.apply_card_effects(current_player=self.current_player, card=card)
                    break
                    
        elif action.type == ActionType.BUY_CARD:
            # Find the card in trade row
            for i, card in enumerate(self.trade_row):
                if card.name == action.card_id and self.current_player.trade >= card.cost:
                    self.current_player.trade -= card.cost
                    self.current_player.discard_pile.append(card)
                    self.log(f"{self.current_player.name} bought {card.name} for {card.cost} trade")
                    self.trade_row.pop(i)
                    # Replace card in trade row
                    if self.trade_deck:
                        new_card = self.trade_deck.pop()
                        self.trade_row.append(new_card)
                        self.log(f"Added {new_card.name} to trade row")
                    break
        
        elif action.type == ActionType.ATTACK_BASE:
            # Find target base
            for player in self.players:
                if player != self.current_player:
                    for base in player.bases:
                        if base.name == action.target_id and self.current_player.combat >= base.defense:
                            self.current_player.combat -= base.defense
                            player.bases.remove(base)
                            player.discard_pile.append(base)
                            self.log(f"{self.current_player.name} destroyed {player.name}'s {base.name}")
                            break

        elif action.type == ActionType.ATTACK_PLAYER:
            # Attack player directly
            for player in self.players:
                if player != self.current_player and player.name == action.target_id:
                    # Only allow attacking if no outposts present
                    if not any(b.is_outpost() for b in player.bases):
                        damage = self.current_player.combat
                        player.health -= damage
                        self.current_player.combat = 0
                        self.log(f"{self.current_player.name} attacked {player.name} for {damage} damage")
                        # Check for game over
                        if player.health <= 0:
                            self.log(f"{player.name} has been defeated!")
                            self.is_game_over = True
                        break
        
        elif action.type == ActionType.SCRAP_CARD:
            if action.source and 'hand' in action.source:
                # Scrap card from hand
                for card in self.current_player.hand:
                    if card.name == action.card_id:
                        self.current_player.hand.remove(card)
                        self.log(f"{self.current_player.name} scrapped {card.name} from hand")
                        break
            elif action.source and 'discard' in action.source:
                # Scrap card from discard pile
                for card in self.current_player.discard_pile:
                    if card.name == action.card_id:
                        self.current_player.discard_pile.remove(card)
                        self.log(f"{self.current_player.name} scrapped {card.name} from discard pile")
                        break
        
        return False  # Turn continues

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
        
    def get_winner(self):
        if self.is_game_over:
            # Determine winner based on remaining health
            winner = max(self.players, key=lambda p: p.health)
            return winner.name
        return None
