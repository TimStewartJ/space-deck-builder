from typing import List
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
    
    def execute_action(self, action):
        """Execute a player's action and update game state"""
        self.log(f"{self.current_player.name} executing action: {action}")
        
        if action.type == ActionType.END_TURN:
            return True  # End the turn
            
        elif action.type == ActionType.PLAY_CARD:
            # Find the card in player's hand
            for card in self.current_player.hand:
                if card.name == action.card_id:
                    self.current_player.play_card(card)
                    self.log(f"{self.current_player.name} played {card.name}")
                    self.apply_card_effects(card)
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

        
        return False  # Turn continues
    
    def apply_card_effects(self, card: Card, scrap=False):
        """
        Apply the effects of a played card
        
        Args:
            card: The card being played
            scrap: Boolean indicating if this is a scrap effect being activated
        """
        self.log(f"Applying effects for {card.name}")
        
        import re

        for effect in card.effects:
            effect = effect.strip()
            if not effect:
                continue
            
            self.log(f"Processing effect: {effect}")
                
            # Handle scrap abilities - only apply if card is being scrapped
            if effect.startswith("{Scrap}:"):
                if scrap:
                    effect = effect.replace("{Scrap}:", "").strip()
                    self.log(f"Applying scrap effect: {effect}")
                    self._parse_and_apply_effect(effect, card)
                continue
                
            # Handle faction ally abilities
            ally_match = re.search(r"\{(\w+) Ally\}:\s*(.*)", effect)
            if ally_match:
                faction = ally_match.group(1)
                ally_effect = ally_match.group(2)
                
                # Check if the player has played another card of this faction
                if self._has_faction_ally(faction, card):
                    self.log(f"Applying faction ally effect for {faction}: {ally_effect}")
                    self._parse_and_apply_effect(ally_effect, card)
                continue
            
            # Handle OR choices
            if "OR" in effect:
                # For now just apply the first choice, later implement player choice
                choices = effect.split("OR")
                self.log(f"Applying first choice of: {effect}")
                self._parse_and_apply_effect(choices[0].strip(), card)
                continue
                
            # Handle standard effects
            self._parse_and_apply_effect(effect, card)
    
    def _has_faction_ally(self, faction, current_card):
        """Check if player has played another card of the specified faction this turn"""
        for card in self.current_player.played_cards:
            if card.faction.lower() == faction.lower() and card != current_card:
                return True
        return False
        
    def _parse_and_apply_effect(self, effect_text, card):
        """Parse and apply a specific card effect"""
        import re
        
        # Handle resource gains enclosed in curly braces
        trade_match = re.search(r"\{Gain (\d+) Trade\}", effect_text)
        if trade_match:
            self.current_player.trade += int(trade_match.group(1))
            
        combat_match = re.search(r"\{Gain (\d+) Combat\}", effect_text)
        if combat_match:
            self.current_player.combat += int(combat_match.group(1))
            
        # Handle card draw
        if "Draw a card" in effect_text:
            self.current_player.draw_card()
            
        # Handle conditional card draw
        draw_match = re.search(r"Draw a card for each (\w+) card", effect_text)
        if draw_match:
            faction = draw_match.group(1).lower()
            count = sum(1 for c in self.current_player.played_cards if c.faction.lower() == faction)
            for _ in range(count):
                self.current_player.draw_card()
                
        # Handle complex effects that require player choice
        if any(x in effect_text for x in [
                "Acquire any ship for free", 
                "destroy target base", 
                "scrap a card in the trade row",
                "scrap a card in your hand or discard pile"
            ]):
            self._create_player_choice_action(effect_text, card)

    def _create_player_choice_action(self, effect_text, card):
        """Create appropriate actions for effects requiring player decisions"""
        from src.engine.actions import Action, ActionType
        
        # Example implementation - will need to be integrated with your action system
        if "Acquire any ship for free" in effect_text:
            ships = [c for c in self.trade_row if getattr(c, 'type', '') == 'ship']
            if ships:
                # Create action for player to choose a ship
                action = Action(
                    ActionType.CHOOSE_CARD,
                    card_id=card.name,
                    valid_targets=[c.name for c in ships],
                    effect_text=effect_text
                )
                # Store this pending action or return it
                self.current_player.pending_actions.append(action)
                
        elif "destroy target base" in effect_text:
            # Find opponent bases
            bases = []
            for player in self.players:
                if player != self.current_player:
                    bases.extend(player.bases)
                    
            if bases:
                action = Action(
                    ActionType.DESTROY_BASE,
                    card_id=card.name,
                    valid_targets=[b.name for b in bases],
                    effect_text=effect_text
                )
                self.current_player.pending_actions.append(action)    

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