import copy
from typing import List
from src.cards.effects import CardEffectType
from src.utils.logger import log
from src.cards.card import Card
from src.engine.player import Player
from src.engine.actions import ActionType, Action, get_available_actions
from src.engine.game_stats import GameStats

class Game:
    def __init__(self, cards=None):
        self.players: List[Player] = []
        self.current_turn = 0
        self.is_game_over = False
        self.trade_deck = copy.deepcopy(cards) if cards else []
        self.trade_row: List[Card] = []
        self.is_running = False
        self.current_player: Player = None
        self.stats = GameStats()

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
        self.fill_trade_row()
    
    def fill_trade_row(self):
        # Fill the trade row with 5 cards
        while len(self.trade_row) < 5 and self.trade_deck:
            card = self.trade_deck.pop()
            self.trade_row.append(card)
            log(f"Added {card.name} to trade row", v=True)

    def shuffle_trade_deck(self):
        import random
        random.shuffle(self.trade_deck)

    def setup_players(self):
        # Shuffle players to randomize turn order
        import random
        random.shuffle(self.players)

        for player in self.players:
            # Initialize statistics for this player
            self.stats.add_player(player.name)
            # Initialize each player's starting deck with 8 Scouts and 2 Vipers
            starting_deck = self.create_starting_deck()
            player.deck.extend(starting_deck)
            player.shuffle_deck()
            # Draw initial hand
            for _ in range(5):
                player.draw_card()
                self.stats.record_card_draw(player.name)

    def create_starting_deck(self):
        from src.cards.effects import Effect

        # Create a deck of 8 Scouts and 2 Vipers
        from src.cards.card import Card
        starting_deck = []
        # Add 8 Scouts
        for _ in range(8):
            starting_deck.append(Card("Scout", 0, [Effect(CardEffectType.TRADE, 1)], "ship"))
        # Add 2 Vipers
        for _ in range(2):
            starting_deck.append(Card("Viper", 0, [Effect(CardEffectType.COMBAT, 1)], "ship"))
        return starting_deck

    def next_step(self):
        if self.is_game_over:
            return
        
        turn_ended = False

        # Get player decision (through UI or AI)
        action = self.current_player.make_decision(self)

        log(f"Getting action for {self.current_player.name}", v=True)

        if action:
            turn_ended = self.execute_action(action)
        
        if turn_ended:
            log(f"Ended turn for {self.current_player.name}", v=True)

            # Tell player turn is over
            self.current_player.reset_resources()
            self.current_player.end_turn()

            # Increment turn counter and move onto next player
            self.stats.total_turns += 1
            self.current_turn = (self.current_turn + 1) % len(self.players)
            self.current_player = self.players[self.current_turn]

        return action
        
    
    def execute_action(self, action: Action):
        """Execute a player's action and update game state"""
        log(f"{self.current_player.name} executing action: {action}", v=True)

        # Check for pending actions. Decrement the pending action count
        if self.current_player.pending_actions:
            # Reset actions if the action is a skip decision
            if action.type == ActionType.SKIP_DECISION:
                self.current_player.reset_pending_actions()
            # Otherwise remove the action from pending actions
            # and decrement the count
            else:
                self.current_player.pending_actions.remove(action)
                self.current_player.pending_actions_left -= 1
                if self.current_player.pending_actions_left <= 0:
                    self.current_player.reset_pending_actions()
        
        if action.type == ActionType.END_TURN:
            return True  # End the turn
            
        elif action.type == ActionType.PLAY_CARD:
            # Find the card in player's hand
            for card in self.current_player.hand:
                if card.name == action.card_id:
                    self.current_player.play_card(card)
                    self.stats.record_card_play(self.current_player.name)
                    log(f"{self.current_player.name} played {card.name}", v=True)
                    # Apply first card effect if ship
                    if card.card_type == "ship" and card.effects and len(card.effects) > 0:
                        card.effects[0].apply(self, self.current_player, card)
                    break
        
        elif action.type == ActionType.APPLY_EFFECT and action.additional_params is not None:
            # Apply the effect directly
            effect = action.additional_params.get('effect')
            if effect:
                effect.apply(self, self.current_player)
                log(f"{self.current_player.name} applied effect: {effect}", v=True)
                    
        elif action.type == ActionType.BUY_CARD:
            # Find the card in trade row
            for i, card in enumerate(self.trade_row):
                if card.name == action.card_id and self.current_player.trade >= card.cost:
                    self.current_player.trade -= card.cost
                    self.current_player.discard_pile.append(card)
                    self.stats.record_card_buy(self.current_player.name)
                    log(f"{self.current_player.name} bought {card.name} for {card.cost} trade", v=True)
                    self.trade_row.pop(i)
                    # Replace card in trade row
                    if self.trade_deck:
                        new_card = self.trade_deck.pop()
                        self.trade_row.append(new_card)
                        self.stats.trade_row_refreshes += 1
                        log(f"Added {new_card.name} to trade row", v=True)
                    break
        
        elif action.type == ActionType.ATTACK_BASE:
            # Find target base
            for player in self.players:
                if player != self.current_player:
                    for base in player.bases:
                        if base.name == action.target_id and base.defense and self.current_player.combat >= base.defense:
                            self.current_player.combat -= base.defense
                            player.bases.remove(base)
                            player.discard_pile.append(base)
                            self.stats.record_base_destroy(self.current_player.name)
                            log(f"{self.current_player.name} destroyed {player.name}'s {base.name}", v=True)
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
                        self.stats.record_damage(self.current_player.name, damage)
                        log(f"{self.current_player.name} attacked {player.name} for {damage} damage", v=True)
                        # Check for game over
                        if player.health <= 0:
                            log(f"{player.name} has been defeated!")
                            self.is_game_over = True
                            self.stats.end_game(self.current_player.name)
                        break
        
        elif action.type == ActionType.SCRAP_CARD and action.card_source is not None:
            self.stats.record_card_scrap(self.current_player.name, action.card_source)
            if action.card_source == 'hand':
                # Scrap card from hand
                for card in self.current_player.hand:
                    if card.name == action.card_id:
                        self.current_player.hand.remove(card)
                        log(f"{self.current_player.name} scrapped {card.name} from hand", v=True)
                        break
            elif action.card_source == 'discard':
                # Scrap card from discard pile
                for card in self.current_player.discard_pile:
                    if card.name == action.card_id:
                        self.current_player.discard_pile.remove(card)
                        log(f"{self.current_player.name} scrapped {card.name} from discard pile", v=True)
                        break
            elif action.card_source == 'trade':
                # Scrap card from trade row
                for i, card in enumerate(self.trade_row):
                    if card.name == action.card_id:
                        self.trade_row.pop(i)
                        log(f"{self.current_player.name} scrapped {card.name} from trade row", v=True)
                        break
                # If this was the last pending action, refresh the trade row
                if self.current_player.pending_actions_left <= 0:
                    self.fill_trade_row()
        elif action.type == ActionType.DISCARD_CARDS:
            # Discard card from hand
            self.stats.record_cards_discarded_from_hand(self.current_player.name, 1)
            for card in self.current_player.hand:
                if card.name == action.card_id:
                    self.current_player.hand.remove(card)
                    self.current_player.discard_pile.append(card)
                    log(f"{self.current_player.name} discarded {card.name}", v=True)
                    break

        return False  # Turn continues

    def end_game(self):
        self.is_game_over = True
        self.is_running = False
        # Logic to determine the winner and end the game
        pass

    def add_player(self, name: str):
        if len(self.players) < 4:  # Maximum 4 players
            from src.engine.player import Player
            player = Player(name)
            self.players.append(player)
            return player
        else:
            raise ValueError("Maximum number of players reached")
        
    def get_opponent(self, player: Player):
        """Get the opponent of the given player"""
        for p in self.players:
            if p != player:
                return p
        return None
        
    def get_winner(self):
        if self.is_game_over:
            # Determine winner based on remaining health
            winner = max(self.players, key=lambda p: p.health)
            return winner.name
        return None
