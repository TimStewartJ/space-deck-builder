from src.engine.actions import Action, ActionType

class Player:
    def __init__(self, name, agent=None):
        self.name = name
        self.hand = []
        self.deck = []
        self.discard_pile = []
        self.bases = []
        self.played_cards = []
        self.health = 50  # Starting authority
        self.agent = agent  # Could be human or AI
        
        # Resources that reset each turn
        self.trade = 0
        self.combat = 0
        self.authority_gained = 0
        
    def draw_card(self):
        if not self.deck:
            # Shuffle discard pile into deck if deck is empty
            if self.discard_pile:
                self.deck = self.discard_pile.copy()
                self.discard_pile = []
                self.shuffle_deck()
            else:
                return None  # No cards to draw
                
        if self.deck:
            card = self.deck.pop()
            self.hand.append(card)
            return card
        return None
    
    def play_card(self, card):
        if card in self.hand:
            self.hand.remove(card)
            self.played_cards.append(card)
            return True
        return False
    
    def make_decision(self, game_state):
        """Use the player's agent to decide the next action"""
        if self.agent:
            return self.agent.make_decision(game_state)
        return None  # Should be overridden if no agent
    
    def reset_resources(self):
        """Reset resources at the start of turn"""
        self.trade = 0
        self.combat = 0
        self.authority_gained = 0
    
    def end_turn(self):
        # Move played cards and hand to discard pile
        self.discard_pile.extend(self.played_cards)
        self.discard_pile.extend(self.hand)
        self.played_cards = []
        self.hand = []
        
        # Draw a new hand
        for _ in range(5):
            self.draw_card()
        
        self.reset_resources()

    def shuffle_deck(self):
        import random
        random.shuffle(self.deck)