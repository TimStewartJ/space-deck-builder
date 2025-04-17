from typing import List, Optional
from src.cards.card import Card
from src.ai.agent import Agent
from src.engine.actions import ActionType, Action, get_available_actions, PendingActionSet

class Player:
    def __init__(self, name, agent):
        self.name: str = name
        self.hand: List[Card] = []
        self.deck: List[Card] = []
        self.discard_pile: List[Card] = []
        self.bases: List[Card] = []
        self.played_cards: List[Card] = []
        self.health = 50  # Starting authority
        self.agent: Agent = agent  # Could be human or AI
        
        # Resources that reset each turn
        self.trade = 0
        self.combat = 0
        self.authority_gained = 0
        # Support multiple sets of pending actions
        self.pending_action_sets: List[PendingActionSet] = []
        self.cards_drawn = 0  # Track the number of cards drawn
        
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
            self.cards_drawn += 1  # Increment the card draw count
            return card
        return None
    
    def play_card(self, card):
        """Play a card from hand"""
        if card in self.hand:
            self.hand.remove(card)
            self.played_cards.append(card)
            if card.card_type == "base":
                self.bases.append(card)
            return True
        return False
    
    def make_decision(self, game_state):
        """Use the player's agent to decide the next action"""
        if self.agent:
            return self.agent.make_decision(game_state)
        return None  # Should be overridden if no agent
    
    def reset_pending_actions(self):
        """Reset pending actions"""
        # Clear all pending action sets
        self.pending_action_sets = []
    
    def reset_resources(self):
        """Reset resources at the start of turn"""
        self.trade = 0
        self.combat = 0
        self.authority_gained = 0
    
    def end_turn(self):
        """End the current turn"""
        # Move played cards to discard pile (except bases)
        for card in self.played_cards:
            card.reset_effects()  # Reset card effects
            if card not in self.bases:
                self.discard_pile.append(card)
                
        self.played_cards = []
        self.reset_pending_actions()
        
        # Move hand to discard pile
        self.discard_pile.extend(self.hand)
        self.hand = []
        
        # Draw a new hand
        for _ in range(5):
            self.draw_card()
        
        self.reset_resources()

    def shuffle_deck(self):
        import random
        random.shuffle(self.deck)
    
    def add_pending_actions(self, actions: List[Action], decisions_left: int, mandatory: bool):
        """Add a new set of pending actions"""
        self.pending_action_sets.append(PendingActionSet(
            actions=list(actions),
            decisions_left=decisions_left,
            mandatory=mandatory
        ))

    def get_current_pending_set(self) -> Optional[PendingActionSet]:
        """Get the first pending action set"""
        return self.pending_action_sets[0] if self.pending_action_sets else None

    def advance_pending_set(self):
        """Remove the completed pending action set and move to next"""
        if self.pending_action_sets:
            self.pending_action_sets.pop(0)

    def skip_current_pending_set(self):
        """Skip the current pending action set"""
        self.advance_pending_set()
    
    def get_faction_ally_count(self, faction):
        """Count the number of cards of the specified faction in play"""
        return sum(1 for card in self.played_cards if card.faction and card.faction.lower() == faction.lower())
