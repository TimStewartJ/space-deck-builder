from typing import List, Optional
from src.cards.card import Card
from src.ai.agent import Agent
from src.engine.actions import ActionType, Action, get_available_actions, PendingActionSet

class Player:
    def __init__(self, name, agent, starting_health: int = 50, hand_size: int = 5):
        self.name: str = name
        self.hand: List[Card] = []
        self.deck: List[Card] = []
        self.discard_pile: List[Card] = []
        self.bases: List[Card] = []
        self.played_cards: List[Card] = []
        self.health = starting_health
        self.hand_size = hand_size
        self.agent: Agent = agent  # Could be human or AI
        
        # Resources that reset each turn
        self.trade = 0
        self.combat = 0
        self.authority_gained = 0
        # Support multiple sets of pending actions
        self.pending_action_sets: List[PendingActionSet] = []
        self.cards_drawn = 0  # Track the number of cards drawn
        # Cached faction ally counts — updated incrementally on card play,
        # reset on end_turn. Maps lowercase faction name → count.
        self._faction_counts: dict[str, int] = {}
        self._faction_counts_dirty: bool = True
        self._faction_counts_card_count: int = 0  # track card count for staleness detection
        
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
            if card.is_base():
                self.bases.append(card)
            # Update cached faction counts for the newly played card
            self._update_faction_counts_for_card(card)
            return True
        return False

    def _update_faction_counts_for_card(self, card):
        """Incrementally update faction counts when a card enters play."""
        if self._faction_counts_dirty:
            self._rebuild_faction_counts()
            return
        factions = self._get_card_faction_keys(card)
        for f in factions:
            self._faction_counts[f] = self._faction_counts.get(f, 0) + 1

    def _get_card_faction_keys(self, card) -> list[str]:
        """Get the lowercase faction keys a card contributes to ally counts."""
        if card.ally_factions is not None:
            if "*" in card.ally_factions:
                return ["*"]
            return [f.lower() for f in card.ally_factions]
        if not card.faction:
            return []
        if isinstance(card.faction, list):
            return [f.lower() for f in card.faction]
        return [card.faction.lower()]

    def _rebuild_faction_counts(self):
        """Full rebuild of faction counts from played_cards + bases."""
        self._faction_counts.clear()
        for card in set(self.played_cards) | set(self.bases):
            for f in self._get_card_faction_keys(card):
                self._faction_counts[f] = self._faction_counts.get(f, 0) + 1
        self._faction_counts_dirty = False
        self._faction_counts_card_count = len(self.played_cards) + len(self.bases)

    def invalidate_faction_cache(self):
        """Mark faction count cache as stale.

        Call this when cards are removed from play (scrap, destroy)
        outside the normal play_card/end_turn flow.
        """
        self._faction_counts_dirty = True
    
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
            card.reset_effects()
            if card not in self.bases:
                self.discard_pile.append(card)
                
        self.played_cards = []
        self.reset_pending_actions()

        # Reset base effects so they're available again next turn
        for base in self.bases:
            base.reset_effects()
        
        # Move hand to discard pile
        self.discard_pile.extend(self.hand)
        self.hand = []
        
        # Draw a new hand
        for _ in range(self.hand_size):
            self.draw_card()
        
        self.reset_resources()
        # Reset faction count cache for next turn
        self._faction_counts.clear()
        self._faction_counts_dirty = True

    def shuffle_deck(self):
        import random
        random.shuffle(self.deck)
    
    def add_pending_actions(self, actions: List[Action], decisions_left: int, mandatory: bool,
                           on_complete_draw: bool = False):
        """Add a new set of pending actions.

        Args:
            on_complete_draw: If True, draw one card per resolved action when
                              the set completes (used by Brain World, Recycling Station).
        """
        self.pending_action_sets.append(PendingActionSet(
            actions=list(actions),
            decisions_left=decisions_left,
            mandatory=mandatory,
            on_complete_draw=on_complete_draw,
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
        """Count cards that count as allies for the specified faction.

        Uses cached faction counts when available (updated incrementally
        on card play). Falls back to full rebuild if cache is dirty or
        the card count has changed (e.g. direct list manipulation).
        """
        current_count = len(self.played_cards) + len(self.bases)
        if self._faction_counts_dirty or self._faction_counts_card_count != current_count:
            self._rebuild_faction_counts()
        target = faction.lower()
        count = self._faction_counts.get(target, 0)
        # Wildcard cards ("*") count as allies for every faction
        if target != "*":
            count += self._faction_counts.get("*", 0)
        return count
