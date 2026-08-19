from collections import Counter
from typing import List, Optional
from src.cards.card import Card
from src.cards.factions import Faction, parse_faction
from src.ai.agent import Agent
from src.config import GameConfig
from src.engine.actions import ActionType, Action, get_available_actions, PendingActionSet

class Player:
    def __init__(self, name, agent, starting_health: int | None = None,
                 hand_size: int | None = None,
                 game_config: GameConfig | None = None):
        cfg = game_config or GameConfig()
        self.name: str = name
        self.hand: List[Card] = []
        self.deck: List[Card] = []
        self.ordered_top: List[Card] = []
        self.discard_pile: List[Card] = []
        self.bases: List[Card] = []
        self.played_cards: List[Card] = []
        self.health = starting_health if starting_health is not None else cfg.starting_health
        self.hand_size = hand_size if hand_size is not None else cfg.hand_size
        self.agent: Agent = agent  # Could be human or AI
        
        # Resources that reset each turn
        self.trade = 0
        self.combat = 0
        self.authority_gained = 0
        # Support multiple sets of pending actions
        self.pending_action_sets: List[PendingActionSet] = []
        # Deferred forced discards — incremented when an opponent plays a
        # "target opponent discards" effect. Materialized into pending
        # actions at the start of this player's next turn so that the
        # player (not the attacker) chooses which cards to discard.
        self.pending_start_of_turn_discards: int = 0
        self.next_ship_to_top: int = 0
        # Cached faction ally counts — updated incrementally on card play,
        # reset on end_turn. Maps Faction bit → count of cards contributing.
        self._faction_counts: dict[Faction, int] = {}
        self._faction_counts_dirty: bool = True
        self._faction_counts_card_count: int = 0  # track card count for staleness detection
        self.public_owned_cards: Counter[int] = Counter()
        self.public_owned_card_instances: List[Card] = []
        self.public_known_hand: List[Card] = []
        self.public_ownership_initialized = False
        self.rng = None

    def initialize_public_ownership(self, cards: List[Card]):
        """Initialize the publicly known multiset of cards this player owns."""
        self.public_owned_cards = Counter(card.index for card in cards)
        self.public_owned_card_instances = list(cards)
        self.public_known_hand.clear()
        self.public_ownership_initialized = True

    def add_owned_card(self, card: Card):
        if card in self.public_owned_card_instances:
            raise RuntimeError(
                f"{card.name} is already owned by {self.name}"
            )
        self.public_owned_cards[card.index] += 1
        self.public_owned_card_instances.append(card)

    def remove_owned_card(self, card: Card):
        count = self.public_owned_cards[card.index]
        if count <= 0:
            if self.public_ownership_initialized:
                raise RuntimeError(
                    f"public ownership has no {card.name} for {self.name}"
                )
            return
        if card not in self.public_owned_card_instances:
            if self.public_ownership_initialized:
                raise RuntimeError(
                    f"public ownership has no {card.name} instance for "
                    f"{self.name}"
                )
            return
        if count == 1:
            del self.public_owned_cards[card.index]
        else:
            self.public_owned_cards[card.index] = count - 1
        self.public_owned_card_instances.remove(card)

    def _mark_public_hand_card(self, card: Card):
        self.public_known_hand.append(card)

    def _consume_public_hand_card(self, card: Card):
        if card in self.public_known_hand:
            self.public_known_hand.remove(card)
        
    def draw_card(self):
        if not self.deck:
            # Shuffle discard pile into deck if deck is empty
            if self.discard_pile:
                self.deck = self.discard_pile.copy()
                self.discard_pile = []
                for card in self.deck:
                    card.reset_effects()
                self.shuffle_deck()
            else:
                return None  # No cards to draw
                
        if self.deck:
            card = self.deck.pop()
            if self.ordered_top:
                expected = self.ordered_top.pop()
                if expected is not card:
                    raise RuntimeError("ordered top-deck state is inconsistent")
                self._mark_public_hand_card(card)
            self.hand.append(card)
            return card
        return None
    
    def play_card(self, card):
        """Play a card from hand"""
        if card in self.hand:
            self.hand.remove(card)
            self._consume_public_hand_card(card)
            self.played_cards.append(card)
            if card.is_base():
                self.bases.append(card)
            # Update cached faction counts for the newly played card
            self._update_faction_counts_for_card(card)
            return True
        return False

    def put_on_top(self, card):
        """Place an acquired card above every card currently in the deck."""
        self.deck.append(card)
        self.ordered_top.append(card)

    def _update_faction_counts_for_card(self, card):
        """Incrementally update faction counts when a card enters play."""
        if self._faction_counts_dirty:
            self._rebuild_faction_counts()
            return
        factions = self._get_card_faction_keys(card)
        for f in factions:
            self._faction_counts[f] = self._faction_counts.get(f, 0) + 1

    def _get_card_faction_keys(self, card) -> list[Faction]:
        """Get the Faction bits a card contributes to ally counts."""
        if card.ally_factions:
            if card.ally_factions == Faction.ALL:
                return [Faction.ALL]
            # Return each individual faction bit
            return [f for f in Faction if f in card.ally_factions and f != Faction.NONE and f != Faction.ALL]
        if not card.effective_faction:
            return []
        # Return each individual faction bit from the card's faction bitmask
        return [
            f
            for f in Faction
            if f in card.effective_faction
            and f != Faction.NONE
            and f != Faction.ALL
        ]

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
        self.next_ship_to_top = 0

        # Reset base effects so they're available again next turn
        for base in self.bases:
            base.reset_effects()
        
        # Move hand to discard pile
        for card in self.hand:
            card.reset_effects()
        self.discard_pile.extend(self.hand)
        self.hand = []
        self.public_known_hand.clear()
        
        # Draw a new hand
        for _ in range(self.hand_size):
            self.draw_card()
        
        self.reset_resources()
        # Reset faction count cache for next turn
        self._faction_counts.clear()
        self._faction_counts_dirty = True

    def shuffle_deck(self):
        import random
        self.ordered_top = []
        (self.rng or random).shuffle(self.deck)
    
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
    
    def get_faction_ally_count(self, faction) -> int:
        """Count cards that count as allies for the specified faction.

        Uses cached faction counts when available (updated incrementally
        on card play). Falls back to full rebuild if cache is dirty or
        the card count has changed (e.g. direct list manipulation).

        Args:
            faction: A Faction bitmask or a string faction name.
        """
        current_count = len(self.played_cards) + len(self.bases)
        if self._faction_counts_dirty or self._faction_counts_card_count != current_count:
            self._rebuild_faction_counts()
        # Accept string for backward compatibility during transition
        if isinstance(faction, str):
            target = parse_faction(faction)
        else:
            target = faction
        count = self._faction_counts.get(target, 0)
        # Wildcard cards (Faction.ALL) count as allies for every faction
        if target != Faction.ALL:
            count += self._faction_counts.get(Faction.ALL, 0)
        return count
