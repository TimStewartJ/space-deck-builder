from enum import Enum
from typing import TYPE_CHECKING, List, Optional
from dataclasses import dataclass

if TYPE_CHECKING:
    from src.engine.player import Player
    from src.engine.game import Game

from src.cards.factions import Faction, parse_faction


def _faction_matches(card_faction, target) -> bool:
    """Check if a card's faction bitmask contains the target faction.

    Args:
        card_faction: A Faction bitmask (from card.faction).
        target: A Faction bitmask or string faction name to check.
    """
    if not card_faction:
        return False
    if isinstance(target, str):
        target = parse_faction(target)
    if isinstance(card_faction, str):
        card_faction = parse_faction(card_faction)
    return bool(card_faction & target)

class CardEffectType(Enum):
    COMBAT = "combat"
    TRADE = "trade"
    DRAW = "draw"
    HEAL = "heal"
    SCRAP = "scrap"  # For effects that allow scrapping other cards
    PARENT = "parent"  # For effects that are parent effects
    TARGET_DISCARD = "target_discard"  # For effects that make the target player discard cards
    DESTROY_BASE = "destroy_base"  # For effects that destroy bases
    COMPLEX = "complex"  # Unimplemented effects (kept for documentation only)
    # Precompiled complex effects — no runtime text parsing needed
    DRAW_PER_FACTION = "draw_per_faction"  # Draw a card for each card of a faction in play
    CONDITIONAL_DRAW = "conditional_draw"  # Draw N if condition met (e.g. "two or more bases")
    SCRAP_FROM_HAND_DISCARD = "scrap_from_hand_discard"  # Scrap up to N from hand/discard with draw-on-complete
    DISCARD_DRAW = "discard_draw"  # Discard up to N, then draw that many

class CardTargetType(Enum):
    HAND = "hand"
    DISCARD = "discard"
    TRADE = "trade"

@dataclass
class Effect:
    effect_type: CardEffectType
    value: int = 0
    text: str = ""
    faction_requirement: Optional[str] = None
    is_scrap_effect: bool = False
    is_ally_effect: bool = False
    is_or_effect: bool = False
    is_mandatory: bool = False
    faction_requirement_count: int = 0
    child_effects: Optional[List['Effect']] = None
    card_targets: Optional[List[str]] = None
    
    def __init__(self, effect_type: CardEffectType, value: int = 0, text: str = "", 
                 faction_requirement: Optional[str] = None, is_scrap_effect: bool = False,
                 is_ally_effect: bool = False, faction_requirement_count: int = 0, is_or_effect: bool = False, 
                 child_effects: Optional[List['Effect']] = None, card_targets: Optional[List[str]] = None,
                 is_mandatory: bool = False, faction_target: Optional[str] = None):
        self.effect_type = effect_type
        self.value = value
        self.text = text
        self.faction_requirement = faction_requirement
        self.is_scrap_effect = is_scrap_effect
        self.is_ally_effect = is_ally_effect
        self.is_or_effect = is_or_effect
        self.faction_requirement_count = faction_requirement_count if faction_requirement_count > 0 else (1 if faction_requirement else 0)
        self.applied = False
        self.child_effects = child_effects
        self.card_targets = card_targets
        self.is_mandatory = is_mandatory
        # Target faction for DRAW_PER_FACTION (distinct from faction_requirement which gates ally abilities)
        self.faction_target = faction_target

    def any_child_effects_used(self):
        """Check if any child effects have been used"""
        if self.child_effects:
            for effect in self.child_effects:
                if effect.applied:
                    return True
        return False
    
    def all_child_effects_used(self):
        """Check if all child effects have been used"""
        if self.child_effects:
            for effect in self.child_effects:
                if not effect.all_child_effects_used():
                    return False
            return True
        return self.applied
    
    def apply(self, game: 'Game', player: 'Player', card=None):
        from src.engine.actions import Action, ActionType, CardSource
        # If it has already been applied, do nothing
        if self.applied:
            return
        
        # Check if the player meets the faction requirement
        if self.faction_requirement and player.get_faction_ally_count(self.faction_requirement) <= self.faction_requirement_count:
            return
            
        if self.effect_type == CardEffectType.COMBAT:
            player.combat += self.value
        elif self.effect_type == CardEffectType.TRADE:
            player.trade += self.value
            game.stats.record_trade(player.name, self.value)
        elif self.effect_type == CardEffectType.DRAW:
            for _ in range(self.value):
                player.draw_card()
                game.stats.record_card_draw(player.name)
        elif self.effect_type == CardEffectType.HEAL:
            player.health += self.value
            game.stats.record_authority_gain(player.name, self.value)
        elif self.effect_type == CardEffectType.SCRAP:
            pending_actions = []
            if self.card_targets and "discard" in self.card_targets:
                for target in player.discard_pile:
                    pending_actions.append(Action(
                        ActionType.SCRAP_CARD,
                        card_id=target.index,
                        card=target,
                        card_source=CardSource.DISCARD
                    ))
            if self.card_targets and "hand" in self.card_targets:
                for target in player.hand:
                    pending_actions.append(Action(
                        ActionType.SCRAP_CARD,
                        card_id=target.index,
                        card=target,
                        card_source=CardSource.HAND
                    ))
            if self.card_targets and "trade" in self.card_targets:
                for target in game.trade_row:
                    pending_actions.append(Action(
                        ActionType.SCRAP_CARD,
                        card_id=target.index,
                        card=target,
                        card_source=CardSource.TRADE
                    ))
            if pending_actions:
                player.add_pending_actions(pending_actions, self.value, self.is_mandatory)
        elif self.effect_type == CardEffectType.TARGET_DISCARD:
            # Current player chooses which card the opponent discards
            if self.card_targets and "opponent" in self.card_targets:
                opponent = game.get_opponent(player)
                if opponent is None:
                    return
                pending_actions = []
                for target in opponent.hand:
                    action = Action(
                        ActionType.DISCARD_CARDS,
                        card_id=target.index,
                        card=target,
                        card_source=CardSource.OPPONENT
                    )
                    pending_actions.append(action)
                if pending_actions:
                    player.add_pending_actions(pending_actions, self.value, True)
        elif self.effect_type == CardEffectType.DESTROY_BASE:
            # Current player chooses which opponent base to destroy.
            # Unlike combat attacks, destroy effects can target any base
            # (outposts do NOT have priority here).
            opponent = game.get_opponent(player)
            if opponent is None:
                return
            pending_actions = []
            for base in opponent.bases:
                pending_actions.append(Action(
                    ActionType.DESTROY_BASE,
                    target_id=base.index,
                    card_id=base.index,
                    card=base,
                ))
            if pending_actions:
                player.add_pending_actions(pending_actions, self.value, False)
                
        elif self.effect_type == CardEffectType.PARENT:
            # Apply child effects if this card has any
            if self.child_effects:
                for effect in self.child_effects:
                    effect.apply(game, player, card)

        elif self.effect_type == CardEffectType.DRAW_PER_FACTION:
            # Draw a card for each card of the specified faction in play
            faction = self.faction_target.lower() if self.faction_target else ""
            count = sum(1 for c in player.played_cards
                        if _faction_matches(c.faction, faction))
            for _ in range(count):
                player.draw_card()

        elif self.effect_type == CardEffectType.CONDITIONAL_DRAW:
            # Draw N cards if condition is met (e.g. "bases_ge_2")
            condition_met = False
            if self.text == "bases_ge_2":
                condition_met = len(player.bases) >= 2
            if condition_met:
                for _ in range(self.value):
                    player.draw_card()
                    game.stats.record_card_draw(player.name)

        elif self.effect_type == CardEffectType.SCRAP_FROM_HAND_DISCARD:
            # Scrap up to N cards from hand/discard with draw-on-complete
            pending_actions = []
            for target in player.hand:
                pending_actions.append(Action(
                    ActionType.SCRAP_CARD, card_id=target.index,
                    card=target, card_source=CardSource.HAND
                ))
            for target in player.discard_pile:
                pending_actions.append(Action(
                    ActionType.SCRAP_CARD, card_id=target.index,
                    card=target, card_source=CardSource.DISCARD
                ))
            if pending_actions:
                player.add_pending_actions(
                    pending_actions, self.value, mandatory=False,
                    on_complete_draw=True
                )

        elif self.effect_type == CardEffectType.DISCARD_DRAW:
            # Discard up to N cards, then draw that many
            pending_actions = []
            for target in player.hand:
                pending_actions.append(Action(
                    ActionType.DISCARD_CARDS, card_id=target.index,
                    card=target, card_source=CardSource.SELF
                ))
            if pending_actions:
                player.add_pending_actions(
                    pending_actions, self.value, mandatory=False,
                    on_complete_draw=True
                )

        elif self.effect_type == CardEffectType.COMPLEX:
            # Unimplemented complex effects — silently skip
            pass
        
        # parent effects should only be marked as applied if it is an OR effect
        if self.effect_type != CardEffectType.PARENT:
            self.applied = True

        # if the effect is a scrap effect, remove the card from the game
        if self.is_scrap_effect and card:
            game.stats.record_card_scrap(player.name, "card")
            # Remove from played cards
            if card in player.played_cards:
                player.played_cards.remove(card)
            # Also remove from bases if it was a base/outpost
            if card in player.bases:
                player.bases.remove(card)
            # Invalidate faction cache since a card left play
            player.invalidate_faction_cache()
            # If this is an explorer, add it back to the explorer pile
            if card.name == "Explorer":
                game.explorer_pile.append(card)

    def clone(self) -> 'Effect':
        """Create a lightweight copy with fresh mutable state.
        
        Shares immutable config fields by reference but creates new Effect
        objects with applied=False and independently cloned child_effects.
        """
        return Effect(
            effect_type=self.effect_type,
            value=self.value,
            text=self.text,
            faction_requirement=self.faction_requirement,
            is_scrap_effect=self.is_scrap_effect,
            is_ally_effect=self.is_ally_effect,
            faction_requirement_count=self.faction_requirement_count,
            is_or_effect=self.is_or_effect,
            child_effects=[c.clone() for c in self.child_effects] if self.child_effects else None,
            card_targets=list(self.card_targets) if self.card_targets else None,
            is_mandatory=self.is_mandatory,
            faction_target=self.faction_target,
        )

    def reset(self):
        """Reset the effect's applied status at the end of turn, including child effects"""
        self.applied = False
        if self.child_effects:
            for child in self.child_effects:
                child.reset()
        
    def __str__(self):
        base = f"{self.effect_type.name.capitalize()}: "
        if self.is_scrap_effect:
            base = f"(Scrap required): {base}"
        if self.child_effects:
            base += "[ Child Effects: " + (" OR " if self.is_or_effect else ", ").join(str(effect) for effect in self.child_effects) + " ] "
            return base
        base += f"{self.value}" if self.value else self.text
        base += f" from {self.card_targets}" if self.card_targets else ""
        if self.is_ally_effect and self.faction_requirement:
            base = f"{self.faction_requirement} Ally: {base}"
        if self.is_mandatory:
            base = f"{base} (Mandatory)"
        return base