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
    ACQUIRE_SHIP_TOP = "acquire_ship_top"
    NEXT_SHIP_TOP = "next_ship_top"
    COPY_SHIP = "copy_ship"
    PASSIVE_SHIP_COMBAT = "passive_ship_combat"
    WILDCARD_ALLY = "wildcard_ally"

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
    
    def apply(self, game: 'Game', player: 'Player', card=None, *, auto_only=False):
        from src.rules.scalar import apply_effect

        apply_effect(self, game, player, card, auto_only=auto_only)

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