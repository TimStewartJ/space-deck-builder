from enum import Enum
import re
from typing import TYPE_CHECKING, List, Optional
from dataclasses import dataclass

if TYPE_CHECKING:
    from src.engine.player import Player
    from src.engine.game import Game


def _faction_matches(card_faction, target: str) -> bool:
    """Check if a card's faction field matches a target faction (case-insensitive).

    Handles both single-faction strings and multi-faction lists.
    Returns False for None/empty factions.
    """
    if not card_faction:
        return False
    if isinstance(card_faction, list):
        return any(f.lower() == target for f in card_faction)
    return card_faction.lower() == target

class CardEffectType(Enum):
    COMBAT = "combat"
    TRADE = "trade"
    DRAW = "draw"
    HEAL = "heal"
    SCRAP = "scrap"  # For effects that allow scrapping other cards
    PARENT = "parent"  # For effects that are parent effects
    TARGET_DISCARD = "target_discard"  # For effects that make the target player discard cards
    DESTROY_BASE = "destroy_base"  # For effects that destroy bases
    COMPLEX = "complex"  # For complex effects that require special handling

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
                 is_mandatory: bool = False):
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
        from src.engine.actions import Action, ActionType
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
                        card_id=target.name,
                        card=target,
                        card_source="discard"
                    ))
            if self.card_targets and "hand" in self.card_targets:
                for target in player.hand:
                    pending_actions.append(Action(
                        ActionType.SCRAP_CARD,
                        card_id=target.name,
                        card=target,
                        card_source="hand"
                    ))
            if self.card_targets and "trade" in self.card_targets:
                for target in game.trade_row:
                    pending_actions.append(Action(
                        ActionType.SCRAP_CARD,
                        card_id=target.name,
                        card=target,
                        card_source="trade"
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
                        card_id=target.name,
                        card=target,
                        card_source="opponent"
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
                    target_id=base.name,
                    card_id=base.name,
                    card=base,
                ))
            if pending_actions:
                player.add_pending_actions(pending_actions, self.value, False)
                
        elif self.effect_type == CardEffectType.PARENT:
            # Apply child effects if this card has any
            if self.child_effects:
                for effect in self.child_effects:
                    effect.apply(game, player, card)
        elif self.effect_type == CardEffectType.COMPLEX:
            self.handle_complex_effect(game, player, card)
        
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

    def handle_complex_effect(self, game: 'Game', player: 'Player', card):
        from src.engine.actions import Action, ActionType

        # "Draw a card for each <Faction> card that you've played this turn"
        draw_match = re.search(r"Draw a card for each (\w+) card", self.text)
        if draw_match:
            faction = draw_match.group(1).lower()
            count = sum(1 for c in player.played_cards
                        if _faction_matches(c.faction, faction))
            for _ in range(count):
                player.draw_card()
            return

        # "If you have two or more bases in play, draw two cards" (Embassy Yacht)
        if "two or more bases in play" in self.text.lower():
            if len(player.bases) >= 2:
                for _ in range(2):
                    player.draw_card()
                    game.stats.record_card_draw(player.name)
            return

        # "Scrap up to two cards from your hand and/or discard pile" (Brain World)
        # Creates pending scrap actions with draw-on-complete. The sibling child
        # effect ("Draw a card for each card scrapped this way") is handled by the
        # completion mechanism and should be skipped — see PARENT handler below.
        scrap_up_to = re.search(r"Scrap up to (\w+) cards? from your hand and/or discard pile", self.text)
        if scrap_up_to:
            count_word = scrap_up_to.group(1)
            count_map = {"one": 1, "two": 2, "three": 3}
            max_scrap = count_map.get(count_word, int(count_word) if count_word.isdigit() else 1)
            pending_actions = []
            for target in player.hand:
                pending_actions.append(Action(
                    ActionType.SCRAP_CARD, card_id=target.name,
                    card=target, card_source="hand"
                ))
            for target in player.discard_pile:
                pending_actions.append(Action(
                    ActionType.SCRAP_CARD, card_id=target.name,
                    card=target, card_source="discard"
                ))
            if pending_actions:
                player.add_pending_actions(
                    pending_actions, max_scrap, mandatory=False,
                    on_complete_draw=True
                )
            return

        # "discard up to two cards, then draw that many cards" (Recycling Station)
        discard_draw = re.search(r"discard up to (\w+) cards?, then draw that many", self.text)
        if discard_draw:
            count_word = discard_draw.group(1)
            count_map = {"one": 1, "two": 2, "three": 3}
            max_discard = count_map.get(count_word, int(count_word) if count_word.isdigit() else 1)
            pending_actions = []
            for target in player.hand:
                pending_actions.append(Action(
                    ActionType.DISCARD_CARDS, card_id=target.name,
                    card=target, card_source="self"
                ))
            if pending_actions:
                player.add_pending_actions(
                    pending_actions, max_discard, mandatory=False,
                    on_complete_draw=True
                )
            return

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