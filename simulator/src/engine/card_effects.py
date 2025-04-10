from src.utils.logger import log
from src.engine.player import Player
from src.cards.card import Card

class CardEffects:
    def __init__(self):
        # this is blank
        pass

    def apply_card_effects(self, current_player: Player, card: Card, scrap=False):
        """
        Apply the effects of a played card
        
        Args:
            card: The card being played
            scrap: Boolean indicating if this is a scrap effect being activated
        """
        log(f"Applying effects for {card.name}")
        
        import re

        for effect in card.effects:
            effect = effect.strip()
            if not effect:
                continue
            
            log(f"Processing effect: {effect}")
                
            # Handle scrap abilities - only apply if card is being scrapped
            if effect.startswith("{Scrap}:"):
                if scrap:
                    effect = effect.replace("{Scrap}:", "").strip()
                    log(f"Applying scrap effect: {effect}")
                    self._parse_and_apply_effect(current_player, effect, card)
                continue
                
            # Handle faction ally abilities
            ally_match = re.search(r"\{(\w+) Ally\}:\s*(.*)", effect)
            if ally_match:
                faction = ally_match.group(1)
                ally_effect = ally_match.group(2)
                
                # Check if the player has played another card of this faction
                if current_player._has_faction_ally(faction, card):
                    log(f"Applying faction ally effect for {faction}: {ally_effect}")
                    self._parse_and_apply_effect(current_player, ally_effect, card)
                continue
            
            # Handle OR choices
            if "OR" in effect:
                # For now just apply the first choice, later implement player choice
                choices = effect.split("OR")
                log(f"Applying first choice of: {effect}")
                self._parse_and_apply_effect(current_player, choices[0].strip(), card)
                continue
                
            # Handle standard effects
            self._parse_and_apply_effect(current_player, effect, card)
        
    def _parse_and_apply_effect(self, current_player: Player, effect_text, card):
        """Parse and apply a specific card effect"""
        import re
        
        # Handle resource gains enclosed in curly braces
        trade_match = re.search(r"\{Gain (\d+) Trade\}", effect_text)
        if trade_match:
            current_player.trade += int(trade_match.group(1))
            
        combat_match = re.search(r"\{Gain (\d+) Combat\}", effect_text)
        if combat_match:
            current_player.combat += int(combat_match.group(1))
            
        # Handle card draw
        if "Draw a card" in effect_text:
            current_player.draw_card()
            
        # Handle conditional card draw
        draw_match = re.search(r"Draw a card for each (\w+) card", effect_text)
        if draw_match:
            faction = draw_match.group(1).lower()
            count = sum(1 for c in current_player.played_cards if c.faction.lower() == faction)
            for _ in range(count):
                current_player.draw_card()
                
        # Handle complex effects that require player choice
        if any(x in effect_text for x in [
                "Acquire any ship for free", 
                "destroy target base", 
                "scrap a card in the trade row",
                "scrap a card in your hand or discard pile"
            ]):
            self._create_player_choice_action(current_player, effect_text, card)

    def _create_player_choice_action(self, current_player: Player, effect_text, card):
        """Create appropriate actions for effects requiring player decisions"""
        from src.engine.actions import Action, ActionType
        
        if "scrap a card in your hand or discard pile":
            discard_targets = [c.name for c in current_player.discard_pile]
            hand_targets = [c.name for c in current_player.hand]

            for target in discard_targets:
                action = Action(
                    ActionType.SCRAP_CARD,
                    card_id=target,
                    source=["discard"]
                )
                current_player.pending_actions.append(action)

            for target in hand_targets:
                action = Action(
                    ActionType.SCRAP_CARD,
                    card_id=target,
                    source=["hand"]
                )
                current_player.pending_actions.append(action)

