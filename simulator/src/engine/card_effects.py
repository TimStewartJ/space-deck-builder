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

        for effect in card.effects:
            log(f"\tProcessing effect: {effect}")
                
            # Handle scrap abilities - only apply if card is being scrapped
            if effect.is_scrap_effect:
                if scrap:
                    log(f"Applying scrap effect: {effect}")
                    effect.apply(current_player, card)
                    current_player.played_cards.remove(card)  # Remove card from played cards
                continue
                
            # Handle faction ally abilities
            if effect.is_ally_effect:
                faction = effect.faction_requirement
                
                # Check if the player has sufficient faction allies
                if current_player.get_faction_ally_count(faction) > effect.faction_requirement_count:
                    log(f"Applying faction ally effect for {faction}: {effect}")
                    effect.apply(current_player, card)
                continue
                
            # Handle standard effects
            effect.apply(current_player, card)
