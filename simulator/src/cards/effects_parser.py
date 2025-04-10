import re
from .effects import CardEffectType, Effect

def parse_effect_text(text: str) -> Effect:
    """Parse a single effect text and return an Effect object"""
    # Check for scrap effects
    is_scrap = False
    if text.startswith("{Scrap}:"):
        is_scrap = True
        text = text.replace("{Scrap}:", "").strip()
    
    # Check for ally effects
    is_ally = False
    faction_requirement = None
    faction_requirement_count = 0
    ally_match = re.search(r"\{(?:(Double)\s+)?([^}]+?)\s+Ally\}:\s*(.*)", text)
    if ally_match:
        is_ally = True
        is_double = ally_match.group(1) == "Double"
        faction_requirement = ally_match.group(2)
        faction_requirement_count = 2 if is_double else 1
        text = ally_match.group(3).strip()
    
    # Parse common resource gains
    trade_match = re.search(r"\{Gain (\d+) Trade\}", text)
    if trade_match:
        return Effect(CardEffectType.TRADE, int(trade_match.group(1)), text, 
                     faction_requirement, is_scrap, is_ally, faction_requirement_count)
        
    combat_match = re.search(r"\{Gain (\d+) Combat\}", text)
    if combat_match:
        return Effect(CardEffectType.COMBAT, int(combat_match.group(1)), text,
                     faction_requirement, is_scrap, is_ally, faction_requirement_count)
    
    healing_match = re.search(r"\{Gain (\d+) Authority\}", text)
    if healing_match:
        return Effect(CardEffectType.HEAL, int(healing_match.group(1)), text,
                     faction_requirement, is_scrap, is_ally, faction_requirement_count)
    
    # Parse draw effects
    if text == "Draw a card.":
        return Effect(CardEffectType.DRAW, 1, text, faction_requirement, is_scrap, is_ally, faction_requirement_count)
    
    draw_match = re.search(r"Draw (\d+) cards?", text)
    if draw_match:
        return Effect(CardEffectType.DRAW, int(draw_match.group(1)), text,
                     faction_requirement, is_scrap, is_ally, faction_requirement_count)
    
    # Default case - store as text for complex effects
    return Effect(CardEffectType.COMPLEX, 0, text, faction_requirement, is_scrap, is_ally, faction_requirement_count)
