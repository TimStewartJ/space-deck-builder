import re
from .effects import CardEffectType, Effect

def parse_effect_text(text: str) -> Effect:
    """Parse a single effect text and return an Effect object"""

    # If text ends with a period, remove it
    if text.endswith("."):
        text = text[:-1]

    # Check for scrap effects
    is_scrap = False
    if text.startswith("{Scrap}:"):
        is_scrap = True
        text = text.replace("{Scrap}:", "").strip()

    # If this effect is an OR effect, we will parse it separately and add it as a child effect
    # Split text by OR
    or_split = [chunk for chunk in text.split("OR") if chunk.strip()]
    if len(or_split) > 1:
        child_effects = []
        for effect_text in or_split:
            child_effects.append(parse_effect_text(effect_text.strip()))
        return Effect(CardEffectType.PARENT, 0, text, child_effects=child_effects, is_scrap_effect=is_scrap, is_or_effect=True)

    # if this effect has multiple effects, we will parse them separately and add them as child effects
    # Split text by both new line and period
    split_text = [chunk for chunk in re.split(r"[.\n]", text) if chunk.strip()]
    if len(split_text) > 1:
        child_effects = []
        for effect_text in split_text:
            child_effects.append(parse_effect_text(effect_text.strip()))
        return Effect(CardEffectType.PARENT, 0, text, child_effects=child_effects, is_scrap_effect=is_scrap)
    
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
    
    # Parse common resource gains, allowing multiple on one line
    gain_patterns = [
        (CardEffectType.TRADE,   r"\{Gain (\d+) Trade\}"),
        (CardEffectType.COMBAT,  r"\{Gain (\d+) Combat\}"),
        (CardEffectType.HEAL,    r"\{Gain (\d+) Authority\}")
    ]
    children = []
    for effect_type, pattern in gain_patterns:
        for match in re.finditer(pattern, text):
            amount = int(match.group(1))
            segment = match.group(0)
            children.append(
                Effect(effect_type, amount, segment,
                       faction_requirement, is_scrap, is_ally, faction_requirement_count)
            )
    if children:
        if len(children) == 1:
            return children[0]
        return Effect(
            CardEffectType.PARENT, 0, text,
            child_effects=children,
            is_scrap_effect=is_scrap
        )
    
    # Parse scrap effects
    if text == "You may scrap a card in your hand or discard pile" or text == "Scrap a card in your hand or discard pile":
        return Effect(CardEffectType.SCRAP, 1, text, faction_requirement, is_scrap, 
                      is_ally, faction_requirement_count, card_targets=["hand", "discard"])
    
    if text == "You may scrap a card in the trade row":
        return Effect(CardEffectType.SCRAP, 1, text, faction_requirement, is_scrap,
                      is_ally, faction_requirement_count, card_targets=["trade"])
    
    # Parse discard effects
    if text == "Target opponent discards a card":
        return Effect(CardEffectType.TARGET_DISCARD, 1, text, faction_requirement, is_scrap,
                      is_ally, faction_requirement_count, card_targets=["opponent"])

    # Parse special destroy base + scrap trade row effect
    if text == "You may destroy target base and/or scrap a card in the trade row":
        return Effect(CardEffectType.PARENT, 1, text, faction_requirement, is_scrap,
                      is_ally, faction_requirement_count, child_effects=[
                          Effect(CardEffectType.DESTROY_BASE, 1, "Destroy target base", faction_requirement, is_scrap,
                                 is_ally, faction_requirement_count),
                          Effect(CardEffectType.SCRAP, 1, "Scrap a card in the trade row", faction_requirement,
                                 is_scrap, is_ally, faction_requirement_count, card_targets=["trade"])
                      ])
    
    # Parse draw then scrap effect
    if text == "Draw a card, then scrap a card from your hand":
        return Effect(CardEffectType.PARENT, 1, text, faction_requirement, is_scrap,
                      is_ally, faction_requirement_count, child_effects=[
                          Effect(CardEffectType.DRAW, 1, "Draw a card", faction_requirement, is_scrap,
                                 is_ally, faction_requirement_count),
                          Effect(CardEffectType.SCRAP, 1, "Scrap a card from your hand", faction_requirement,
                                 is_scrap, is_ally, faction_requirement_count, card_targets=["hand"], is_mandatory=True)
                      ])

    # Parse draw effects
    if text == "Draw a card":
        return Effect(CardEffectType.DRAW, 1, text, faction_requirement, is_scrap, is_ally, faction_requirement_count)
    if text == "Draw two cards":
        return Effect(CardEffectType.DRAW, 2, text, faction_requirement, is_scrap, is_ally, faction_requirement_count)
    
    draw_match = re.search(r"Draw (\d+) cards?", text)
    if draw_match:
        return Effect(CardEffectType.DRAW, int(draw_match.group(1)), text,
                     faction_requirement, is_scrap, is_ally, faction_requirement_count)
    
    # Parse destroy base effects
    if text == "Destroy target base" or text == "You may destroy target base":
        return Effect(CardEffectType.DESTROY_BASE, 1, text, faction_requirement, is_scrap,
                      is_ally, faction_requirement_count)
    
    # Default case - store as text for complex effects
    return Effect(CardEffectType.COMPLEX, 0, text, faction_requirement, is_scrap, is_ally, faction_requirement_count)
