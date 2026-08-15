"""Authoritative scalar interpreter for compiled Effect opcodes."""
from __future__ import annotations

from src.cards.effects import CardEffectType, Effect
from src.rules.ir import (
    EffectOpcode,
    EffectTrigger,
    PENDING_DRAW_ON_COMPLETE,
    PENDING_MANDATORY,
    TARGET_DISCARD,
    TARGET_HAND,
    TARGET_TRADE,
    compile_effect_op,
    compile_effect_trigger,
)


def is_passive_effect(effect: Effect) -> bool:
    return effect.effect_type in {
        CardEffectType.PASSIVE_SHIP_COMBAT,
        CardEffectType.WILDCARD_ALLY,
    }


def _auto_applicable(effect: Effect) -> bool:
    trigger, _, _ = compile_effect_trigger(
        effect,
        inherited_auto=True,
    )
    return trigger in {EffectTrigger.AUTO, EffectTrigger.PASSIVE}


def _queue_scrap(effect, game, player, sources, draw_on_complete=False):
    from src.engine.actions import Action, ActionType, CardSource

    actions = []
    if sources & TARGET_DISCARD:
        actions.extend(
            Action(
                ActionType.SCRAP_CARD,
                card_id=target.index,
                card=target,
                card_source=CardSource.DISCARD,
            )
            for target in player.discard_pile
        )
    if sources & TARGET_HAND:
        actions.extend(
            Action(
                ActionType.SCRAP_CARD,
                card_id=target.index,
                card=target,
                card_source=CardSource.HAND,
            )
            for target in player.hand
        )
    if sources & TARGET_TRADE:
        actions.extend(
            Action(
                ActionType.SCRAP_CARD,
                card_id=target.index,
                card=target,
                card_source=CardSource.TRADE,
            )
            for target in game.trade_row
        )
    if actions:
        player.add_pending_actions(
            actions,
            effect.value,
            bool(sources & PENDING_MANDATORY),
            on_complete_draw=draw_on_complete,
        )


def _queue_destroy(effect, game, player):
    from src.engine.actions import Action, ActionType

    opponent = game.get_opponent(player)
    if opponent is None:
        return
    actions = [
        Action(
            ActionType.DESTROY_BASE,
            target_id=base.index,
            card_id=base.index,
            card=base,
        )
        for base in opponent.bases
    ]
    if actions:
        player.add_pending_actions(
            actions, effect.value, effect.is_mandatory
        )


def _queue_discard_draw(effect, player):
    from src.engine.actions import Action, ActionType, CardSource

    actions = [
        Action(
            ActionType.DISCARD_CARDS,
            card_id=target.index,
            card=target,
            card_source=CardSource.SELF,
        )
        for target in player.hand
    ]
    if actions:
        player.add_pending_actions(
            actions,
            effect.value,
            mandatory=False,
            on_complete_draw=True,
        )


def _queue_free_ship(effect, game, player):
    from src.engine.actions import Action, ActionType

    choices = [
        Action(
            ActionType.BUY_CARD,
            card_id=target.index,
            card=target,
            additional_params={"free": True, "destination": "top"},
        )
        for target in game.trade_row
        if target.card_type.value == "ship"
    ]
    if game.explorer_pile:
        target = game.explorer_pile[-1]
        choices.append(
            Action(
                ActionType.BUY_CARD,
                card_id=target.index,
                card=target,
                additional_params={"free": True, "destination": "top"},
            )
        )
    if choices:
        player.add_pending_actions(choices, 1, mandatory=True)


def _queue_copy(effect, game, player, source_card):
    from src.engine.actions import Action, ActionType

    if source_card is None:
        return
    choices = []
    seen = set()
    for target in player.played_cards:
        if (
            target is source_card
            or target.card_type.value != "ship"
            or id(target) in seen
        ):
            continue
        seen.add(id(target))
        choices.append(
            Action(
                ActionType.APPLY_EFFECT,
                card_id=target.index,
                card=target,
                additional_params={
                    "copy_source": source_card,
                    "copy_target": target,
                },
            )
        )
    if choices:
        player.add_pending_actions(choices, 1, mandatory=True)


def _scrap_source(game, player, card):
    if card is None:
        return
    game.stats.record_card_scrap(player.name, "card")
    if card in player.played_cards:
        player.played_cards.remove(card)
    if card in player.bases:
        player.bases.remove(card)
    player.invalidate_faction_cache()
    if card.name == "Explorer":
        game.explorer_pile.append(card)


def apply_effect(
    effect: Effect,
    game,
    player,
    card=None,
    *,
    auto_only: bool = False,
) -> None:
    """Apply one mutable Effect through the shared opcode contract."""
    if effect.applied:
        return
    if (
        effect.faction_requirement
        and player.get_faction_ally_count(effect.faction_requirement)
        <= effect.faction_requirement_count
    ):
        return

    if effect.effect_type == CardEffectType.PARENT:
        for child in effect.child_effects or ():
            if auto_only and not _auto_applicable(child):
                continue
            apply_effect(child, game, player, card, auto_only=auto_only)
        if effect.is_scrap_effect:
            _scrap_source(game, player, card)
        return
    if auto_only and not _auto_applicable(effect):
        return

    opcode, value, arg = compile_effect_op(effect)
    if opcode == EffectOpcode.ADD_COMBAT:
        player.combat += value
    elif opcode == EffectOpcode.ADD_TRADE:
        player.trade += value
        game.stats.record_trade(player.name, value)
    elif opcode == EffectOpcode.ADD_HEALTH:
        player.health += value
        game.stats.record_authority_gain(player.name, value)
    elif opcode == EffectOpcode.DRAW:
        for _ in range(value):
            player.draw_card()
            game.stats.record_card_draw(player.name)
    elif opcode == EffectOpcode.QUEUE_SCRAP:
        _queue_scrap(effect, game, player, arg)
    elif opcode == EffectOpcode.ADD_DISCARD_DEBT:
        opponent = game.get_opponent(player)
        if opponent is not None:
            opponent.pending_start_of_turn_discards += value
    elif opcode == EffectOpcode.QUEUE_DESTROY_BASE:
        _queue_destroy(effect, game, player)
    elif opcode == EffectOpcode.DRAW_PER_FACTION:
        count = sum(
            1
            for active in player.played_cards
            if int(active.effective_faction) & arg
        )
        for _ in range(count):
            player.draw_card()
    elif opcode == EffectOpcode.DRAW_IF_BASES_GE_2:
        if len(player.bases) >= arg:
            for _ in range(value):
                player.draw_card()
                game.stats.record_card_draw(player.name)
    elif opcode == EffectOpcode.QUEUE_SCRAP_DRAW:
        _queue_scrap(
            effect,
            game,
            player,
            TARGET_HAND | TARGET_DISCARD,
            draw_on_complete=True,
        )
    elif opcode == EffectOpcode.QUEUE_DISCARD_DRAW:
        _queue_discard_draw(effect, player)
    elif opcode == EffectOpcode.QUEUE_ACQUIRE_SHIP_TOP:
        _queue_free_ship(effect, game, player)
    elif opcode == EffectOpcode.ADD_NEXT_SHIP_TOP:
        player.next_ship_to_top += value
    elif opcode == EffectOpcode.QUEUE_COPY_SHIP:
        _queue_copy(effect, game, player, card)
    elif opcode in {
        EffectOpcode.PASSIVE_SHIP_COMBAT,
        EffectOpcode.WILDCARD_ALLY,
        EffectOpcode.NOP,
    }:
        pass
    else:
        raise AssertionError(f"unhandled scalar opcode {opcode}")

    effect.applied = True

    if effect.is_scrap_effect and card:
        _scrap_source(game, player, card)
