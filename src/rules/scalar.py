"""Authoritative scalar interpreter for compiled Effect opcodes."""
from __future__ import annotations

from src.cards.effects import CardEffectType, Effect
from src.cards.factions import Faction
from src.rules.ir import (
    PASSIVE_EFFECT_TYPES,
    EffectOpcode,
    EffectTrigger,
    PlayResolution,
    PENDING_DRAW_ON_COMPLETE,
    PENDING_MANDATORY,
    TARGET_DISCARD,
    TARGET_HAND,
    TARGET_TRADE,
    ability_play_resolution,
    compile_effect_op,
    compile_effect_trigger,
)


def is_passive_effect(effect: Effect) -> bool:
    return effect.effect_type in PASSIVE_EFFECT_TYPES


def _resolves_on_play(effect: Effect, resolution: PlayResolution) -> bool:
    trigger, _, _ = compile_effect_trigger(effect, resolution=resolution)
    return trigger in {EffectTrigger.AUTO, EffectTrigger.PASSIVE}


def _unresolved_on_play(effect: Effect, resolution: PlayResolution) -> bool:
    """Whether a part of *effect* that resolves on play is not applied yet."""
    if effect.applied:
        return False
    if effect.effect_type == CardEffectType.PARENT:
        return any(
            _resolves_on_play(child, resolution)
            and _unresolved_on_play(child, resolution)
            for child in effect.child_effects or ()
        )
    return _resolves_on_play(effect, resolution)


def _activation_resolution(effect: Effect, card) -> PlayResolution | None:
    """What a player's activation of *card*'s ability *effect* resolves.

    None means all of it. A base's primary ability whose automatic parts
    resolve when the base is played resolves only those parts while they are
    unapplied, which on a later turn is its first activation; its choices
    are a separate activation, as on the turn the base is played (Central
    Office: the trade, then the optional top-deck).
    """
    if card is None or not card.is_base():
        return None
    if not any(effect is ability for ability in card.effects):
        return None
    resolution = ability_play_resolution(effect, ship=False)
    if resolution == PlayResolution.AUTO_PARTS and _unresolved_on_play(
        effect, resolution
    ):
        return resolution
    return None


def resolve_on_play(effect: Effect, game, player, card) -> None:
    """Resolve what of *card*'s top-level ability *effect* happens on play.

    A ship's primary ability resolves completely: its choices queue pending
    decisions (none when a choice has no legal target), and an OR ability
    opens a mandatory choice between its options. A base's primary ability
    resolves its automatic parts, unless a mandatory choice must follow them
    (Machine Base): then the whole ability is one later activation. Nothing
    resolves twice.
    """
    resolution = ability_play_resolution(effect, ship=not card.is_base())
    if resolution == PlayResolution.NONE:
        return
    if effect.is_or_effect:
        # The ability's own flag records that its choice has been offered;
        # choosing an option marks the options used.
        if not effect.applied:
            effect.applied = True
            _queue_option_choice(effect, player, card)
        return
    apply_effect(effect, game, player, card, resolution=resolution)


def _queue_option_choice(effect, player, card):
    from src.engine.actions import Action, ActionType

    player.add_pending_actions(
        [
            Action(
                ActionType.APPLY_EFFECT,
                card_id=card.index,
                card=card,
                card_effect=option,
            )
            for option in effect.child_effects
        ],
        1,
        mandatory=True,
    )


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
    player.remove_owned_card(card)
    player.invalidate_faction_cache()
    if card.name == "Explorer":
        game._return_explorer_to_pile(card)


def _mark_used(effect: Effect) -> None:
    effect.applied = True
    for child in effect.child_effects or ():
        _mark_used(child)


def apply_effect(
    effect: Effect,
    game,
    player,
    card=None,
    *,
    resolution: PlayResolution | None = None,
) -> None:
    """Apply one mutable Effect through the shared opcode contract.

    Without *resolution* the effect is applied as a player-activated ability
    (see :func:`_activation_resolution` for what an activation resolves).
    With one, only the parts that resolve when the card is played under that
    :class:`PlayResolution` are applied (see :func:`resolve_on_play`).

    An OR ability is never applied as a whole. Applying one of its options
    uses the ability: every option is marked used for the rest of the turn.
    """
    if effect.is_or_effect:
        raise ValueError(
            "an OR ability gives exactly one option; apply the chosen option"
        )
    if effect.applied:
        return
    activated = resolution is None
    if activated:
        resolution = _activation_resolution(effect, card)
    if not _apply(effect, game, player, card, resolution=resolution):
        return
    if effect.or_parent is not None and activated:
        for option in effect.or_parent.child_effects:
            _mark_used(option)


def _apply(
    effect: Effect,
    game,
    player,
    card,
    *,
    resolution: PlayResolution | None,
) -> bool:
    """Apply *effect*; return False when its ally requirement is unmet."""
    if effect.applied:
        return True
    if (
        effect.faction_requirement
        and player.get_faction_ally_count(effect.faction_requirement)
        <= effect.faction_requirement_count
    ):
        return False

    if effect.effect_type == CardEffectType.PARENT:
        for child in effect.child_effects or ():
            if resolution is not None and not _resolves_on_play(
                child, resolution
            ):
                continue
            _apply(child, game, player, card, resolution=resolution)
        if effect.is_scrap_effect:
            _scrap_source(game, player, card)
        return True
    if resolution is not None and not _resolves_on_play(effect, resolution):
        return True

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
        # Every card of the faction played this turn counts, also one
        # scrapped since it was played.
        for _ in range(player.cards_played_of_faction(Faction(arg))):
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
    return True
