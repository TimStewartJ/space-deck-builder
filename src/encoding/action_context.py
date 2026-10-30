"""Unified action context: legal mask + resolvers in a single legality pass.

Replaces the pattern of get_available_actions() → encode_action() per action
→ build mask → decode via list.index(). Instead, one pass produces the mask,
state-encoding flags, and a sparse resolver map for decoding sampled actions.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.engine.actions import Action, ActionType, PendingActionSet
from src.rules.scalar import is_passive_effect
from src.encoding.action_encoder import (
    END_TURN_INDEX,
    SKIP_INDEX,
    encode_action_index,
    get_action_offsets,
)

if TYPE_CHECKING:
    from src.engine.game import Game
    from src.engine.player import Player


@dataclass
class ActionContext:
    """Result of a single legality pass over the game state.

    Attributes:
        mask: bool array of shape (action_dim,) — True for legal actions.
        has_meaningful: True if PLAY_CARD or ATTACK_PLAYER is legal.
        can_buy: True if any BUY_CARD action is legal.
        resolvers: sparse map from encoded action index → representative Action.
            Only populated for indices where mask is True.
    """
    mask: np.ndarray
    has_meaningful: bool = False
    can_buy: bool = False
    resolvers: dict[int, Action] = field(default_factory=dict)


def build_action_context(
    game: 'Game',
    player: 'Player',
    card_index_map: dict[str, int],
    action_dim: int,
    mask_buf: np.ndarray | None = None,
) -> ActionContext:
    """Build legal action mask and resolvers in a single pass.

    This replaces the pattern:
        available = get_available_actions(game, player)
        encoded = [encode_action(a, ...) for a in available]
        mask = build_mask(encoded)
        action = available[encoded.index(sampled)]

    With:
        ctx = build_action_context(game, player, ...)
        # ctx.mask for the model
        # ctx.resolvers[sampled_idx] for the executable Action

    Args:
        game: Current game state.
        player: The player whose actions to enumerate.
        card_index_map: name → index mapping (from registry).
        action_dim: Total action space size.
        mask_buf: Optional pre-allocated bool buffer to reuse.

    Returns:
        ActionContext with mask, flags, and resolvers.
    """
    if mask_buf is not None:
        mask_buf[:] = False
        mask = mask_buf
    else:
        mask = np.zeros(action_dim, dtype=bool)

    resolvers: dict[int, Action] = {}
    has_meaningful = False
    can_buy = False

    # If it's not the player's turn, return empty context
    if game.current_player != player:
        return ActionContext(mask=mask)

    num_cards = len(card_index_map)

    # Offsets come from the action layout, the single source of truth.
    offsets = get_action_offsets(num_cards)
    IDX_END_TURN = END_TURN_INDEX
    IDX_SKIP = SKIP_INDEX
    OFF_PLAY = offsets.play_card
    OFF_BUY = offsets.buy_card
    IDX_ATTACK_PLAYER = offsets.attack_player
    OFF_ATTACK_BASE = offsets.attack_base
    OFF_EFFECT_NONSCRAP = offsets.effect_nonscrap
    OFF_EFFECT_SCRAP = offsets.effect_scrap

    # Handle pending action sets — these override normal action generation
    pending_set: PendingActionSet | None = player.get_current_pending_set()
    if pending_set:
        for act in pending_set.actions:
            idx = encode_action_index(act, offsets)
            if idx > 0 and not mask[idx]:
                mask[idx] = True
                resolvers[idx] = act
        if not pending_set.mandatory:
            mask[IDX_SKIP] = True
            resolvers[IDX_SKIP] = Action(type=ActionType.SKIP_DECISION)
        return ActionContext(mask=mask, has_meaningful=False, can_buy=False,
                            resolvers=resolvers)

    # --- Normal action generation (no pending sets) ---

    # PLAY_CARD: each card in hand
    for card in player.hand:
        ci = card.index
        if ci is not None:
            idx = OFF_PLAY + ci
            if not mask[idx]:
                mask[idx] = True
                resolvers[idx] = Action(type=ActionType.PLAY_CARD,
                                        card=card, card_id=ci)
                has_meaningful = True

    # BUY_CARD: affordable cards in trade row
    for card in game.trade_row:
        if player.trade >= card.cost:
            ci = card.index
            if ci is not None:
                idx = OFF_BUY + ci
                if not mask[idx]:
                    mask[idx] = True
                    resolvers[idx] = Action(type=ActionType.BUY_CARD,
                                            card=card, card_id=ci)
                    can_buy = True

    # BUY Explorer — reference the last explorer (top of pile)
    if game.explorer_pile and player.trade >= game.explorer_pile[-1].cost:
        ci = card_index_map.get("Explorer")
        if ci is not None:
            idx = OFF_BUY + ci
            if not mask[idx]:
                mask[idx] = True
                resolvers[idx] = Action(type=ActionType.BUY_CARD,
                                        card=game.explorer_pile[-1],
                                        card_id=ci)
                can_buy = True

    # ATTACK actions (combat > 0)
    if player.combat > 0:
        for opponent in game.players:
            if opponent is player:
                continue
            outposts = [b for b in opponent.bases if b.is_outpost()]
            if outposts:
                for outpost in outposts:
                    if outpost.defense and player.combat >= outpost.defense:
                        ci = outpost.index
                        if ci is not None:
                            idx = OFF_ATTACK_BASE + ci
                            if not mask[idx]:
                                mask[idx] = True
                                resolvers[idx] = Action(
                                    type=ActionType.ATTACK_BASE,
                                    target_id=ci,
                                    card_id=ci,
                                    card=outpost)
            else:
                for base in opponent.bases:
                    if not base.is_outpost() and base.defense and player.combat >= base.defense:
                        ci = base.index
                        if ci is not None:
                            idx = OFF_ATTACK_BASE + ci
                            if not mask[idx]:
                                mask[idx] = True
                                resolvers[idx] = Action(
                                    type=ActionType.ATTACK_BASE,
                                    target_id=ci,
                                    card_id=ci,
                                    card=base)
                # ATTACK_PLAYER
                mask[IDX_ATTACK_PLAYER] = True
                resolvers[IDX_ATTACK_PLAYER] = Action(
                    type=ActionType.ATTACK_PLAYER,
                    target_id=game.players.index(opponent))
                has_meaningful = True

    # APPLY_EFFECT: unused effects on played cards and bases
    seen_cards: set[int] = set()
    for card in list(player.played_cards) + list(player.bases):
        card_obj_id = id(card)
        if card_obj_id in seen_cards:
            continue
        seen_cards.add(card_obj_id)

        ci = card.index
        if ci is None:
            continue

        for effect in card.effects:
            if is_passive_effect(effect):
                continue
            if effect.all_child_effects_used():
                continue
            if effect.faction_requirement:
                faction_count = player.get_faction_ally_count(effect.faction_requirement)
                if faction_count > effect.faction_requirement_count:
                    _set_effect_mask(mask, resolvers, ci, card, effect,
                                    OFF_EFFECT_NONSCRAP, OFF_EFFECT_SCRAP)
            elif effect.is_or_effect and effect.child_effects and not effect.any_child_effects_used():
                for child_effect in effect.child_effects:
                    _set_effect_mask(mask, resolvers, ci, card, child_effect,
                                    OFF_EFFECT_NONSCRAP, OFF_EFFECT_SCRAP)
            else:
                _set_effect_mask(mask, resolvers, ci, card, effect,
                                OFF_EFFECT_NONSCRAP, OFF_EFFECT_SCRAP)

    # END_TURN — always legal
    mask[IDX_END_TURN] = True
    resolvers[IDX_END_TURN] = Action(type=ActionType.END_TURN)

    return ActionContext(mask=mask, has_meaningful=has_meaningful,
                        can_buy=can_buy, resolvers=resolvers)


def _set_effect_mask(mask, resolvers, card_idx, card, effect,
                     off_nonscrap, off_scrap):
    """Set the mask bit and resolver for an APPLY_EFFECT action."""
    if effect.is_scrap_effect:
        idx = off_scrap + card_idx
    else:
        idx = off_nonscrap + card_idx
    if not mask[idx]:
        mask[idx] = True
        resolvers[idx] = Action(type=ActionType.APPLY_EFFECT,
                                card_id=card_idx, card=card,
                                card_effect=effect)
