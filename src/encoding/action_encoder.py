"""Action index layout and Action <-> index encoding.

``get_action_space_layout`` is the single source of truth for how flat
action indices map to semantic groups. Every module that needs an offset,
the action dimension, or the number of cards behind an action dimension
derives it from the layout instead of hard-coding group counts.
"""
from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

from src.cards.effects import CardEffectType, Effect
from src.engine.actions import Action, ActionType, CardSource
from src.utils.logger import log

# Canonical action-index constants. Import these instead of hardcoding
# magic numbers when building or inspecting action masks.
INVALID_INDEX = 0
END_TURN_INDEX = 1
SKIP_INDEX = 2

# Ordered names of the card-indexed action groups; each group holds one action
# per card. The global ``attack_player`` action sits between the first
# ``CARD_GROUPS_BEFORE_ATTACK_PLAYER`` groups and the rest. New groups are
# appended at the end so that every earlier index keeps its position.
CARD_ACTION_TYPE_NAMES = (
    'play_card', 'buy_card', 'attack_base', 'destroy_base',
    'effect_nonscrap', 'effect_scrap',
    'scrap_hand', 'scrap_discard', 'scrap_trade',
    'discard',
    # The second option of a card's OR ability (decision 0016); its first
    # option is the card's ``effect_nonscrap`` action.
    'effect_alt',
)
CARD_GROUPS_BEFORE_ATTACK_PLAYER = 2

# Number of card-indexed action type groups in the encoding scheme.
NUM_CARD_ACTION_TYPES = len(CARD_ACTION_TYPE_NAMES)

# Actions that are not indexed by card: invalid, end_turn, skip, attack_player.
NUM_GLOBAL_ACTIONS = 4


def get_action_space_layout(num_cards: int) -> dict:
    """Return the action space layout as a dict of offsets and indices.

    This is the single source of truth for how action indices map to
    semantic groups. Use this instead of hardcoding offsets.

    Returns a dict with:
        'global_indices': dict mapping global action names to their flat indices
        'card_groups': list of (name, offset, count) for each card-indexed group
        'num_cards': the num_cards used to compute offsets
        'action_dim': the total number of actions
    """
    off = SKIP_INDEX + 1

    card_groups = []
    attack_player_idx = None
    for position, name in enumerate(CARD_ACTION_TYPE_NAMES):
        if position == CARD_GROUPS_BEFORE_ATTACK_PLAYER:
            attack_player_idx = off
            off += 1
        card_groups.append((name, off, num_cards))
        off += num_cards

    global_indices = {
        'invalid': INVALID_INDEX,
        'end_turn': END_TURN_INDEX,
        'skip': SKIP_INDEX,
        'attack_player': attack_player_idx,
    }
    assert len(global_indices) == NUM_GLOBAL_ACTIONS

    return {
        'global_indices': global_indices,
        'card_groups': card_groups,
        'num_cards': num_cards,
        'action_dim': off,
    }


def get_action_space_size(cards: list[str]) -> int:
    """Total number of encoded actions for a card list.

    Args:
        cards: The list of all possible cards in the game.
    """
    return get_action_space_layout(len(cards))['action_dim']


def num_cards_for_action_dim(action_dim: int) -> int:
    """Number of cards whose action layout has *action_dim* actions.

    Raises ``ValueError`` when no card count produces that dimension.
    """
    num_cards, remainder = divmod(
        action_dim - NUM_GLOBAL_ACTIONS, NUM_CARD_ACTION_TYPES
    )
    if (
        num_cards <= 0
        or remainder
        or get_action_space_layout(num_cards)['action_dim'] != action_dim
    ):
        raise ValueError(
            f"action dimension {action_dim} is not canonical for "
            f"{NUM_CARD_ACTION_TYPES} card-indexed action groups"
        )
    return num_cards


class ActionOffsets(NamedTuple):
    """Flat offsets of every card-indexed group, plus ATTACK_PLAYER."""

    play_card: int
    buy_card: int
    attack_player: int
    attack_base: int
    destroy_base: int
    effect_nonscrap: int
    effect_scrap: int
    scrap_hand: int
    scrap_discard: int
    scrap_trade: int
    discard: int
    effect_alt: int
    num_cards: int


@lru_cache(maxsize=None)
def get_action_offsets(num_cards: int) -> ActionOffsets:
    """Cached named view of :func:`get_action_space_layout` for hot paths."""
    layout = get_action_space_layout(num_cards)
    groups = {name: offset for name, offset, _ in layout['card_groups']}
    return ActionOffsets(
        attack_player=layout['global_indices']['attack_player'],
        num_cards=num_cards,
        **groups,
    )


def card_group_of_index(action_idx: int, num_cards: int) -> tuple[str, int] | None:
    """Return ``(group name, card index)`` for a card-indexed action index."""
    for name, offset, count in get_action_space_layout(num_cards)['card_groups']:
        if offset <= action_idx < offset + count:
            return name, action_idx - offset
    return None


def effect_action_group(effect: Effect) -> str:
    """Layout group of the APPLY_EFFECT action that applies *effect*.

    Scrap abilities use ``effect_scrap``. The second option of an OR ability
    uses ``effect_alt``; every other ability, including the first option of
    an OR ability, uses ``effect_nonscrap``.
    """
    if effect.is_scrap_effect:
        return 'effect_scrap'
    option = effect.option_index
    if option is None or option == 0:
        return 'effect_nonscrap'
    if option == 1:
        return 'effect_alt'
    raise ValueError(
        "the action layout encodes at most two options of an OR ability"
    )


def action_slot(action: Action) -> tuple[str, int | None] | None:
    """Return the layout group of *action* and its card index.

    Global actions return their global name with ``None``; card-indexed
    actions return their group name and card index. Returns ``None`` when
    the action has no encoding.
    """
    kind = action.type
    if kind == ActionType.END_TURN:
        return 'end_turn', None
    if kind == ActionType.SKIP_DECISION:
        return 'skip', None
    if kind == ActionType.ATTACK_PLAYER:
        return 'attack_player', None
    card_index = action.card_id
    if card_index is None:
        return None
    if kind == ActionType.PLAY_CARD:
        return 'play_card', card_index
    if kind == ActionType.BUY_CARD:
        return 'buy_card', card_index
    if kind == ActionType.ATTACK_BASE:
        return 'attack_base', card_index
    if kind == ActionType.DESTROY_BASE:
        return 'destroy_base', card_index
    if kind == ActionType.APPLY_EFFECT:
        if action.card_effect is None:
            return 'effect_nonscrap', card_index
        return effect_action_group(action.card_effect), card_index
    if kind == ActionType.SCRAP_CARD:
        if action.card_source == CardSource.HAND:
            return 'scrap_hand', card_index
        if action.card_source == CardSource.DISCARD:
            return 'scrap_discard', card_index
        if action.card_source == CardSource.TRADE:
            return 'scrap_trade', card_index
        return None
    if kind == ActionType.DISCARD_CARDS:
        return 'discard', card_index
    return None


_GLOBAL_SLOTS = {
    'end_turn': END_TURN_INDEX,
    'skip': SKIP_INDEX,
}


def encode_action_index(action: Action, offsets: ActionOffsets) -> int:
    """Encoded index of *action* under *offsets*; 0 when it has none."""
    slot = action_slot(action)
    if slot is None:
        return INVALID_INDEX
    name, card_index = slot
    if card_index is None:
        if name == 'attack_player':
            return offsets.attack_player
        return _GLOBAL_SLOTS[name]
    return getattr(offsets, name) + card_index


def encode_action(action: Action | None, cards: list[str], card_index_map: dict[str, int] = None) -> int:
    """Convert an Action object to a numerical index for neural network processing

    Maps different action types to different index ranges.
    Action.card_id is already an integer card index.

    Returns an integer representation of the action.
    """
    if action is None:
        return INVALID_INDEX
    index = encode_action_index(action, get_action_offsets(len(cards)))
    if index == INVALID_INDEX:
        log(f"Invalid action: {action}")
    return index


def decode_action(action_idx: int, card_names: list[str]) -> Action:
    """Decode an action index into a new Action object.

    Returns Actions with integer card_id (card index) instead of string names.
    The card reference (action.card) is NOT set — callers that need it must
    resolve from live game state.
    """
    if action_idx == SKIP_INDEX:
        return Action(type=ActionType.SKIP_DECISION)
    num_cards = len(card_names)
    offsets = get_action_offsets(num_cards)
    if action_idx == offsets.attack_player:
        return Action(type=ActionType.ATTACK_PLAYER)
    slot = card_group_of_index(action_idx, num_cards)
    if slot is None:
        # Invalid, END_TURN, and out-of-range indices all end the turn.
        return Action(type=ActionType.END_TURN)
    name, card_idx = slot
    if name == 'play_card':
        return Action(type=ActionType.PLAY_CARD, card_id=card_idx)
    if name == 'buy_card':
        return Action(type=ActionType.BUY_CARD, card_id=card_idx)
    if name == 'attack_base':
        return Action(type=ActionType.ATTACK_BASE, card_id=card_idx, target_id=card_idx)
    if name == 'destroy_base':
        return Action(type=ActionType.DESTROY_BASE, card_id=card_idx, target_id=card_idx)
    if name == 'effect_nonscrap':
        return Action(type=ActionType.APPLY_EFFECT, card_id=card_idx, card_effect=Effect(effect_type=CardEffectType.DRAW, is_scrap_effect=False))
    if name == 'effect_scrap':
        return Action(type=ActionType.APPLY_EFFECT, card_id=card_idx, card_effect=Effect(effect_type=CardEffectType.SCRAP, is_scrap_effect=True))
    if name == 'scrap_hand':
        return Action(type=ActionType.SCRAP_CARD, card_id=card_idx, card_source=CardSource.HAND)
    if name == 'scrap_discard':
        return Action(type=ActionType.SCRAP_CARD, card_id=card_idx, card_source=CardSource.DISCARD)
    if name == 'scrap_trade':
        return Action(type=ActionType.SCRAP_CARD, card_id=card_idx, card_source=CardSource.TRADE)
    if name == 'discard':
        return Action(type=ActionType.DISCARD_CARDS, card_id=card_idx, card_source=CardSource.OPPONENT)
    if name == 'effect_alt':
        return Action(type=ActionType.APPLY_EFFECT, card_id=card_idx, card_effect=_second_option())
    raise AssertionError(f"undecodable action group {name!r}")


def _second_option() -> Effect:
    """A placeholder second option of an OR ability, for decoded actions."""
    ability = Effect(
        CardEffectType.PARENT,
        is_or_effect=True,
        child_effects=[Effect(CardEffectType.DRAW), Effect(CardEffectType.DRAW)],
    )
    return ability.child_effects[1]
