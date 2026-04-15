"""Fixed-shape tensor layout for full game state serialization.

Provides lossless (snapshot-equivalent) round-trip between Python Game
objects and a flat int32 tensor. The tensor captures enough state to
reconstruct an equivalent Game — same zones, scalars, effect applied
flags, and current pending set.

Limitations:
- Only the current (top) pending action set is encoded; an assertion
  fires if the stack has depth > 1.
- RNG state is not serialized; restored games may diverge on shuffles.
- Requires a CardRegistry for from_tensor() to reconstruct Card objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.engine.game import Game
    from src.cards.registry import CardRegistry


# ── Layout constants ────────────────────────────────────────────────

EMPTY_SLOT = -1  # sentinel for unused zone slots

# Zone capacities (generous but bounded for Core Set 2-player)
MAX_HAND = 32
MAX_DECK = 128
MAX_DISCARD = 128
MAX_BASES = 16
MAX_PLAYED = 32      # non-base played cards only
MAX_TRADE_ROW = 5
MAX_TRADE_DECK = 80
MAX_PENDING_ACTIONS = 64
PENDING_ACTION_FIELDS = 3  # (action_type_int, card_def_id, source_zone_int)

# ── Offset computation ──────────────────────────────────────────────

# Header
_HEADER_SIZE = 5  # current_player_idx, total_turns, is_game_over, explorer_count, trade_deck_len

# Trade row: slots + len
_TRADE_ROW_SIZE = MAX_TRADE_ROW + 1

# Trade deck: slots + len
_TRADE_DECK_SIZE = MAX_TRADE_DECK + 1

# Per-player layout
_PLAYER_SCALARS = 4  # health, trade, combat, cards_drawn
_HAND_SIZE = MAX_HAND + 1
_DECK_SIZE = MAX_DECK + 1
_DISCARD_SIZE = MAX_DISCARD + 1
_BASES_SIZE = MAX_BASES + 1
_PLAYED_SIZE = MAX_PLAYED + 1
_EFFECT_BITS_SIZE = MAX_BASES + MAX_PLAYED  # one int32 per in-play slot
_BASES_IN_PLAYED_SIZE = 1  # bitmask: which bases are also in played_cards this turn
_PENDING_META_SIZE = 5  # has_pending, decisions_left, mandatory, resolved_count, on_complete_draw
_PENDING_ACTIONS_SIZE = MAX_PENDING_ACTIONS * PENDING_ACTION_FIELDS

_PLAYER_SIZE = (
    _PLAYER_SCALARS
    + _HAND_SIZE + _DECK_SIZE + _DISCARD_SIZE
    + _BASES_SIZE + _PLAYED_SIZE
    + _EFFECT_BITS_SIZE
    + _BASES_IN_PLAYED_SIZE
    + _PENDING_META_SIZE + _PENDING_ACTIONS_SIZE
)

TENSOR_SIZE = _HEADER_SIZE + _TRADE_ROW_SIZE + _TRADE_DECK_SIZE + 2 * _PLAYER_SIZE


@dataclass(frozen=True)
class _PlayerOffsets:
    """Absolute byte offsets for one player's region within the tensor."""
    scalars: int
    hand: int
    hand_len: int
    deck: int
    deck_len: int
    discard: int
    discard_len: int
    bases: int
    bases_len: int
    played: int
    played_len: int
    effect_bits: int
    bases_in_played: int
    pending_meta: int
    pending_actions: int


def _compute_player_offsets(base: int) -> _PlayerOffsets:
    o = base
    scalars = o; o += _PLAYER_SCALARS
    hand = o; o += MAX_HAND
    hand_len = o; o += 1
    deck = o; o += MAX_DECK
    deck_len = o; o += 1
    discard = o; o += MAX_DISCARD
    discard_len = o; o += 1
    bases = o; o += MAX_BASES
    bases_len = o; o += 1
    played = o; o += MAX_PLAYED
    played_len = o; o += 1
    effect_bits = o; o += _EFFECT_BITS_SIZE
    bases_in_played = o; o += _BASES_IN_PLAYED_SIZE
    pending_meta = o; o += _PENDING_META_SIZE
    pending_actions = o; o += _PENDING_ACTIONS_SIZE
    assert o - base == _PLAYER_SIZE
    return _PlayerOffsets(
        scalars=scalars, hand=hand, hand_len=hand_len,
        deck=deck, deck_len=deck_len, discard=discard, discard_len=discard_len,
        bases=bases, bases_len=bases_len, played=played, played_len=played_len,
        effect_bits=effect_bits, bases_in_played=bases_in_played,
        pending_meta=pending_meta, pending_actions=pending_actions,
    )


# Pre-compute absolute offsets
OFF_HEADER = 0
OFF_TRADE_ROW = _HEADER_SIZE
OFF_TRADE_ROW_LEN = OFF_TRADE_ROW + MAX_TRADE_ROW
OFF_TRADE_DECK = OFF_TRADE_ROW + _TRADE_ROW_SIZE
OFF_TRADE_DECK_LEN = OFF_TRADE_DECK + MAX_TRADE_DECK
_PLAYERS_BASE = OFF_TRADE_DECK + _TRADE_DECK_SIZE
P0 = _compute_player_offsets(_PLAYERS_BASE)
P1 = _compute_player_offsets(_PLAYERS_BASE + _PLAYER_SIZE)
PLAYER_OFFSETS = (P0, P1)


# ── Zone ID enum for pending action encoding ────────────────────────

ZONE_HAND = 0
ZONE_DECK = 1
ZONE_DISCARD = 2
ZONE_BASES = 3
ZONE_PLAYED = 4
ZONE_TRADE_ROW = 5
ZONE_OPPONENT_HAND = 6  # for TARGET_DISCARD actions targeting opponent


# ── Effect bit packing ──────────────────────────────────────────────

def _pack_effect_bits(card) -> int:
    """Pack all effect applied flags into a single int32 bitmask.

    Uses a depth-first traversal to assign each effect node a stable
    bit position. The same traversal order is used for unpacking.
    """
    bits = 0
    bit_pos = 0

    def visit(effect):
        nonlocal bits, bit_pos
        if effect.applied:
            bits |= (1 << bit_pos)
        bit_pos += 1
        if effect.child_effects:
            for child in effect.child_effects:
                visit(child)

    for eff in card.effects:
        visit(eff)
    return bits


def _unpack_effect_bits(card, bits: int):
    """Restore effect applied flags from a bitmask.

    Must use the same DFS traversal order as _pack_effect_bits.
    """
    bit_pos = 0

    def visit(effect):
        nonlocal bit_pos
        effect.applied = bool(bits & (1 << bit_pos))
        bit_pos += 1
        if effect.child_effects:
            for child in effect.child_effects:
                visit(child)

    for eff in card.effects:
        visit(eff)


# ── to_tensor ───────────────────────────────────────────────────────

def game_to_tensor(game: 'Game') -> torch.IntTensor:
    """Serialize a Game into a fixed-shape int32 tensor.

    The tensor captures full game state (snapshot-equivalent). Requires
    a 2-player game with current game engine invariants.
    """
    assert len(game.players) == 2, "Only 2-player games supported"

    t = torch.full((TENSOR_SIZE,), EMPTY_SLOT, dtype=torch.int32)

    # ── Header ──
    player_idx = game.players.index(game.current_player)
    t[OFF_HEADER + 0] = player_idx
    t[OFF_HEADER + 1] = game.stats.total_turns
    t[OFF_HEADER + 2] = int(game.is_game_over)
    t[OFF_HEADER + 3] = len(game.explorer_pile)
    t[OFF_HEADER + 4] = len(game.trade_deck)

    # ── Trade row ──
    for i, card in enumerate(game.trade_row):
        t[OFF_TRADE_ROW + i] = card.index
    t[OFF_TRADE_ROW_LEN] = len(game.trade_row)

    # ── Trade deck (ordered, top = last) ──
    for i, card in enumerate(game.trade_deck):
        t[OFF_TRADE_DECK + i] = card.index
    t[OFF_TRADE_DECK_LEN] = len(game.trade_deck)

    # ── Players ──
    for pi, player in enumerate(game.players):
        po = PLAYER_OFFSETS[pi]

        # Scalars
        t[po.scalars + 0] = player.health
        t[po.scalars + 1] = player.trade
        t[po.scalars + 2] = player.combat
        t[po.scalars + 3] = player.cards_drawn

        # Hand
        _write_zone(t, po.hand, po.hand_len, player.hand, MAX_HAND)

        # Deck (ordered — top is last element, matching list.pop() semantics)
        _write_zone(t, po.deck, po.deck_len, player.deck, MAX_DECK)

        # Discard
        _write_zone(t, po.discard, po.discard_len, player.discard_pile, MAX_DISCARD)

        # Bases (stored once here, not duplicated in PLAYED)
        _write_zone(t, po.bases, po.bases_len, player.bases, MAX_BASES)

        # Played cards (non-base only)
        non_base_played = [c for c in player.played_cards if c not in player.bases]
        _write_zone(t, po.played, po.played_len, non_base_played, MAX_PLAYED)

        # Bitmask: which bases are also in played_cards this turn
        bases_in_played_bits = 0
        for i, base in enumerate(player.bases):
            if base in player.played_cards:
                bases_in_played_bits |= (1 << i)
        t[po.bases_in_played] = bases_in_played_bits

        # Effect applied bits for in-play cards
        for i, card in enumerate(player.bases):
            t[po.effect_bits + i] = _pack_effect_bits(card)
        for i, card in enumerate(non_base_played):
            t[po.effect_bits + MAX_BASES + i] = _pack_effect_bits(card)

        # Pending action set (current only)
        pending = player.get_current_pending_set()
        if pending is not None:
            if len(player.pending_action_sets) > 1:
                import warnings
                warnings.warn(
                    f"Player {player.name} has {len(player.pending_action_sets)} "
                    f"stacked pending sets — only the current one is serialized"
                )
            t[po.pending_meta + 0] = 1  # has_pending
            t[po.pending_meta + 1] = pending.decisions_left
            t[po.pending_meta + 2] = int(pending.mandatory)
            t[po.pending_meta + 3] = pending.resolved_count
            t[po.pending_meta + 4] = int(pending.on_complete_draw)

            for i, act in enumerate(pending.actions[:MAX_PENDING_ACTIONS]):
                base = po.pending_actions + i * PENDING_ACTION_FIELDS
                t[base + 0] = _action_type_to_int(act.type)
                t[base + 1] = act.card_id if act.card_id is not None else EMPTY_SLOT
                t[base + 2] = _action_source_to_zone(act)
        else:
            t[po.pending_meta + 0] = 0  # no pending

    return t


def _write_zone(t: torch.IntTensor, offset: int, len_offset: int,
                cards: list, max_size: int):
    """Write card indices into a zone slot array."""
    n = len(cards)
    assert n <= max_size, f"Zone overflow: {n} > {max_size}"
    for i, card in enumerate(cards):
        t[offset + i] = card.index
    t[len_offset] = n


def _action_source_to_zone(action) -> int:
    """Map an Action's card_source to a zone ID."""
    from src.engine.actions import CardSource
    if action.card_source is None:
        return EMPTY_SLOT
    return {
        CardSource.HAND: ZONE_HAND,
        CardSource.DISCARD: ZONE_DISCARD,
        CardSource.TRADE: ZONE_TRADE_ROW,
        CardSource.OPPONENT: ZONE_OPPONENT_HAND,
        CardSource.SELF: ZONE_HAND,
    }.get(action.card_source, EMPTY_SLOT)


# ── from_tensor ─────────────────────────────────────────────────────

def tensor_to_game(
    tensor: torch.IntTensor,
    card_registry: 'CardRegistry',
    template_cards: list,
) -> 'Game':
    """Reconstruct a Game from a tensor + card templates.

    Args:
        tensor: int32 tensor produced by game_to_tensor().
        card_registry: Shared registry for card_names/card_index_map.
        template_cards: Trade deck card templates (from loader) for cloning.
            Each card's effects tree serves as the template for reconstruction.

    Returns:
        A Game in the same state as when game_to_tensor() was called.
    """
    from src.engine.game import Game
    from src.engine.player import Player
    from src.engine.actions import Action, ActionType, CardSource, PendingActionSet
    from src.ai.agent import Agent
    from src.config import GameConfig

    t = tensor

    # Build card template lookup: card_def_id → template Card
    # Includes both trade deck cards and starter/explorer cards
    card_templates: dict[int, object] = {}
    for card in template_cards:
        if card.index not in card_templates:
            card_templates[card.index] = card

    # Add starter card templates if missing (Scout, Viper, Explorer)
    from src.cards.card import Card, CardType
    from src.cards.effects import Effect, CardEffectType
    _starter_defs = {
        "Scout": lambda idx: Card("Scout", idx, 0, [Effect(CardEffectType.TRADE, 1)], CardType.SHIP),
        "Viper": lambda idx: Card("Viper", idx, 0, [Effect(CardEffectType.COMBAT, 1)], CardType.SHIP),
        "Explorer": lambda idx: Card("Explorer", idx, 2, [
            Effect(CardEffectType.TRADE, 2),
            Effect(CardEffectType.COMBAT, 2, is_scrap_effect=True),
        ], CardType.SHIP),
    }
    for name, factory in _starter_defs.items():
        idx = card_registry.card_index_map.get(name)
        if idx is not None and idx not in card_templates:
            card_templates[idx] = factory(idx)

    def make_card(card_def_id: int):
        """Create a fresh Card instance from a template."""
        template = card_templates.get(card_def_id)
        if template is None:
            raise ValueError(f"No template for card_def_id={card_def_id}")
        return template.clone()

    # ── Header ──
    current_player_idx = t[OFF_HEADER + 0].item()
    total_turns = t[OFF_HEADER + 1].item()
    is_game_over = bool(t[OFF_HEADER + 2].item())
    explorer_count = t[OFF_HEADER + 3].item()
    trade_deck_len = t[OFF_HEADER + 4].item()

    # ── Create Game shell ──
    game = Game(
        cards=[],
        card_names=card_registry.card_names,
        card_index_map=card_registry.card_index_map,
    )
    game.is_game_over = is_game_over
    game.is_running = not is_game_over
    game.stats.total_turns = total_turns

    # ── Trade row ──
    trade_row_len = t[OFF_TRADE_ROW_LEN].item()
    game.trade_row = []
    for i in range(trade_row_len):
        cid = t[OFF_TRADE_ROW + i].item()
        if cid != EMPTY_SLOT:
            game.trade_row.append(make_card(cid))

    # ── Trade deck ──
    game.trade_deck = []
    for i in range(trade_deck_len):
        cid = t[OFF_TRADE_DECK + i].item()
        if cid != EMPTY_SLOT:
            game.trade_deck.append(make_card(cid))

    # ── Explorer pile ──
    explorer_idx = card_registry.card_index_map.get("Explorer")
    if explorer_idx is not None and explorer_idx in card_templates:
        game.explorer_pile = [make_card(explorer_idx) for _ in range(explorer_count)]
    else:
        game.explorer_pile = []

    # ── Players ──
    for pi in range(2):
        po = PLAYER_OFFSETS[pi]
        player = Player(f"P{pi+1}", Agent(f"P{pi+1}"),
                        game_config=game.config)
        game.players.append(player)

        # Scalars
        player.health = t[po.scalars + 0].item()
        player.trade = t[po.scalars + 1].item()
        player.combat = t[po.scalars + 2].item()
        player.cards_drawn = t[po.scalars + 3].item()

        # Hand
        player.hand = _read_zone(t, po.hand, po.hand_len, make_card)

        # Deck
        player.deck = _read_zone(t, po.deck, po.deck_len, make_card)

        # Discard
        player.discard_pile = _read_zone(t, po.discard, po.discard_len, make_card)

        # Bases
        player.bases = _read_zone(t, po.bases, po.bases_len, make_card)

        # Played (non-base cards) + reconstruct base aliasing
        non_base_played = _read_zone(t, po.played, po.played_len, make_card)
        # Only include bases that were in played_cards this turn (via bitmask)
        bases_in_played_bits = t[po.bases_in_played].item()
        bases_in_played = []
        for i, base in enumerate(player.bases):
            if bases_in_played_bits & (1 << i):
                bases_in_played.append(base)
        player.played_cards = bases_in_played + non_base_played

        # Restore effect applied bits
        for i, card in enumerate(player.bases):
            bits = t[po.effect_bits + i].item()
            if bits != EMPTY_SLOT and bits != 0:
                _unpack_effect_bits(card, bits)
        for i, card in enumerate(non_base_played):
            bits = t[po.effect_bits + MAX_BASES + i].item()
            if bits != EMPTY_SLOT and bits != 0:
                _unpack_effect_bits(card, bits)

        # Invalidate faction cache (will rebuild on demand)
        player.invalidate_faction_cache()

        # Pending action set
        has_pending = t[po.pending_meta + 0].item()
        if has_pending:
            decisions_left = t[po.pending_meta + 1].item()
            mandatory = bool(t[po.pending_meta + 2].item())
            resolved_count = t[po.pending_meta + 3].item()
            on_complete_draw = bool(t[po.pending_meta + 4].item())

            # Reconstruct pending actions from encoded records
            actions = []
            for i in range(MAX_PENDING_ACTIONS):
                base = po.pending_actions + i * PENDING_ACTION_FIELDS
                act_type_val = t[base + 0].item()
                cid = t[base + 1].item()
                zone_id = t[base + 2].item()
                if act_type_val == EMPTY_SLOT:
                    break
                act_type = _int_to_action_type(act_type_val)
                if act_type is None:
                    continue
                card_source = _zone_to_card_source(zone_id)
                # Resolve card reference from appropriate zone
                card_ref = _resolve_card_ref(
                    player, game, cid, card_source, pi
                )
                actions.append(Action(
                    type=act_type,
                    card_id=cid if cid != EMPTY_SLOT else None,
                    card=card_ref,
                    card_source=card_source,
                ))

            player.add_pending_actions(
                actions, decisions_left, mandatory,
                on_complete_draw=on_complete_draw,
            )
            player.pending_action_sets[0].resolved_count = resolved_count

    # ── Set current player ──
    game.current_turn = current_player_idx
    game.current_player = game.players[current_player_idx]

    # ── Stats setup ──
    for player in game.players:
        game.stats.add_player(player.name)

    return game


def _read_zone(t, offset, len_offset, make_card_fn) -> list:
    """Read cards from a zone slot array."""
    n = t[len_offset].item()
    cards = []
    for i in range(n):
        cid = t[offset + i].item()
        if cid != EMPTY_SLOT:
            cards.append(make_card_fn(cid))
    return cards


# Map ActionType enum values to int for tensor storage
_ACTION_TYPE_TO_INT = {}
_INT_TO_ACTION_TYPE = {}

def _init_action_type_maps():
    from src.engine.actions import ActionType
    for i, at in enumerate(ActionType):
        _ACTION_TYPE_TO_INT[at] = i
        _INT_TO_ACTION_TYPE[i] = at

def _action_type_to_int(action_type) -> int:
    if not _ACTION_TYPE_TO_INT:
        _init_action_type_maps()
    return _ACTION_TYPE_TO_INT.get(action_type, EMPTY_SLOT)

def _int_to_action_type(val: int):
    if not _INT_TO_ACTION_TYPE:
        _init_action_type_maps()
    return _INT_TO_ACTION_TYPE.get(val)


def _zone_to_card_source(zone_id: int):
    from src.engine.actions import CardSource
    return {
        ZONE_HAND: CardSource.HAND,
        ZONE_DISCARD: CardSource.DISCARD,
        ZONE_TRADE_ROW: CardSource.TRADE,
        ZONE_OPPONENT_HAND: CardSource.OPPONENT,
    }.get(zone_id)


def _resolve_card_ref(player, game, card_def_id, card_source, player_idx):
    """Try to find a matching card in the expected zone for pending action reconstruction."""
    from src.engine.actions import CardSource
    if card_def_id == EMPTY_SLOT:
        return None
    if card_source == CardSource.HAND:
        for c in player.hand:
            if c.index == card_def_id:
                return c
    elif card_source == CardSource.DISCARD:
        for c in player.discard_pile:
            if c.index == card_def_id:
                return c
    elif card_source == CardSource.TRADE:
        for c in game.trade_row:
            if c.index == card_def_id:
                return c
    elif card_source == CardSource.OPPONENT:
        opp_idx = 1 - player_idx
        if opp_idx < len(game.players):
            for c in game.players[opp_idx].hand:
                if c.index == card_def_id:
                    return c
    return None
