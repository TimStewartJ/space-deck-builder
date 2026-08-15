"""Struct-of-arrays state and scalar-engine conversion for Gate 2."""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Iterable

import numpy as np

from src.engine.game import Game
from src.engine.actions import ActionType, CardSource
from src.vector_engine.rules import CompiledRules, EffectTrigger


ZONE_HAND = 0
ZONE_DECK = 1
ZONE_DISCARD = 2
ZONE_PLAYED = 3
ZONE_BASES = 4
NUM_ZONES = 5
NUM_PLAYERS = 2
MAX_CARD_INSTANCES = 16
MAX_ORDERED_TOP = 128
INSTANCE_ACTIVE = np.uint16(1 << 15)
INSTANCE_PLAYED_THIS_TURN = np.uint16(1 << 14)
MAX_PENDING_SETS = 4
PENDING_NONE = 0
PENDING_SCRAP = 1
PENDING_DESTROY_BASE = 2
PENDING_DISCARD = 3
PENDING_ACQUIRE_SHIP_TOP = 4
PENDING_COPY_SHIP = 5
SOURCE_HAND = 1
SOURCE_DISCARD = 2
SOURCE_TRADE = 4
PENDING_TARGET_HAND = 0
PENDING_TARGET_DISCARD = 1
PENDING_TARGET_TRADE = 2
PENDING_TARGET_BASE = 3
PENDING_TARGET_GROUPS = 4

NO_CARD = -1
NO_WINNER = -1


@dataclass
class VectorState:
    games: int
    num_cards: int
    zones: np.ndarray
    trade: np.ndarray
    combat: np.ndarray
    health: np.ndarray
    authority_gained: np.ndarray
    trade_row: np.ndarray
    trade_deck: np.ndarray
    explorer_count: np.ndarray
    current_player: np.ndarray
    first_player: np.ndarray
    training_player: np.ndarray
    turns: np.ndarray
    done: np.ndarray
    winner: np.ndarray
    effect_state: np.ndarray
    instance_rule_card: np.ndarray
    instance_faction: np.ndarray
    pending_mode: np.ndarray
    pending_remaining: np.ndarray
    pending_sources: np.ndarray
    pending_mandatory: np.ndarray
    pending_draw_on_complete: np.ndarray
    pending_resolved: np.ndarray
    pending_targets: np.ndarray
    pending_source_card: np.ndarray
    pending_source_instance: np.ndarray
    pending_count: np.ndarray
    pending_overflow: np.ndarray
    discard_debt: np.ndarray
    ordered_top: np.ndarray
    ordered_top_count: np.ndarray
    next_ship_to_top: np.ndarray
    rng_state: np.ndarray
    hand_size: np.ndarray
    turn_cap: np.ndarray

    @classmethod
    def empty(cls, games: int, num_cards: int) -> "VectorState":
        if games <= 0:
            raise ValueError(f"games must be positive, got {games}")
        return cls(
            games=games,
            num_cards=num_cards,
            zones=np.zeros(
                (games, NUM_PLAYERS, NUM_ZONES, num_cards), dtype=np.int16
            ),
            trade=np.zeros((games, NUM_PLAYERS), dtype=np.int16),
            combat=np.zeros((games, NUM_PLAYERS), dtype=np.int16),
            health=np.zeros((games, NUM_PLAYERS), dtype=np.int16),
            authority_gained=np.zeros(
                (games, NUM_PLAYERS), dtype=np.int16
            ),
            trade_row=np.full((games, 5), NO_CARD, dtype=np.int16),
            trade_deck=np.zeros((games, num_cards), dtype=np.int16),
            explorer_count=np.zeros(games, dtype=np.int16),
            current_player=np.zeros(games, dtype=np.int8),
            first_player=np.zeros(games, dtype=np.int8),
            training_player=np.zeros(games, dtype=np.int8),
            turns=np.zeros(games, dtype=np.int16),
            done=np.zeros(games, dtype=np.bool_),
            winner=np.full(games, NO_WINNER, dtype=np.int8),
            effect_state=np.zeros(
                (
                    games,
                    NUM_PLAYERS,
                    num_cards,
                    MAX_CARD_INSTANCES,
                ),
                dtype=np.uint16,
            ),
            instance_rule_card=np.full(
                (
                    games,
                    NUM_PLAYERS,
                    num_cards,
                    MAX_CARD_INSTANCES,
                ),
                NO_CARD,
                dtype=np.int16,
            ),
            instance_faction=np.zeros(
                (
                    games,
                    NUM_PLAYERS,
                    num_cards,
                    MAX_CARD_INSTANCES,
                ),
                dtype=np.int8,
            ),
            pending_mode=np.zeros(
                (games, NUM_PLAYERS, MAX_PENDING_SETS), dtype=np.int8
            ),
            pending_remaining=np.zeros(
                (games, NUM_PLAYERS, MAX_PENDING_SETS), dtype=np.int8
            ),
            pending_sources=np.zeros(
                (games, NUM_PLAYERS, MAX_PENDING_SETS), dtype=np.int16
            ),
            pending_mandatory=np.zeros(
                (games, NUM_PLAYERS, MAX_PENDING_SETS), dtype=np.bool_
            ),
            pending_draw_on_complete=np.zeros(
                (games, NUM_PLAYERS, MAX_PENDING_SETS), dtype=np.bool_
            ),
            pending_resolved=np.zeros(
                (games, NUM_PLAYERS, MAX_PENDING_SETS), dtype=np.int8
            ),
            pending_targets=np.zeros(
                (
                    games,
                    NUM_PLAYERS,
                    MAX_PENDING_SETS,
                    PENDING_TARGET_GROUPS,
                    num_cards,
                ),
                dtype=np.int8,
            ),
            pending_source_card=np.full(
                (games, NUM_PLAYERS, MAX_PENDING_SETS),
                NO_CARD,
                dtype=np.int16,
            ),
            pending_source_instance=np.full(
                (games, NUM_PLAYERS, MAX_PENDING_SETS),
                -1,
                dtype=np.int8,
            ),
            pending_count=np.zeros(
                (games, NUM_PLAYERS), dtype=np.int8
            ),
            pending_overflow=np.zeros(games, dtype=np.bool_),
            discard_debt=np.zeros(
                (games, NUM_PLAYERS), dtype=np.int8
            ),
            ordered_top=np.full(
                (games, NUM_PLAYERS, MAX_ORDERED_TOP),
                NO_CARD,
                dtype=np.int16,
            ),
            ordered_top_count=np.zeros(
                (games, NUM_PLAYERS), dtype=np.int16
            ),
            next_ship_to_top=np.zeros(
                (games, NUM_PLAYERS), dtype=np.int8
            ),
            rng_state=(
                (np.arange(games, dtype=np.int64) + 1) * 1_103_515_245
            )
            & 0x7FFF_FFFF,
            hand_size=np.full(games, 5, dtype=np.int8),
            turn_cap=np.full(games, 1000, dtype=np.int16),
        )

    def copy(self) -> "VectorState":
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = (
                value.copy() if isinstance(value, np.ndarray) else value
            )
        return VectorState(**values)


def _count_cards(cards: Iterable, num_cards: int) -> np.ndarray:
    counts = np.zeros(num_cards, dtype=np.int16)
    for card in cards:
        index = int(card.index)
        if not 0 <= index < num_cards:
            raise ValueError(f"card index {index} outside 0..{num_cards - 1}")
        counts[index] += 1
    return counts


def _player_index(game: Game, player) -> int:
    return game.players.index(player)


def _leaf_effects(effect) -> list:
    if effect.child_effects:
        leaves = []
        for child in effect.child_effects:
            leaves.extend(_leaf_effects(child))
        return leaves
    return [effect]


def _effect_instance_state(
    game: Game,
    player_index: int,
    rules: CompiledRules,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode ordered per-copy activation readiness for cards in play."""
    states = np.zeros(
        (rules.num_cards, MAX_CARD_INSTANCES), dtype=np.uint16
    )
    rule_cards = np.full(
        (rules.num_cards, MAX_CARD_INSTANCES), NO_CARD, dtype=np.int16
    )
    factions = np.zeros(
        (rules.num_cards, MAX_CARD_INSTANCES), dtype=np.int8
    )
    next_slot = np.zeros(rules.num_cards, dtype=np.int8)
    player = game.players[player_index]
    played_object_ids = {id(card) for card in player.played_cards}
    seen: set[int] = set()
    for card in list(player.played_cards) + list(player.bases):
        object_id = id(card)
        if object_id in seen:
            continue
        seen.add(object_id)
        card_index = int(card.index)
        if not rules.supported[card_index]:
            continue
        slot = int(next_slot[card_index])
        if slot >= MAX_CARD_INSTANCES:
            raise ValueError(
                f"more than {MAX_CARD_INSTANCES} active copies of "
                f"{rules.card_names[card_index]}"
            )
        value = int(INSTANCE_ACTIVE)
        if object_id in played_object_ids:
            value |= int(INSTANCE_PLAYED_THIS_TURN)
        rule_card = int(card.effective_rule_card_index)
        source_offsets: dict[int, int] = {}
        for bundle in range(int(rules.bundle_count[rule_card])):
            source = int(
                rules.bundle_source_effect[rule_card, bundle]
            )
            if source < 0 or source >= len(card.effects):
                continue
            effect = card.effects[source]
            leaves = _leaf_effects(effect)
            offset = source_offsets.get(source, 0)
            count = int(rules.bundle_op_count[rule_card, bundle])
            relevant = leaves[offset : offset + count]
            source_offsets[source] = offset + count
            available = any(
                not leaf.all_child_effects_used() for leaf in relevant
            )
            if (
                available
                and rules.bundle_trigger[rule_card, bundle]
                != int(EffectTrigger.PASSIVE)
            ):
                value |= 1 << bundle
        states[card_index, slot] = value
        rule_cards[card_index, slot] = rule_card
        factions[card_index, slot] = int(card.effective_faction)
        next_slot[card_index] += 1
    return states, rule_cards, factions


def _active_instance_locations(player) -> dict[int, tuple[int, int]]:
    locations: dict[int, tuple[int, int]] = {}
    next_slot: dict[int, int] = {}
    seen: set[int] = set()
    for card in list(player.played_cards) + list(player.bases):
        object_id = id(card)
        if object_id in seen:
            continue
        seen.add(object_id)
        slot = next_slot.get(card.index, 0)
        locations[object_id] = (card.index, slot)
        next_slot[card.index] = slot + 1
    return locations


def _pending_state(player, rules: CompiledRules):
    modes = np.zeros(MAX_PENDING_SETS, dtype=np.int8)
    remaining = np.zeros(MAX_PENDING_SETS, dtype=np.int8)
    sources = np.zeros(MAX_PENDING_SETS, dtype=np.int16)
    mandatory = np.zeros(MAX_PENDING_SETS, dtype=np.bool_)
    draw_on_complete = np.zeros(MAX_PENDING_SETS, dtype=np.bool_)
    resolved = np.zeros(MAX_PENDING_SETS, dtype=np.int8)
    targets = np.zeros(
        (MAX_PENDING_SETS, PENDING_TARGET_GROUPS, rules.num_cards),
        dtype=np.int8,
    )
    source_card = np.full(MAX_PENDING_SETS, NO_CARD, dtype=np.int16)
    source_instance = np.full(MAX_PENDING_SETS, -1, dtype=np.int8)
    locations = _active_instance_locations(player)
    if len(player.pending_action_sets) > MAX_PENDING_SETS:
        raise ValueError(
            f"{len(player.pending_action_sets)} pending sets exceed "
            f"MAX_PENDING_SETS={MAX_PENDING_SETS}"
        )
    for index, pending in enumerate(player.pending_action_sets):
        remaining[index] = pending.decisions_left
        mandatory[index] = pending.mandatory
        draw_on_complete[index] = pending.on_complete_draw
        resolved[index] = pending.resolved_count
        if not pending.actions:
            # An optional set can outlive its snapshotted targets and expose
            # only SKIP. Mode/source no longer affect that legal action.
            modes[index] = PENDING_SCRAP
            continue
        action_type = pending.actions[0].type
        if action_type == ActionType.SCRAP_CARD:
            modes[index] = PENDING_SCRAP
            for action in pending.actions:
                if action.card_source == CardSource.HAND:
                    sources[index] |= SOURCE_HAND
                    targets[
                        index, PENDING_TARGET_HAND, action.card_id
                    ] += 1
                elif action.card_source == CardSource.DISCARD:
                    sources[index] |= SOURCE_DISCARD
                    targets[
                        index, PENDING_TARGET_DISCARD, action.card_id
                    ] += 1
                elif action.card_source == CardSource.TRADE:
                    sources[index] |= SOURCE_TRADE
                    targets[
                        index, PENDING_TARGET_TRADE, action.card_id
                    ] += 1
        elif action_type == ActionType.DESTROY_BASE:
            modes[index] = PENDING_DESTROY_BASE
            for action in pending.actions:
                targets[
                    index, PENDING_TARGET_BASE, action.card_id
                ] += 1
        elif action_type == ActionType.DISCARD_CARDS:
            modes[index] = PENDING_DISCARD
            sources[index] = SOURCE_HAND
            for action in pending.actions:
                targets[
                    index, PENDING_TARGET_HAND, action.card_id
                ] += 1
        elif (
            action_type == ActionType.BUY_CARD
            and pending.actions[0].additional_params
            and pending.actions[0].additional_params.get("free")
        ):
            modes[index] = PENDING_ACQUIRE_SHIP_TOP
            for action in pending.actions:
                targets[
                    index, PENDING_TARGET_HAND, action.card_id
                ] += 1
        elif (
            action_type == ActionType.APPLY_EFFECT
            and pending.actions[0].additional_params
            and pending.actions[0].additional_params.get("copy_source")
            is not None
        ):
            modes[index] = PENDING_COPY_SHIP
            source = pending.actions[0].additional_params["copy_source"]
            source_card[index], source_instance[index] = locations[id(source)]
            for action in pending.actions:
                targets[
                    index, PENDING_TARGET_HAND, action.card_id
                ] += 1
        else:
            raise ValueError(f"unsupported pending action {action_type}")
    return (
        modes,
        remaining,
        sources,
        mandatory,
        draw_on_complete,
        resolved,
        targets,
        source_card,
        source_instance,
    )


def from_games(
    games: list[Game],
    rules: CompiledRules,
    *,
    training_players: list[int] | None = None,
    rng_seed: int = 0,
) -> VectorState:
    """Convert live scalar games into the canonical SoA representation."""
    state = VectorState.empty(len(games), rules.num_cards)
    training_players = training_players or [0] * len(games)
    if len(training_players) != len(games):
        raise ValueError("training_players length must match games")

    for game_index, game in enumerate(games):
        if len(game.players) != NUM_PLAYERS:
            raise ValueError("Gate 2 supports exactly two players")

        for player_index, player in enumerate(game.players):
            state.zones[game_index, player_index, ZONE_HAND] = _count_cards(
                player.hand, rules.num_cards
            )
            state.zones[game_index, player_index, ZONE_DECK] = _count_cards(
                player.deck, rules.num_cards
            )
            state.zones[
                game_index, player_index, ZONE_DISCARD
            ] = _count_cards(player.discard_pile, rules.num_cards)
            # Scalar state intentionally encodes a newly played base in both
            # played_cards and bases. Preserve that duplicate representation.
            state.zones[
                game_index, player_index, ZONE_PLAYED
            ] = _count_cards(player.played_cards, rules.num_cards)
            state.zones[game_index, player_index, ZONE_BASES] = _count_cards(
                player.bases, rules.num_cards
            )
            state.trade[game_index, player_index] = player.trade
            state.combat[game_index, player_index] = player.combat
            state.health[game_index, player_index] = player.health
            state.authority_gained[
                game_index, player_index
            ] = player.authority_gained
            (
                state.effect_state[game_index, player_index],
                state.instance_rule_card[game_index, player_index],
                state.instance_faction[game_index, player_index],
            ) = _effect_instance_state(game, player_index, rules)
            (
                state.pending_mode[game_index, player_index],
                state.pending_remaining[game_index, player_index],
                state.pending_sources[game_index, player_index],
                state.pending_mandatory[game_index, player_index],
                state.pending_draw_on_complete[game_index, player_index],
                state.pending_resolved[game_index, player_index],
                state.pending_targets[game_index, player_index],
                state.pending_source_card[game_index, player_index],
                state.pending_source_instance[game_index, player_index],
            ) = _pending_state(player, rules)
            state.pending_count[game_index, player_index] = len(
                player.pending_action_sets
            )
            state.discard_debt[
                game_index, player_index
            ] = player.pending_start_of_turn_discards
            state.next_ship_to_top[
                game_index, player_index
            ] = player.next_ship_to_top
            if len(player.ordered_top) > MAX_ORDERED_TOP:
                raise ValueError("ordered top-deck segment exceeds capacity")
            state.ordered_top_count[
                game_index, player_index
            ] = len(player.ordered_top)
            for position, card in enumerate(player.ordered_top):
                state.ordered_top[
                    game_index, player_index, position
                ] = card.index

        if len(game.trade_row) > state.trade_row.shape[1]:
            raise ValueError("trade row exceeds configured width")
        for slot, card in enumerate(game.trade_row):
            state.trade_row[game_index, slot] = card.index
        state.trade_deck[game_index] = _count_cards(
            game.trade_deck, rules.num_cards
        )
        state.explorer_count[game_index] = len(game.explorer_pile)
        state.current_player[game_index] = _player_index(
            game, game.current_player
        )
        state.first_player[game_index] = next(
            (
                index
                for index, player in enumerate(game.players)
                if player.name == game.first_player_name
            ),
            0,
        )
        state.training_player[game_index] = training_players[game_index]
        state.turns[game_index] = game.stats.total_turns
        state.done[game_index] = game.is_game_over
        winner_name = game.get_winner() if game.is_game_over else None
        if winner_name:
            state.winner[game_index] = next(
                (
                    index
                    for index, player in enumerate(game.players)
                    if player.name == winner_name
                ),
                NO_WINNER,
            )
        state.hand_size[game_index] = game.config.hand_size
        state.turn_cap[game_index] = game.config.turn_cap
        state.rng_state[game_index] = (
            (rng_seed + game_index + 1) * 1_103_515_245
        ) & 0x7FFF_FFFF

    return state


def canonical_scalar_snapshot(
    game: Game,
    rules: CompiledRules,
) -> dict[str, np.ndarray | int | bool]:
    """Canonical state used by the differential oracle.

    Deck order is deliberately omitted; Gate 2 compares the multiset and uses
    a draw tape captured from the scalar engine for exact transitions.
    """
    players = []
    for player_index, player in enumerate(game.players):
        effect_state, rule_cards, factions = _effect_instance_state(
            game, player_index, rules
        )
        players.append(
            {
                "hand": _count_cards(player.hand, rules.num_cards),
                "deck": _count_cards(player.deck, rules.num_cards),
                "discard": _count_cards(
                    player.discard_pile, rules.num_cards
                ),
                "played": _count_cards(
                    player.played_cards, rules.num_cards
                ),
                "bases": _count_cards(player.bases, rules.num_cards),
                "trade": player.trade,
                "combat": player.combat,
                "health": player.health,
                "authority_gained": player.authority_gained,
                "effect_state": effect_state,
                "instance_rule_card": rule_cards,
                "instance_faction": factions,
                "pending": _pending_state(player, rules),
                "pending_count": len(player.pending_action_sets),
                "discard_debt": player.pending_start_of_turn_discards,
                "ordered_top": np.asarray(
                    [card.index for card in player.ordered_top],
                    dtype=np.int16,
                ),
                "next_ship_to_top": player.next_ship_to_top,
            }
        )
    return {
        "players": players,
        "trade_row": np.asarray(
            [card.index for card in game.trade_row], dtype=np.int16
        ),
        "trade_deck": _count_cards(game.trade_deck, rules.num_cards),
        "explorer_count": len(game.explorer_pile),
        "current_player": _player_index(game, game.current_player),
        "first_player": next(
            (
                index
                for index, player in enumerate(game.players)
                if player.name == game.first_player_name
            ),
            0,
        ),
        "turns": game.stats.total_turns,
        "done": game.is_game_over,
        "winner": next(
            (
                index
                for index, player in enumerate(game.players)
                if player.name
                == (game.get_winner() if game.is_game_over else None)
            ),
            NO_WINNER,
        ),
    }


def canonical_vector_snapshot(
    state: VectorState,
    game_index: int,
) -> dict[str, np.ndarray | int | bool]:
    players = []
    for player in range(NUM_PLAYERS):
        players.append(
            {
                "hand": state.zones[
                    game_index, player, ZONE_HAND
                ].copy(),
                "deck": state.zones[
                    game_index, player, ZONE_DECK
                ].copy(),
                "discard": state.zones[
                    game_index, player, ZONE_DISCARD
                ].copy(),
                "played": state.zones[
                    game_index, player, ZONE_PLAYED
                ].copy(),
                "bases": state.zones[
                    game_index, player, ZONE_BASES
                ].copy(),
                "trade": int(state.trade[game_index, player]),
                "combat": int(state.combat[game_index, player]),
                "health": int(state.health[game_index, player]),
                "authority_gained": int(
                    state.authority_gained[game_index, player]
                ),
                "effect_state": state.effect_state[
                    game_index, player
                ].copy(),
                "instance_rule_card": state.instance_rule_card[
                    game_index, player
                ].copy(),
                "instance_faction": state.instance_faction[
                    game_index, player
                ].copy(),
                "pending": (
                    state.pending_mode[game_index, player].copy(),
                    state.pending_remaining[game_index, player].copy(),
                    state.pending_sources[game_index, player].copy(),
                    state.pending_mandatory[game_index, player].copy(),
                    state.pending_draw_on_complete[
                        game_index, player
                    ].copy(),
                    state.pending_resolved[game_index, player].copy(),
                    state.pending_targets[game_index, player].copy(),
                    state.pending_source_card[
                        game_index, player
                    ].copy(),
                    state.pending_source_instance[
                        game_index, player
                    ].copy(),
                ),
                "pending_count": int(
                    state.pending_count[game_index, player]
                ),
                "discard_debt": int(
                    state.discard_debt[game_index, player]
                ),
                "ordered_top": state.ordered_top[
                    game_index,
                    player,
                    : state.ordered_top_count[game_index, player],
                ].copy(),
                "next_ship_to_top": int(
                    state.next_ship_to_top[game_index, player]
                ),
            }
        )
    row = state.trade_row[game_index]
    return {
        "players": players,
        "trade_row": row[row != NO_CARD].copy(),
        "trade_deck": state.trade_deck[game_index].copy(),
        "explorer_count": int(state.explorer_count[game_index]),
        "current_player": int(state.current_player[game_index]),
        "first_player": int(state.first_player[game_index]),
        "turns": int(state.turns[game_index]),
        "done": bool(state.done[game_index]),
        "winner": int(state.winner[game_index]),
    }


def assert_canonical_equal(
    scalar: dict,
    vector: dict,
) -> None:
    """Deep exact comparison with a useful first divergence."""
    for key in (
        "trade_row",
        "trade_deck",
        "explorer_count",
        "current_player",
        "first_player",
        "turns",
        "done",
        "winner",
    ):
        left, right = scalar[key], vector[key]
        if isinstance(left, np.ndarray):
            if not np.array_equal(left, right):
                raise AssertionError(
                    f"{key} diverged: scalar={left}, vector={right}"
                )
        elif left != right:
            raise AssertionError(
                f"{key} diverged: scalar={left!r}, vector={right!r}"
            )

    for player in range(NUM_PLAYERS):
        for key in (
            "hand",
            "deck",
            "discard",
            "played",
            "bases",
            "trade",
            "combat",
            "health",
            "authority_gained",
            "effect_state",
            "instance_rule_card",
            "instance_faction",
            "pending",
            "pending_count",
            "discard_debt",
            "ordered_top",
            "next_ship_to_top",
        ):
            left = scalar["players"][player][key]
            right = vector["players"][player][key]
            if isinstance(left, np.ndarray):
                if not np.array_equal(left, right):
                    differing = np.argwhere(left != right)
                    first = tuple(int(value) for value in differing[0])
                    raise AssertionError(
                        f"player {player} {key} diverged at {first}: "
                        f"scalar={left[first]!r}, vector={right[first]!r}"
                    )
            elif isinstance(left, tuple):
                for component, (left_array, right_array) in enumerate(
                    zip(left, right)
                ):
                    if not np.array_equal(left_array, right_array):
                        raise AssertionError(
                            f"player {player} {key}[{component}] diverged: "
                            f"scalar={left_array}, vector={right_array}"
                        )
            elif left != right:
                raise AssertionError(
                    f"player {player} {key} diverged: "
                    f"scalar={left!r}, vector={right!r}"
                )
