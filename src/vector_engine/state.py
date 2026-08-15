"""Struct-of-arrays state and scalar-engine conversion for Gate 2."""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Iterable

import numpy as np

from src.engine.game import Game
from src.vector_engine.rules import CompiledRules


ZONE_HAND = 0
ZONE_DECK = 1
ZONE_DISCARD = 2
ZONE_PLAYED = 3
ZONE_BASES = 4
NUM_ZONES = 5
NUM_PLAYERS = 2
MAX_CARD_INSTANCES = 16
INSTANCE_ACTIVE = np.uint16(1 << 15)
INSTANCE_PLAYED_THIS_TURN = np.uint16(1 << 14)

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


def _effect_instance_state(
    game: Game,
    player_index: int,
    rules: CompiledRules,
) -> np.ndarray:
    """Encode ordered per-copy activation readiness for cards in play."""
    states = np.zeros(
        (rules.num_cards, MAX_CARD_INSTANCES), dtype=np.uint16
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
        for bundle in range(int(rules.bundle_count[card_index])):
            source = int(
                rules.bundle_source_effect[card_index, bundle]
            )
            if source < 0 or source >= len(card.effects):
                continue
            effect = card.effects[source]
            if not effect.all_child_effects_used():
                value |= 1 << bundle
        states[card_index, slot] = value
        next_slot[card_index] += 1
    return states


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
            state.effect_state[
                game_index, player_index
            ] = _effect_instance_state(game, player_index, rules)

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
                "effect_state": _effect_instance_state(
                    game, player_index, rules
                ),
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
            elif left != right:
                raise AssertionError(
                    f"player {player} {key} diverged: "
                    f"scalar={left!r}, vector={right!r}"
                )
