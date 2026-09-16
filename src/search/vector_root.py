"""Acting-player search roots built from public counts without copying games.

A search root is the vector representation of the acting player's
information set. It is written straight from the live game's visible zones
and public ledgers:

- visible zones (hands of the acting player, discard piles, cards in play,
  the trade row, pending choices of the acting player) are counted directly;
- the acting player's deck is public ownership minus visible zones;
- the opponent's unseen cards are public ownership minus visible zones, held
  as publicly known hand cards plus a canonical allocation of the rest;
- the trade deck is the public trade supply minus the trade row.

Hidden zones of the true game are never read, and nothing is deep copied.
Every root equals the vector conversion of
``project_information_set(game).canonical_game``, and its digest depends
only on public information.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from typing import Sequence

import numpy as np

from src.engine.game import Game
from src.search.information_set import (
    InformationSetError,
    _card_is_pristine,
    _card_signature,
    _physical_play_area,
    _validate_card_identity,
    _validate_pending_visibility,
)
from src.vector_engine.rules import CompiledRules
from src.vector_engine.state import VectorState, _count_cards, write_game_row

_DIGEST_VERSION = b"search-root-v1"
_DIGEST_FIELDS = tuple(
    field.name
    for field in fields(VectorState)
    if field.name not in ("games", "num_cards", "rng_state")
)
_DIGEST_DTYPES: dict[tuple[str, str], bytes] = {}


@dataclass(frozen=True)
class SearchRoot:
    """Public facts about one root row that search needs besides the row."""

    viewer: int
    known_hand: np.ndarray
    digest: str


class SearchRootBuilder:
    """Write acting-player roots for many games into one vector population."""

    def __init__(self, rules: CompiledRules):
        self.rules = rules
        self._order_keys: dict[tuple[int, str], str] = {}

    def build(self, games: Sequence[Game]) -> tuple[VectorState, list[SearchRoot]]:
        if not games:
            raise ValueError("at least one game is required")
        state = VectorState.empty(len(games), self.rules.num_cards)
        roots = [self._write(state, row, game) for row, game in enumerate(games)]
        return state, roots

    def _write(self, state: VectorState, row: int, game: Game) -> SearchRoot:
        num_cards = self.rules.num_cards
        if len(game.players) != 2:
            raise InformationSetError("information search requires two players")
        _validate_card_identity(game)
        viewer_player = game.current_player
        viewer = game.players.index(viewer_player)
        if not game.public_trade_supply_initialized:
            raise InformationSetError(
                "game has no initialized public trade-supply ledger"
            )
        if any(not player.public_ownership_initialized for player in game.players):
            raise InformationSetError(
                "game has no initialized public ownership ledger"
            )
        _validate_pending_visibility(game, viewer_player)

        allocations = []
        known_hand = np.zeros(num_cards, dtype=np.int64)
        for player_index, player in enumerate(game.players):
            owned = self._ledger_counts(player.public_owned_cards)
            visible = (
                _count_cards(player.discard_pile, num_cards)
                + _count_cards(_physical_play_area(player), num_cards)
            ).astype(np.int64)
            top = _count_cards(player.ordered_top, num_cards).astype(np.int64)
            if player_index == viewer:
                hand = _count_cards(player.hand, num_cards).astype(np.int64)
                deck = owned - visible - hand
                if (deck < 0).any():
                    raise InformationSetError(
                        f"{player.name} visible zones exceed public ownership"
                    )
                if int(deck.sum()) != len(player.deck):
                    raise InformationSetError(
                        f"{player.name} public ownership does not match deck size"
                    )
                if (deck - top < 0).any():
                    raise InformationSetError(
                        f"{player.name} known top-deck segment is not in the deck"
                    )
            else:
                unseen = owned - visible
                if (unseen < 0).any():
                    raise InformationSetError(
                        f"{player.name} public zones exceed public ownership"
                    )
                if int(unseen.sum()) != len(player.hand) + len(player.deck):
                    raise InformationSetError(
                        f"{player.name} public ownership does not match hidden "
                        "hand and deck sizes"
                    )
                known_hand = _count_cards(
                    player.public_known_hand, num_cards
                ).astype(np.int64)
                unknown = unseen - known_hand - top
                if (unknown < 0).any():
                    raise InformationSetError(
                        f"{player.name} publicly known hand and top-deck cards "
                        "exceed unseen cards"
                    )
                unknown_hand_size = len(player.hand) - int(known_hand.sum())
                unordered_deck_size = len(player.deck) - len(player.ordered_top)
                if unknown_hand_size < 0 or unordered_deck_size < 0:
                    raise InformationSetError(
                        f"{player.name} public constraints exceed hidden zone sizes"
                    )
                if int(unknown.sum()) != unknown_hand_size + unordered_deck_size:
                    raise InformationSetError(
                        f"{player.name} hidden card pool cannot fill hand and deck"
                    )
                hand = known_hand + self._canonical_prefix(
                    game, unknown, unknown_hand_size
                )
                deck = unseen - hand
            allocations.append((hand, deck))

        supply = self._ledger_counts(game.public_trade_supply)
        trade_deck = supply - _count_cards(game.trade_row, num_cards).astype(np.int64)
        if (trade_deck < 0).any() or int(trade_deck.sum()) != len(game.trade_deck):
            raise InformationSetError(
                "public trade supply does not match trade-deck size"
            )

        write_game_row(
            state,
            row,
            game,
            self.rules,
            training_player=viewer,
            hand_deck_counts=(allocations[0], allocations[1]),
            trade_deck_counts=trade_deck,
        )
        state.rng_state[row] = 1
        return SearchRoot(
            viewer=viewer,
            known_hand=known_hand,
            digest=row_digest(state, row, viewer, known_hand),
        )

    def _ledger_counts(self, counter) -> np.ndarray:
        counts = np.zeros(self.rules.num_cards, dtype=np.int64)
        for index, count in counter.items():
            if not 0 <= int(index) < self.rules.num_cards:
                raise InformationSetError(
                    f"public ledger card index {index} is outside the card set"
                )
            counts[int(index)] += int(count)
        return counts

    def _canonical_prefix(self, game: Game, unknown: np.ndarray, size: int) -> np.ndarray:
        """First *size* unknown cards in the information set's canonical order."""
        prefix = np.zeros_like(unknown)
        if size == 0:
            return prefix
        indices = [int(index) for index in np.flatnonzero(unknown)]
        for index in sorted(indices, key=lambda value: self._order_key(game, value)):
            taken = min(int(unknown[index]), size)
            prefix[index] = taken
            size -= taken
            if size == 0:
                break
        return prefix

    def _order_key(self, game: Game, index: int) -> str:
        template = game.card_templates.get(index)
        if template is None:
            raise InformationSetError(
                f"no printed card template is registered for index {index}"
            )
        cache_key = (index, template.name)
        key = self._order_keys.get(cache_key)
        if key is None:
            if not _card_is_pristine(template):
                raise InformationSetError(
                    f"card template for index {index} carries mutable state"
                )
            key = json.dumps(
                _card_signature(template), sort_keys=True, separators=(",", ":")
            )
            self._order_keys[cache_key] = key
        return key


def row_digest(state: VectorState, row: int, viewer: int, known_hand: np.ndarray) -> str:
    """SHA-256 over a root row's public fields, its viewer, and known hand cards."""
    digest = hashlib.sha256(_DIGEST_VERSION)
    digest.update(np.asarray([viewer, state.num_cards], dtype=np.int64).tobytes())
    for name in _DIGEST_FIELDS:
        array = getattr(state, name)
        label = _DIGEST_DTYPES.get((name, array.dtype.str))
        if label is None:
            label = f"{name}:{array.dtype.str}".encode("ascii")
            _DIGEST_DTYPES[(name, array.dtype.str)] = label
        digest.update(label)
        digest.update(array[row].tobytes())
    digest.update(np.asarray(known_hand, dtype=np.int64).tobytes())
    return digest.hexdigest()
