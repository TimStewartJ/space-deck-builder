"""Seeded hidden-hand sampling for vector search worlds.

A search population starts from the vector conversion of an acting player's
projected information set, whose hidden allocations are canonical rather than
true. Deck and trade-deck order never exist in vector state: draws and market
refills sample from counts with each row's own RNG. The only hidden allocation
that must be sampled is which of the opponent's unseen cards are in hand.
"""
from __future__ import annotations

import hashlib

import numpy as np

from src.search.information_set import InformationSet, InformationSetError
from src.vector_engine.state import (
    NUM_PLAYERS,
    ZONE_DECK,
    ZONE_HAND,
    VectorState,
)

_RNG_MASK = 0x7FFF_FFFF


def derive_search_seed(seed: int, fingerprint: str) -> int:
    """Combine a search seed with a public information fingerprint."""
    digest = hashlib.sha256(f"{int(seed)}:{fingerprint}".encode("ascii"))
    return int.from_bytes(digest.digest()[:8], "little")


def world_rng_state(base_seed: int, world: int) -> int:
    """Nonzero vector RNG state for one search world."""
    digest = hashlib.sha256(f"{int(base_seed)}:{int(world)}".encode("ascii"))
    return (int.from_bytes(digest.digest()[:4], "little") & _RNG_MASK) or 1


def known_hand_counts(
    information: InformationSet, player: int, num_cards: int
) -> np.ndarray:
    """Publicly known hand cards of *player* as a count vector."""
    counts = np.zeros(num_cards, dtype=np.int64)
    for index, count in information.public_known_hand_counts[player]:
        counts[index] += count
    return counts


def tile_state(state: VectorState, repeats: int) -> VectorState:
    """Replicate every game row ``repeats`` times, preserving row order."""
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")
    values = {}
    for name, value in vars(state).items():
        if isinstance(value, np.ndarray):
            values[name] = np.repeat(value, repeats, axis=0)
        elif name == "games":
            values[name] = state.games * repeats
        else:
            values[name] = value
    return VectorState(**values)


def sample_world_hands(
    state: VectorState,
    *,
    viewer: int,
    known_hand: np.ndarray,
    worlds: np.ndarray,
    base_seed: int,
) -> None:
    """Sample the opponent's unseen hand and seed each row's RNG, in place.

    Every row must replicate the same projected root. Rows sharing a world id
    receive an identical opponent allocation and RNG state, so candidate
    actions are compared on common random numbers. Publicly known hand cards
    stay in hand and the opponent's known top-deck segment stays in the deck;
    every other unseen card is dealt uniformly without replacement from the
    canonical multiset.
    """
    worlds = np.asarray(worlds, dtype=np.int64)
    if worlds.shape != (state.games,):
        raise ValueError("one world id is required per row")
    if not 0 <= viewer < NUM_PLAYERS:
        raise ValueError(f"viewer must be 0 or 1, got {viewer}")
    opponent = 1 - viewer
    num_cards = state.num_cards
    root_hand = state.zones[0, opponent, ZONE_HAND].astype(np.int64)
    root_deck = state.zones[0, opponent, ZONE_DECK].astype(np.int64)
    for zone, root in ((ZONE_HAND, root_hand), (ZONE_DECK, root_deck)):
        if not (state.zones[:, opponent, zone] == root).all():
            raise ValueError("search rows must replicate one projected root")
    unseen = root_hand + root_deck
    top_cards = state.ordered_top[
        0, opponent, : int(state.ordered_top_count[0, opponent])
    ].astype(np.int64)
    top = np.bincount(top_cards, minlength=num_cards)
    known_hand = np.asarray(known_hand, dtype=np.int64)
    pool = unseen - known_hand - top
    if (pool < 0).any():
        raise InformationSetError(
            "known hand and top-deck cards exceed the opponent's unseen cards"
        )
    unknown_hand = int(root_hand.sum() - known_hand.sum())
    if unknown_hand < 0 or unknown_hand > int(pool.sum()):
        raise InformationSetError("opponent hand size cannot be filled")
    cards = np.repeat(np.arange(num_cards, dtype=np.int64), pool)

    cache: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
    for row, world in enumerate(worlds.tolist()):
        if world not in cache:
            rng = np.random.default_rng([int(base_seed), int(world)])
            chosen = cards[rng.permutation(len(cards))[:unknown_hand]]
            hand = known_hand + np.bincount(chosen, minlength=num_cards)
            cache[world] = (
                hand,
                unseen - hand,
                world_rng_state(base_seed, world),
            )
        hand, deck, rng_state = cache[world]
        state.zones[row, opponent, ZONE_HAND] = hand
        state.zones[row, opponent, ZONE_DECK] = deck
        state.rng_state[row] = rng_state
