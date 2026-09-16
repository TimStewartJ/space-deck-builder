"""Seeded hidden-hand dealing for vector search worlds.

A search population starts from acting-player roots whose hidden allocations
are canonical rather than true. Deck and trade-deck order never exist in
vector state: draws and market refills sample from counts with each row's own
RNG. The only hidden allocation that must be sampled is which of the
opponent's unseen cards are in hand.

Worlds are dealt with a counter-based hash (SplitMix64 finalizers) of the
search seed, world id, and card position, so every world of every decision in
a population is dealt by a few vectorized array operations and the deal of a
world never depends on which other worlds share the population.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from src.search.information_set import InformationSet, InformationSetError
from src.vector_engine.state import (
    NUM_PLAYERS,
    ZONE_DECK,
    ZONE_HAND,
    VectorState,
)

_RNG_MASK = np.uint64(0x7FFF_FFFF)
_GOLDEN = np.uint64(0x9E37_79B9_7F4A_7C15)
_MIX_1 = np.uint64(0xBF58_476D_1CE4_E5B9)
_MIX_2 = np.uint64(0x94D0_49BB_1331_11EB)
_TAG_DEAL = np.uint64(0xD1B5_4A32_D192_ED03)
_TAG_RNG = np.uint64(0x8CB9_2BA7_2F3D_8DD7)
_POSITION = np.uint64(0xA24B_AED4_963E_E407)


def _mix(values: np.ndarray) -> np.ndarray:
    """SplitMix64 finalizer applied elementwise to uint64 values."""
    with np.errstate(over="ignore"):
        z = values + _GOLDEN
        z = (z ^ (z >> np.uint64(30))) * _MIX_1
        z = (z ^ (z >> np.uint64(27))) * _MIX_2
        return z ^ (z >> np.uint64(31))


def derive_search_seed(seed: int, fingerprint: str) -> int:
    """Combine a search seed with a public information fingerprint."""
    digest = hashlib.sha256(f"{int(seed)}:{fingerprint}".encode("ascii"))
    return int.from_bytes(digest.digest()[:8], "little")


def _world_hashes(base_seeds: np.ndarray, worlds: np.ndarray) -> np.ndarray:
    return _mix(_mix(base_seeds.astype(np.uint64) ^ _TAG_DEAL) ^ worlds.astype(np.uint64))


def world_rng_state(base_seed: int, world: int) -> int:
    """Nonzero vector RNG state for one search world."""
    state = int(
        _mix(_world_hashes(np.asarray([base_seed], dtype=np.uint64), np.asarray([world])) ^ _TAG_RNG)[0]
        & _RNG_MASK
    )
    return state or 1


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


def deal_world_hands(
    state: VectorState,
    *,
    starts: Sequence[int],
    stops: Sequence[int],
    viewers: Sequence[int],
    known_hands: np.ndarray,
    base_seeds: Sequence[int],
    worlds: np.ndarray,
) -> None:
    """Deal opponent hands and seed row RNGs for contiguous blocks, in place.

    Block ``b`` covers rows ``starts[b]:stops[b]``; every row of a block must
    replicate that block's projected root. ``worlds`` holds one world id per
    population row. Rows of a block sharing a world id receive an identical
    opponent allocation and RNG state, so candidate actions are compared on
    common random numbers. Publicly known hand cards stay in hand and the
    opponent's known top-deck segment stays in the deck; every other unseen
    card is dealt uniformly without replacement from the root's multiset.
    """
    starts = np.asarray(starts, dtype=np.int64)
    stops = np.asarray(stops, dtype=np.int64)
    viewers = np.asarray(viewers, dtype=np.int64)
    base_seeds = np.asarray([int(seed) for seed in base_seeds], dtype=np.uint64)
    known_hands = np.asarray(known_hands, dtype=np.int64)
    worlds = np.asarray(worlds, dtype=np.int64)
    blocks = len(starts)
    num_cards = state.num_cards
    if not (len(stops) == len(viewers) == len(base_seeds) == blocks) or blocks == 0:
        raise ValueError("block descriptions must be non-empty and aligned")
    if known_hands.shape != (blocks, num_cards):
        raise ValueError("one known-hand count vector is required per block")
    if ((worlds < 0) | (worlds >= 1 << 32)).any():
        raise ValueError("world ids must be in [0, 2**32)")
    sizes = stops - starts
    if (sizes <= 0).any() or (starts[1:] != stops[:-1]).any() or starts[0] != 0:
        raise ValueError("blocks must tile the population contiguously from row 0")
    rows = int(stops[-1])
    if rows != state.games or worlds.shape != (rows,):
        raise ValueError("one world id is required per population row")
    if ((viewers < 0) | (viewers >= NUM_PLAYERS)).any():
        raise ValueError("viewers must be 0 or 1")

    opponents = 1 - viewers
    root_hand = state.zones[starts, opponents, ZONE_HAND].astype(np.int64)
    root_deck = state.zones[starts, opponents, ZONE_DECK].astype(np.int64)
    block_of_row = np.repeat(np.arange(blocks), sizes)
    opponent_of_row = opponents[block_of_row]
    row_index = np.arange(rows)
    if (
        (state.zones[row_index, opponent_of_row, ZONE_HAND] != root_hand[block_of_row]).any()
        or (state.zones[row_index, opponent_of_row, ZONE_DECK] != root_deck[block_of_row]).any()
    ):
        raise ValueError("search rows must replicate one projected root per block")

    unseen = root_hand + root_deck
    top_counts = state.ordered_top_count[starts, opponents].astype(np.int64)
    top_cards = state.ordered_top[starts, opponents].astype(np.int64)
    top_valid = np.arange(top_cards.shape[1])[None, :] < top_counts[:, None]
    top = np.bincount(
        (np.arange(blocks)[:, None] * num_cards + np.where(top_valid, top_cards, 0))[top_valid],
        minlength=blocks * num_cards,
    ).reshape(blocks, num_cards)
    pool = unseen - known_hands - top
    if (pool < 0).any():
        raise InformationSetError(
            "known hand and top-deck cards exceed the opponent's unseen cards"
        )
    pool_sizes = pool.sum(axis=1)
    unknown_hand = root_hand.sum(axis=1) - known_hands.sum(axis=1)
    if ((unknown_hand < 0) | (unknown_hand > pool_sizes)).any():
        raise InformationSetError("opponent hand size cannot be filled")

    combined = block_of_row * (1 << 32) + worlds
    unique_worlds, inverse = np.unique(combined, return_inverse=True)
    block_of_world = unique_worlds >> 32
    world_ids = unique_worlds & ((1 << 32) - 1)
    world_count = len(unique_worlds)
    width = max(1, int(pool_sizes.max()))

    flat_cards = np.repeat(np.tile(np.arange(num_cards), blocks), pool.ravel())
    card_block = np.repeat(np.arange(blocks), pool_sizes)
    card_position = np.arange(len(flat_cards)) - np.repeat(np.cumsum(pool_sizes) - pool_sizes, pool_sizes)
    cards = np.full((blocks, width), num_cards, dtype=np.int64)
    cards[card_block, card_position] = flat_cards

    hashes = _world_hashes(base_seeds[block_of_world], world_ids)
    positions = np.arange(width, dtype=np.uint64) * _POSITION
    keys = _mix(hashes[:, None] ^ positions[None, :])
    keys[np.arange(width)[None, :] >= pool_sizes[block_of_world][:, None]] = np.iinfo(np.uint64).max
    order = np.argsort(keys, axis=1, kind="stable")
    chosen = np.take_along_axis(cards[block_of_world], order, axis=1)
    taken = np.arange(width)[None, :] < unknown_hand[block_of_world][:, None]
    linear = (np.arange(world_count)[:, None] * num_cards + chosen)[taken]
    hands = known_hands[block_of_world] + np.bincount(
        linear, minlength=world_count * num_cards
    ).reshape(world_count, num_cards)
    rng_states = (_mix(hashes ^ _TAG_RNG) & _RNG_MASK).astype(np.int64)
    rng_states[rng_states == 0] = 1

    state.zones[row_index, opponent_of_row, ZONE_HAND] = hands[inverse]
    state.zones[row_index, opponent_of_row, ZONE_DECK] = unseen[block_of_row] - hands[inverse]
    state.rng_state[:] = rng_states[inverse]


def sample_world_hands(
    state: VectorState,
    *,
    viewer: int,
    known_hand: np.ndarray,
    worlds: np.ndarray,
    base_seed: int,
) -> None:
    """Deal one root's worlds across every row of *state*; see :func:`deal_world_hands`."""
    deal_world_hands(
        state,
        starts=[0],
        stops=[state.games],
        viewers=[viewer],
        known_hands=np.asarray(known_hand, dtype=np.int64)[None, :],
        base_seeds=[base_seed],
        worlds=worlds,
    )
