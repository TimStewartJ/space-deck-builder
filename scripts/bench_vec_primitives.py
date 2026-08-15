"""Measure the array-operation ceiling of a possible vectorized game engine.

This is deliberately not a game engine. It applies deterministic fake state
transitions with the same shapes, dtypes, access patterns, and awkward
primitives that a real vectorized engine would need:

* build a legal-action mask of shape ``[games, 494]``;
* gather card properties by card id;
* scatter cards between ``[games, 2, 5, 49]`` zone-count arrays;
* sample draws without replacement from a deck multiset;
* dispatch between normal play and four pending-choice modes;
* check terminal state and reset completed games.

The benchmark answers only whether those primitives have enough throughput to
justify building a rules-correct vertical slice. It cannot establish that the
rules themselves translate cleanly or predict final training throughput,
because policy inference is intentionally excluded.

Four backends execute one identical deterministic workload:

``numpy``
    Masked/SIMD CPU array operations.
``numba``
    SPMD CPU execution: one imperative game loop per ``prange`` iteration.
``torch-cpu``
    The masked formulation on resident eager CPU tensors.
``torch-cuda``
    The masked formulation on resident eager GPU tensors (ROCm uses ``cuda``
    APIs). This is not a fused GPU compiler ceiling; a poor result rejects only
    eager PyTorch, not generated Triton/CUDA/HIP kernels.

Before timing, every requested backend must produce byte-identical state and
action masks on a verification workload. Timed repetitions are interleaved and
rotated to avoid confounding backend order with machine drift. JIT compilation,
device transfer, state construction, and policy action selection are excluded;
mask construction, RNG, stepping, and synchronization are included.

Examples
--------
Run the intended gate:

    powershell -File scripts/launch_vec_benchmark.ps1

The launcher is load-bearing on this machine: an identical rollout measured
about 3.7x slower as a descendant of the agent runtime, even after its process
priority was raised to Normal.

Small correctness/smoke run:

    python scripts/bench_vec_primitives.py --games 128 --steps 8 --reps 1

Verify implementations without timing:

    python scripts/bench_vec_primitives.py --verify-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Real project dimensions
# ---------------------------------------------------------------------------
NUM_CARDS = 49
ACTION_DIM = 494
NUM_PLAYERS = 2
NUM_ZONES = 5
NUM_SCALARS = 8
HAND_SIZE = 5
NUM_MODES = 5

ZONE_HAND, ZONE_DECK, ZONE_DISCARD, ZONE_PLAYED, ZONE_BASES = range(NUM_ZONES)

PLAY_OFFSET = 3
BUY_OFFSET = PLAY_OFFSET + NUM_CARDS
ATTACK_PLAYER_INDEX = BUY_OFFSET + NUM_CARDS

# Same integer RNG in every backend. The product fits comfortably in int64,
# avoiding backend-specific overflow semantics.
LCG_MULT = 1_664_525
LCG_ADD = 1_013_904_223
LCG_MASK = 0x7FFF_FFFF

# Current full rollout (engine + transport + policy inference), measured outside
# the agent process tree on 2026-08-14. Comparing an engine-only primitive
# ceiling against this is intentionally conservative about the old stack but
# still only an upper bound on final end-to-end speedup: the new stack will
# continue to pay policy inference.
BASELINE_DECISIONS_PER_SEC = 95_000
GATE_SPEEDUP = 10.0

KNOWN_BACKENDS = ("numpy", "numba", "torch-cpu", "torch-cuda")
MUTABLE_FIELDS = (
    "zones",
    "scalars",
    "trade_row",
    "trade_deck",
    "mode",
    "cur_player",
    "done",
    "rng_state",
)


# ---------------------------------------------------------------------------
# Shared struct-of-arrays state
# ---------------------------------------------------------------------------
@dataclass
class VecState:
    games: int
    zones: np.ndarray
    scalars: np.ndarray
    trade_row: np.ndarray
    trade_deck: np.ndarray
    mode: np.ndarray
    cur_player: np.ndarray
    done: np.ndarray
    rng_state: np.ndarray

    # Immutable per-card tables a rule compiler would emit.
    cost: np.ndarray
    eff_trade: np.ndarray
    eff_combat: np.ndarray
    eff_draw: np.ndarray
    choice_kind: np.ndarray
    initial_trade_row: np.ndarray

    @classmethod
    def create(cls, games: int, seed: int = 0) -> "VecState":
        if games <= 0:
            raise ValueError(f"games must be positive, got {games}")

        rng = np.random.default_rng(seed)
        zones = np.zeros(
            (games, NUM_PLAYERS, NUM_ZONES, NUM_CARDS), dtype=np.int8
        )
        scalars = np.zeros((games, NUM_PLAYERS, NUM_SCALARS), dtype=np.int16)

        # Plausible mid-game occupancy. The semantics are intentionally fake;
        # occupancy and mutation volume are what drive primitive throughput.
        zones[:, :, ZONE_HAND, 0] = 2
        zones[:, :, ZONE_HAND, 1] = 1
        zones[:, :, ZONE_HAND, 2] = 1
        zones[:, :, ZONE_HAND, 5] = 1
        zones[:, :, ZONE_DECK, 0] = 5
        zones[:, :, ZONE_DECK, 1] = 2
        zones[:, :, ZONE_DISCARD, 0] = 3
        zones[:, :, ZONE_BASES, 5] = 1

        scalars[:, :, 0] = 3   # trade
        scalars[:, :, 1] = 2   # combat
        scalars[:, :, 2] = 30  # plausible mid-game authority

        mode = np.zeros(games, dtype=np.int8)
        spread = rng.random(games)
        mode[spread > 0.85] = 1
        mode[spread > 0.92] = 2
        mode[spread > 0.96] = 3
        mode[spread > 0.99] = 4

        cost = rng.integers(1, 5, NUM_CARDS, dtype=np.int16)
        eff_trade = np.zeros(NUM_CARDS, dtype=np.int16)
        eff_trade[0] = 3
        eff_trade[1] = 2
        eff_trade[5] = 2
        eff_combat = np.zeros(NUM_CARDS, dtype=np.int16)
        eff_combat[2] = 8
        eff_combat[5] = 3
        eff_draw = np.zeros(NUM_CARDS, dtype=np.int8)
        eff_draw[2] = 1
        choice_kind = np.zeros(NUM_CARDS, dtype=np.int8)
        choice_kind[0] = 1
        choice_kind[2] = 2
        choice_kind[1] = 3
        choice_kind[5] = 4

        game_ids = np.arange(games, dtype=np.int64)[:, None]
        slots = np.arange(5, dtype=np.int64)[None, :]
        trade_row = (
            game_ids * 7 + slots * 11 + seed
        ) % NUM_CARDS

        return cls(
            games=games,
            zones=zones,
            scalars=scalars,
            trade_row=trade_row.astype(np.int8),
            trade_deck=np.full((games, NUM_CARDS), 3, dtype=np.int8),
            mode=mode,
            cur_player=np.zeros(games, dtype=np.int8),
            done=np.zeros(games, dtype=np.bool_),
            rng_state=(
                (np.arange(games, dtype=np.int64) + seed + 1) * 1_103_515_245
            )
            & LCG_MASK,
            cost=cost,
            eff_trade=eff_trade,
            eff_combat=eff_combat,
            eff_draw=eff_draw,
            choice_kind=choice_kind,
            initial_trade_row=trade_row.astype(np.int8).copy(),
        )

    def copy(self) -> "VecState":
        values: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            values[f.name] = value.copy() if isinstance(value, np.ndarray) else value
        return VecState(**values)


@dataclass(frozen=True)
class Snapshot:
    arrays: dict[str, np.ndarray]
    mask: np.ndarray
    mask_trace: np.ndarray | None = None

    def digest(self) -> str:
        """Collision-resistant identity over every field, dtype, and shape."""
        digest = hashlib.blake2b(digest_size=32)
        for name in (*MUTABLE_FIELDS, "mask"):
            array = self.mask if name == "mask" else self.arrays[name]
            contiguous = np.ascontiguousarray(array)
            digest.update(name.encode("ascii"))
            digest.update(contiguous.dtype.str.encode("ascii"))
            digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
            digest.update(contiguous.tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class Result:
    backend: str
    decisions_per_sec: float
    step_ms: float
    digest: str
    elapsed_s: float


@dataclass(frozen=True)
class AggregateResult:
    backend: str
    decisions_per_sec: float
    step_ms: float
    digest: str
    repetitions: tuple[float, ...]


@dataclass
class WorkloadProfile:
    decisions: int = 0
    pending_modes: int = 0
    mode_1: int = 0
    mode_2: int = 0
    mode_3: int = 0
    mode_4: int = 0
    plays: int = 0
    draw_plays: int = 0
    buys: int = 0
    attacks: int = 0
    end_turns: int = 0
    resets: int = 0

    def rate(self, count: int) -> float:
        return count / max(1, self.decisions)

    def as_dict(self) -> dict[str, int]:
        return {
            f.name: int(getattr(self, f.name))
            for f in fields(self)
        }


def build_workload(
    st: VecState, steps: int
) -> tuple[np.ndarray, WorkloadProfile]:
    """Precompute legal, branch-diverse actions with an untimed reference run.

    Action selection belongs to policy inference, not the environment. The
    reference NumPy simulation chooses candidates from the legal mask and
    advances a private copy, so expensive branches remain live over long runs
    instead of degenerating into illegal no-ops.
    """
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")

    reference = st.copy()
    mask = np.zeros((st.games, ACTION_DIM), dtype=np.bool_)
    actions = np.empty((steps, st.games), dtype=np.int32)
    games = np.arange(st.games)
    profile = WorkloadProfile()

    for step in range(steps):
        build_mask_numpy(reference, mask)
        row = actions[step]
        row[:] = 0

        pending = reference.mode != 0
        row[pending] = 1
        profile.pending_modes += int(pending.sum())
        profile.mode_1 += int((reference.mode == 1).sum())
        profile.mode_2 += int((reference.mode == 2).sum())
        profile.mode_3 += int((reference.mode == 3).sum())
        profile.mode_4 += int((reference.mode == 4).sum())

        main = ~pending
        phase = (games + step) % 5
        play_mask = mask[:, PLAY_OFFSET : PLAY_OFFSET + NUM_CARDS]
        buy_mask = mask[:, BUY_OFFSET : BUY_OFFSET + NUM_CARDS]

        # Phase 1: any legal play.
        first_play = play_mask.argmax(axis=1)
        selected = main & (phase == 1)
        preferred_cards = np.asarray((0, 1, 2, 5), dtype=np.intp)
        preferred = preferred_cards[(games + step) % len(preferred_cards)]
        preferred_legal = selected & mask[
            games, PLAY_OFFSET + preferred
        ]
        row[preferred_legal] = PLAY_OFFSET + preferred[preferred_legal]
        fallback = selected & ~preferred_legal & play_mask.any(axis=1)
        row[fallback] = PLAY_OFFSET + first_play[fallback]

        # Phase 2: prefer the draw card (id 2), otherwise any legal play.
        selected = main & (phase == 2)
        draw_legal = selected & mask[:, PLAY_OFFSET + 2]
        row[draw_legal] = PLAY_OFFSET + 2
        fallback = selected & ~draw_legal & play_mask.any(axis=1)
        row[fallback] = PLAY_OFFSET + first_play[fallback]

        # Phase 3: any currently legal market buy.
        selected = main & (phase == 3) & buy_mask.any(axis=1)
        first_buy = buy_mask.argmax(axis=1)
        row[selected] = BUY_OFFSET + first_buy[selected]

        # Phase 4: attack if legal; otherwise play to build combat.
        selected = main & (phase == 4)
        attack = selected & mask[:, ATTACK_PLAYER_INDEX]
        row[attack] = ATTACK_PLAYER_INDEX
        fallback = selected & ~attack & play_mask.any(axis=1)
        row[fallback] = PLAY_OFFSET + first_play[fallback]

        # Phase 0 and all fallbacks end the turn, which is always legal in
        # main mode. Every chosen non-mode action came from the current mask.
        profile.decisions += st.games
        profile.plays += int(
            (
                main
                & (row >= PLAY_OFFSET)
                & (row < PLAY_OFFSET + NUM_CARDS)
            ).sum()
        )
        profile.draw_plays += int((main & (row == PLAY_OFFSET + 2)).sum())
        profile.buys += int(
            (main & (row >= BUY_OFFSET) & (row < BUY_OFFSET + NUM_CARDS)).sum()
        )
        profile.attacks += int((main & (row == ATTACK_PLAYER_INDEX)).sum())
        profile.end_turns += int((main & (row == 0)).sum())
        profile.resets += step_numpy(reference, row)

    validate_workload_profile(profile)
    return actions, profile


def make_action_schedule(st: VecState, steps: int) -> np.ndarray:
    """Compatibility helper returning only the precomputed actions."""
    return build_workload(st, steps)[0]


def validate_workload_profile(profile: WorkloadProfile) -> None:
    """Reject a gate whose advertised expensive paths have disappeared."""
    # Tiny smoke runs cannot contain every branch at stable proportions; the
    # gate-sized workload can and must.
    if profile.decisions < 10_000:
        return
    required_rates = {
        "pending modes": (profile.pending_modes, 0.02),
        "plays": (profile.plays, 0.08),
        "draw-card plays": (profile.draw_plays, 0.01),
        "buys": (profile.buys, 0.04),
        "attacks": (profile.attacks, 0.04),
        "end turns": (profile.end_turns, 0.08),
    }
    failures = [
        f"{name}={profile.rate(count):.2%} < {minimum:.2%}"
        for name, (count, minimum) in required_rates.items()
        if profile.rate(count) < minimum
    ]
    if profile.resets == 0:
        failures.append("terminal reset was never exercised")
    if failures:
        raise RuntimeError(
            "primitive workload degenerated: " + "; ".join(failures)
        )


# ---------------------------------------------------------------------------
# NumPy SIMD backend
# ---------------------------------------------------------------------------
def build_mask_numpy(st: VecState, mask: np.ndarray) -> None:
    mask.fill(False)
    rows = np.arange(st.games)
    cur = st.cur_player.astype(np.intp)
    live = ~st.done
    main = live & (st.mode == 0)
    choice = live & (st.mode != 0)

    mask[main, 0] = True
    mask[choice, 1] = True

    hand = st.zones[rows, cur, ZONE_HAND]
    mask[:, PLAY_OFFSET : PLAY_OFFSET + NUM_CARDS] = (
        (hand > 0) & main[:, None]
    )

    in_row = np.zeros((st.games, NUM_CARDS), dtype=np.bool_)
    np.put_along_axis(
        in_row, st.trade_row.astype(np.intp), np.ones_like(st.trade_row, bool), axis=1
    )
    trade = st.scalars[rows, cur, 0][:, None]
    mask[:, BUY_OFFSET : BUY_OFFSET + NUM_CARDS] = (
        (st.cost[None, :] <= trade)
        & in_row
        & (st.trade_deck > 0)
        & main[:, None]
    )
    mask[:, ATTACK_PLAYER_INDEX] = (
        (st.scalars[rows, cur, 1] > 0) & main
    )


def _draw_numpy(st: VecState, idx: np.ndarray, draws: int) -> None:
    if idx.size == 0:
        return

    cur = st.cur_player[idx].astype(np.intp)
    for _ in range(draws):
        deck = st.zones[idx, cur, ZONE_DECK].astype(np.int64)
        total = deck.sum(axis=1)

        empty = total == 0
        if empty.any():
            e = idx[empty]
            ep = cur[empty]
            st.zones[e, ep, ZONE_DECK] += st.zones[e, ep, ZONE_DISCARD]
            st.zones[e, ep, ZONE_DISCARD] = 0
            deck = st.zones[idx, cur, ZONE_DECK].astype(np.int64)
            total = deck.sum(axis=1)

        live = total > 0
        li = idx[live]
        if li.size == 0:
            continue
        lp = cur[live]

        next_rng = (
            st.rng_state[li] * LCG_MULT + LCG_ADD
        ) & LCG_MASK
        st.rng_state[li] = next_rng
        ticket = next_rng % total[live]

        live_deck = st.zones[li, lp, ZONE_DECK].astype(np.int64)
        cumulative = np.cumsum(live_deck, axis=1)
        card = (cumulative <= ticket[:, None]).sum(axis=1).astype(np.intp)

        st.zones[li, lp, ZONE_DECK, card] -= 1
        st.zones[li, lp, ZONE_HAND, card] += 1


def _refill_market_numpy(
    st: VecState, games: np.ndarray, bought: np.ndarray
) -> None:
    if games.size == 0:
        return
    row_match = st.trade_row[games] == bought[:, None]
    slot = row_match.argmax(axis=1)
    offsets = np.arange(1, NUM_CARDS + 1, dtype=np.intp)
    candidates = (bought[:, None] + offsets[None, :]) % NUM_CARDS
    available = st.trade_deck[games[:, None], candidates] > 0
    first = available.argmax(axis=1)
    replacement = candidates[np.arange(games.size), first]
    st.trade_row[games, slot] = replacement.astype(np.int8)


def _reset_numpy_games(st: VecState, games: np.ndarray) -> None:
    if games.size == 0:
        return
    st.done[games] = False
    st.mode[games] = 0
    st.cur_player[games] = 0
    st.scalars[games] = 0
    st.scalars[games, :, 0] = 3
    st.scalars[games, :, 1] = 2
    st.scalars[games, :, 2] = 30
    st.zones[games] = 0
    st.zones[games, :, ZONE_HAND, 0] = 2
    st.zones[games, :, ZONE_HAND, 1] = 1
    st.zones[games, :, ZONE_HAND, 2] = 1
    st.zones[games, :, ZONE_HAND, 5] = 1
    st.zones[games, :, ZONE_DECK, 0] = 5
    st.zones[games, :, ZONE_DECK, 1] = 2
    st.zones[games, :, ZONE_DISCARD, 0] = 3
    st.zones[games, :, ZONE_BASES, 5] = 1
    st.trade_deck[games] = 3
    st.trade_row[games] = st.initial_trade_row[games]


def step_numpy(st: VecState, actions: np.ndarray) -> int:
    rows = np.arange(st.games)
    cur = st.cur_player.astype(np.intp)
    live = ~st.done
    main = live & (st.mode == 0)

    # Pending modes override normal actions. This models suspension/resumption
    # without pretending to implement the actual choice semantics.
    choice = live & (st.mode != 0)
    i = rows[choice]
    p = cur[choice]
    m = st.mode[choice].astype(np.intp)
    card = np.where((m & 1) == 0, 2, 0).astype(np.intp)
    has = st.zones[i, p, ZONE_HAND, card] > 0
    ci, cp, cc = i[has], p[has], card[has]
    st.zones[ci, cp, ZONE_HAND, cc] -= 1
    st.zones[ci, cp, ZONE_DISCARD, cc] += 1
    st.mode[i] = 0

    # PLAY
    selected = main & (actions >= PLAY_OFFSET) & (
        actions < PLAY_OFFSET + NUM_CARDS
    )
    i = rows[selected]
    p = cur[selected]
    card = (actions[selected] - PLAY_OFFSET).astype(np.intp)
    legal = st.zones[i, p, ZONE_HAND, card] > 0
    i, p, card = i[legal], p[legal], card[legal]
    st.zones[i, p, ZONE_HAND, card] -= 1
    st.zones[i, p, ZONE_PLAYED, card] += 1
    st.scalars[i, p, 0] += st.eff_trade[card]
    st.scalars[i, p, 1] += st.eff_combat[card]
    choice_kind = st.choice_kind[card]
    has_choice = choice_kind > 0
    st.mode[i[has_choice]] = choice_kind[has_choice]
    _draw_numpy(st, i[st.eff_draw[card] > 0], 1)

    # BUY
    selected = main & (actions >= BUY_OFFSET) & (
        actions < BUY_OFFSET + NUM_CARDS
    )
    i = rows[selected]
    p = cur[selected]
    card = (actions[selected] - BUY_OFFSET).astype(np.intp)
    in_row = (st.trade_row[i] == card[:, None]).any(axis=1)
    legal = (
        in_row
        & (st.trade_deck[i, card] > 0)
        & (st.scalars[i, p, 0] >= st.cost[card])
    )
    i, p, card = i[legal], p[legal], card[legal]
    st.scalars[i, p, 0] -= st.cost[card]
    st.zones[i, p, ZONE_DISCARD, card] += 1
    st.trade_deck[i, card] -= 1
    _refill_market_numpy(st, i, card)

    # ATTACK PLAYER
    selected = main & (actions == ATTACK_PLAYER_INDEX)
    i = rows[selected]
    p = cur[selected]
    legal = st.scalars[i, p, 1] > 0
    i, p = i[legal], p[legal]
    opp = 1 - p
    st.scalars[i, opp, 2] -= st.scalars[i, p, 1]
    st.scalars[i, p, 1] = 0

    # END TURN
    selected = main & (actions == 0)
    i = rows[selected]
    p = cur[selected]
    st.zones[i, p, ZONE_DISCARD] += st.zones[i, p, ZONE_PLAYED]
    st.zones[i, p, ZONE_PLAYED] = 0
    st.zones[i, p, ZONE_DISCARD] += st.zones[i, p, ZONE_HAND]
    st.zones[i, p, ZONE_HAND] = 0
    st.scalars[i, p, 0] = 0
    st.scalars[i, p, 1] = 0
    _draw_numpy(st, i, HAND_SIZE)
    st.cur_player[i] = 1 - p

    # Terminal check + immediate reset, as a rollout engine would refill slots.
    dead = (st.scalars[:, :, 2] <= 0).any(axis=1)
    i = rows[dead]
    _reset_numpy_games(st, i)
    return int(i.size)


# ---------------------------------------------------------------------------
# Numba SPMD backend
# ---------------------------------------------------------------------------
_NUMBA_KERNELS: tuple[Callable[..., Any], Callable[..., Any]] | None = None


def _get_numba_kernels() -> tuple[Callable[..., Any], Callable[..., Any]]:
    global _NUMBA_KERNELS
    if _NUMBA_KERNELS is not None:
        return _NUMBA_KERNELS

    try:
        from numba import njit, prange
    except ImportError as exc:
        raise RuntimeError(
            "numba backend unavailable; install the dev dependencies"
        ) from exc

    @njit(inline="always")
    def draw_one(zones, cur_player, rng_state, gi, player):
        total = 0
        for card in range(NUM_CARDS):
            total += zones[gi, player, ZONE_DECK, card]
        if total == 0:
            for card in range(NUM_CARDS):
                zones[gi, player, ZONE_DECK, card] += zones[
                    gi, player, ZONE_DISCARD, card
                ]
                zones[gi, player, ZONE_DISCARD, card] = 0
                total += zones[gi, player, ZONE_DECK, card]
        if total == 0:
            return

        next_rng = (rng_state[gi] * LCG_MULT + LCG_ADD) & LCG_MASK
        rng_state[gi] = next_rng
        ticket = next_rng % total
        cumulative = 0
        picked = 0
        for card in range(NUM_CARDS):
            cumulative += zones[gi, player, ZONE_DECK, card]
            if cumulative > ticket:
                picked = card
                break
        zones[gi, player, ZONE_DECK, picked] -= 1
        zones[gi, player, ZONE_HAND, picked] += 1

    @njit(cache=True, parallel=True, fastmath=False)
    def mask_kernel(
        zones,
        scalars,
        trade_row,
        trade_deck,
        mode,
        cur_player,
        done,
        cost,
        mask,
    ):
        for gi in prange(zones.shape[0]):
            for action in range(mask.shape[1]):
                mask[gi, action] = False
            if done[gi]:
                continue

            player = cur_player[gi]
            if mode[gi] != 0:
                mask[gi, 1] = True
                continue

            mask[gi, 0] = True
            for card in range(NUM_CARDS):
                if zones[gi, player, ZONE_HAND, card] > 0:
                    mask[gi, PLAY_OFFSET + card] = True
            for slot in range(trade_row.shape[1]):
                card = trade_row[gi, slot]
                if (
                    cost[card] <= scalars[gi, player, 0]
                    and trade_deck[gi, card] > 0
                ):
                    mask[gi, BUY_OFFSET + card] = True
            if scalars[gi, player, 1] > 0:
                mask[gi, ATTACK_PLAYER_INDEX] = True

    @njit(cache=True, parallel=True, fastmath=False)
    def step_kernel(
        zones,
        scalars,
        trade_row,
        trade_deck,
        mode,
        cur_player,
        done,
        rng_state,
        cost,
        eff_trade,
        eff_combat,
        eff_draw,
        choice_kind,
        initial_trade_row,
        actions,
    ):
        for gi in prange(zones.shape[0]):
            if done[gi]:
                continue

            player = cur_player[gi]
            opponent = 1 - player
            pending_mode = mode[gi]

            if pending_mode != 0:
                card = 2 if (pending_mode & 1) == 0 else 0
                if zones[gi, player, ZONE_HAND, card] > 0:
                    zones[gi, player, ZONE_HAND, card] -= 1
                    zones[gi, player, ZONE_DISCARD, card] += 1
                mode[gi] = 0
            else:
                action = actions[gi]
                if PLAY_OFFSET <= action < PLAY_OFFSET + NUM_CARDS:
                    card = action - PLAY_OFFSET
                    if zones[gi, player, ZONE_HAND, card] > 0:
                        zones[gi, player, ZONE_HAND, card] -= 1
                        zones[gi, player, ZONE_PLAYED, card] += 1
                        scalars[gi, player, 0] += eff_trade[card]
                        scalars[gi, player, 1] += eff_combat[card]
                        if choice_kind[card] > 0:
                            mode[gi] = choice_kind[card]
                        if eff_draw[card] > 0:
                            draw_one(zones, cur_player, rng_state, gi, player)

                elif BUY_OFFSET <= action < BUY_OFFSET + NUM_CARDS:
                    card = action - BUY_OFFSET
                    in_row = False
                    for slot in range(trade_row.shape[1]):
                        if trade_row[gi, slot] == card:
                            in_row = True
                            break
                    if (
                        in_row
                        and trade_deck[gi, card] > 0
                        and scalars[gi, player, 0] >= cost[card]
                    ):
                        scalars[gi, player, 0] -= cost[card]
                        zones[gi, player, ZONE_DISCARD, card] += 1
                        trade_deck[gi, card] -= 1
                        slot = 0
                        for market_slot in range(trade_row.shape[1]):
                            if trade_row[gi, market_slot] == card:
                                slot = market_slot
                                break
                        replacement = card
                        for offset in range(1, NUM_CARDS + 1):
                            candidate = (card + offset) % NUM_CARDS
                            if trade_deck[gi, candidate] > 0:
                                replacement = candidate
                                break
                        trade_row[gi, slot] = replacement

                elif action == ATTACK_PLAYER_INDEX:
                    if scalars[gi, player, 1] > 0:
                        scalars[gi, opponent, 2] -= scalars[gi, player, 1]
                        scalars[gi, player, 1] = 0

                elif action == 0:
                    for card in range(NUM_CARDS):
                        zones[gi, player, ZONE_DISCARD, card] += zones[
                            gi, player, ZONE_PLAYED, card
                        ]
                        zones[gi, player, ZONE_PLAYED, card] = 0
                        zones[gi, player, ZONE_DISCARD, card] += zones[
                            gi, player, ZONE_HAND, card
                        ]
                        zones[gi, player, ZONE_HAND, card] = 0
                    scalars[gi, player, 0] = 0
                    scalars[gi, player, 1] = 0
                    for _ in range(HAND_SIZE):
                        draw_one(zones, cur_player, rng_state, gi, player)
                    cur_player[gi] = opponent

            if scalars[gi, 0, 2] <= 0 or scalars[gi, 1, 2] <= 0:
                done[gi] = False
                mode[gi] = 0
                cur_player[gi] = 0
                for pp in range(NUM_PLAYERS):
                    for scalar in range(NUM_SCALARS):
                        scalars[gi, pp, scalar] = 0
                    scalars[gi, pp, 0] = 3
                    scalars[gi, pp, 1] = 2
                    scalars[gi, pp, 2] = 30
                    for zone in range(NUM_ZONES):
                        for card in range(NUM_CARDS):
                            zones[gi, pp, zone, card] = 0
                    zones[gi, pp, ZONE_HAND, 0] = 2
                    zones[gi, pp, ZONE_HAND, 1] = 1
                    zones[gi, pp, ZONE_HAND, 2] = 1
                    zones[gi, pp, ZONE_HAND, 5] = 1
                    zones[gi, pp, ZONE_DECK, 0] = 5
                    zones[gi, pp, ZONE_DECK, 1] = 2
                    zones[gi, pp, ZONE_DISCARD, 0] = 3
                    zones[gi, pp, ZONE_BASES, 5] = 1
                for card in range(NUM_CARDS):
                    trade_deck[gi, card] = 3
                for slot in range(trade_row.shape[1]):
                    trade_row[gi, slot] = initial_trade_row[gi, slot]

    _NUMBA_KERNELS = step_kernel, mask_kernel
    return _NUMBA_KERNELS


# ---------------------------------------------------------------------------
# Torch masked backend (CPU or CUDA/ROCm)
# ---------------------------------------------------------------------------
@dataclass
class TorchState:
    games: int
    zones: Any
    scalars: Any
    trade_row: Any
    trade_deck: Any
    mode: Any
    cur_player: Any
    done: Any
    rng_state: Any
    cost: Any
    eff_trade: Any
    eff_combat: Any
    eff_draw: Any
    choice_kind: Any
    initial_trade_row: Any


def _to_torch_state(st: VecState, torch: Any, device: Any) -> TorchState:
    values: dict[str, Any] = {"games": st.games}
    for f in fields(st):
        if f.name == "games":
            continue
        # ``Tensor.to(cpu)`` otherwise returns the original from_numpy view,
        # and the torch-cpu backend would mutate the shared VecState used to
        # initialize later backends. Every backend must start from an
        # independent copy or the equivalence gate becomes order-dependent.
        values[f.name] = torch.from_numpy(getattr(st, f.name)).to(
            device=device, copy=True
        )
    return TorchState(**values)


def build_mask_torch(st: TorchState, mask: Any, torch: Any) -> None:
    mask.zero_()
    rows = torch.arange(st.games, device=st.zones.device)
    cur = st.cur_player.long()
    live = ~st.done
    main = live & (st.mode == 0)
    choice = live & (st.mode != 0)

    mask[main, 0] = True
    mask[choice, 1] = True
    hand = st.zones[rows, cur, ZONE_HAND]
    mask[:, PLAY_OFFSET : PLAY_OFFSET + NUM_CARDS] = (
        (hand > 0) & main[:, None]
    )

    in_row = torch.zeros(
        (st.games, NUM_CARDS), dtype=torch.bool, device=st.zones.device
    )
    in_row.scatter_(1, st.trade_row.long(), True)
    trade = st.scalars[rows, cur, 0][:, None]
    mask[:, BUY_OFFSET : BUY_OFFSET + NUM_CARDS] = (
        (st.cost[None, :] <= trade)
        & in_row
        & (st.trade_deck > 0)
        & main[:, None]
    )
    mask[:, ATTACK_PLAYER_INDEX] = (
        (st.scalars[rows, cur, 1] > 0) & main
    )


def _draw_torch(st: TorchState, idx: Any, draws: int, torch: Any) -> None:
    cur = st.cur_player[idx].long()
    for _ in range(draws):
        deck = st.zones[idx, cur, ZONE_DECK].to(torch.int64)
        total = deck.sum(dim=1)

        empty = total == 0
        e = idx[empty]
        ep = cur[empty]
        st.zones[e, ep, ZONE_DECK] += st.zones[e, ep, ZONE_DISCARD]
        st.zones[e, ep, ZONE_DISCARD] = 0

        deck = st.zones[idx, cur, ZONE_DECK].to(torch.int64)
        total = deck.sum(dim=1)
        live = total > 0
        li = idx[live]
        lp = cur[live]
        live_total = total[live]

        next_rng = (
            st.rng_state[li] * LCG_MULT + LCG_ADD
        ).bitwise_and(LCG_MASK)
        st.rng_state[li] = next_rng
        ticket = torch.remainder(next_rng, live_total)

        live_deck = st.zones[li, lp, ZONE_DECK].to(torch.int64)
        cumulative = torch.cumsum(live_deck, dim=1)
        card = (cumulative <= ticket[:, None]).sum(dim=1).long()
        st.zones[li, lp, ZONE_DECK, card] -= 1
        st.zones[li, lp, ZONE_HAND, card] += 1


def _refill_market_torch(
    st: TorchState, games: Any, bought: Any, torch: Any
) -> None:
    row_match = st.trade_row[games].long() == bought[:, None]
    slot = row_match.to(torch.int8).argmax(dim=1)
    offsets = torch.arange(
        1, NUM_CARDS + 1, dtype=torch.int64, device=st.zones.device
    )
    candidates = torch.remainder(
        bought[:, None] + offsets[None, :], NUM_CARDS
    )
    available = st.trade_deck[games[:, None], candidates] > 0
    first = available.to(torch.int8).argmax(dim=1)
    replacement = candidates[
        torch.arange(games.numel(), device=st.zones.device), first
    ]
    st.trade_row[games, slot] = replacement.to(st.trade_row.dtype)


def _reset_torch_games(st: TorchState, games: Any) -> None:
    st.done[games] = False
    st.mode[games] = 0
    st.cur_player[games] = 0
    st.scalars[games] = 0
    st.scalars[games, :, 0] = 3
    st.scalars[games, :, 1] = 2
    st.scalars[games, :, 2] = 30
    st.zones[games] = 0
    st.zones[games, :, ZONE_HAND, 0] = 2
    st.zones[games, :, ZONE_HAND, 1] = 1
    st.zones[games, :, ZONE_HAND, 2] = 1
    st.zones[games, :, ZONE_HAND, 5] = 1
    st.zones[games, :, ZONE_DECK, 0] = 5
    st.zones[games, :, ZONE_DECK, 1] = 2
    st.zones[games, :, ZONE_DISCARD, 0] = 3
    st.zones[games, :, ZONE_BASES, 5] = 1
    st.trade_deck[games] = 3
    st.trade_row[games] = st.initial_trade_row[games]


def step_torch(st: TorchState, actions: Any, torch: Any) -> None:
    rows = torch.arange(st.games, device=st.zones.device)
    cur = st.cur_player.long()
    live = ~st.done
    main = live & (st.mode == 0)

    choice = live & (st.mode != 0)
    i = rows[choice]
    p = cur[choice]
    m = st.mode[choice].long()
    card = torch.where((m & 1) == 0, 2, 0).long()
    has = st.zones[i, p, ZONE_HAND, card] > 0
    ci, cp, cc = i[has], p[has], card[has]
    st.zones[ci, cp, ZONE_HAND, cc] -= 1
    st.zones[ci, cp, ZONE_DISCARD, cc] += 1
    st.mode[i] = 0

    selected = main & (actions >= PLAY_OFFSET) & (
        actions < PLAY_OFFSET + NUM_CARDS
    )
    i = rows[selected]
    p = cur[selected]
    card = (actions[selected] - PLAY_OFFSET).long()
    legal = st.zones[i, p, ZONE_HAND, card] > 0
    i, p, card = i[legal], p[legal], card[legal]
    st.zones[i, p, ZONE_HAND, card] -= 1
    st.zones[i, p, ZONE_PLAYED, card] += 1
    st.scalars[i, p, 0] += st.eff_trade[card]
    st.scalars[i, p, 1] += st.eff_combat[card]
    choice_kind = st.choice_kind[card]
    has_choice = choice_kind > 0
    st.mode[i[has_choice]] = choice_kind[has_choice]
    _draw_torch(st, i[st.eff_draw[card] > 0], 1, torch)

    selected = main & (actions >= BUY_OFFSET) & (
        actions < BUY_OFFSET + NUM_CARDS
    )
    i = rows[selected]
    p = cur[selected]
    card = (actions[selected] - BUY_OFFSET).long()
    in_row = (st.trade_row[i].long() == card[:, None]).any(dim=1)
    legal = (
        in_row
        & (st.trade_deck[i, card] > 0)
        & (st.scalars[i, p, 0] >= st.cost[card])
    )
    i, p, card = i[legal], p[legal], card[legal]
    st.scalars[i, p, 0] -= st.cost[card]
    st.zones[i, p, ZONE_DISCARD, card] += 1
    st.trade_deck[i, card] -= 1
    _refill_market_torch(st, i, card, torch)

    selected = main & (actions == ATTACK_PLAYER_INDEX)
    i = rows[selected]
    p = cur[selected]
    legal = st.scalars[i, p, 1] > 0
    i, p = i[legal], p[legal]
    opp = 1 - p
    st.scalars[i, opp, 2] -= st.scalars[i, p, 1]
    st.scalars[i, p, 1] = 0

    selected = main & (actions == 0)
    i = rows[selected]
    p = cur[selected]
    st.zones[i, p, ZONE_DISCARD] += st.zones[i, p, ZONE_PLAYED]
    st.zones[i, p, ZONE_PLAYED] = 0
    st.zones[i, p, ZONE_DISCARD] += st.zones[i, p, ZONE_HAND]
    st.zones[i, p, ZONE_HAND] = 0
    st.scalars[i, p, 0] = 0
    st.scalars[i, p, 1] = 0
    _draw_torch(st, i, HAND_SIZE, torch)
    st.cur_player[i] = (1 - p).to(st.cur_player.dtype)

    dead = (st.scalars[:, :, 2] <= 0).any(dim=1)
    i = rows[dead]
    _reset_torch_games(st, i)


# ---------------------------------------------------------------------------
# Execution and verification
# ---------------------------------------------------------------------------
def _snapshot_numpy(
    st: VecState,
    mask: np.ndarray,
    mask_trace: np.ndarray | None = None,
) -> Snapshot:
    return Snapshot(
        arrays={name: getattr(st, name).copy() for name in MUTABLE_FIELDS},
        mask=mask.copy(),
        mask_trace=mask_trace,
    )


def _snapshot_torch(
    st: TorchState,
    mask: Any,
    mask_trace: np.ndarray | None = None,
) -> Snapshot:
    return Snapshot(
        arrays={
            name: getattr(st, name).detach().cpu().numpy().copy()
            for name in MUTABLE_FIELDS
        },
        mask=mask.detach().cpu().numpy().copy(),
        mask_trace=mask_trace,
    )


def _execute_numpy(
    base: VecState,
    actions: np.ndarray,
    *,
    timed: bool,
    capture_mask_trace: bool,
) -> tuple[float, Snapshot]:
    st = base.copy()
    mask = np.zeros((st.games, ACTION_DIM), dtype=np.bool_)
    trace = (
        np.empty((len(actions), st.games, ACTION_DIM), dtype=np.bool_)
        if capture_mask_trace
        else None
    )
    started = time.perf_counter() if timed else 0.0
    for step, action in enumerate(actions):
        build_mask_numpy(st, mask)
        if trace is not None:
            trace[step] = mask
        step_numpy(st, action)
    elapsed = time.perf_counter() - started if timed else 0.0
    # Snapshot mask must describe the final state, not the state before the
    # last transition. This rebuild is outside the timed region.
    build_mask_numpy(st, mask)
    return elapsed, _snapshot_numpy(st, mask, trace)


def _execute_numba(
    base: VecState,
    actions: np.ndarray,
    *,
    timed: bool,
    capture_mask_trace: bool,
) -> tuple[float, Snapshot]:
    step_kernel, mask_kernel = _get_numba_kernels()
    st = base.copy()
    mask = np.zeros((st.games, ACTION_DIM), dtype=np.bool_)
    trace = (
        np.empty((len(actions), st.games, ACTION_DIM), dtype=np.bool_)
        if capture_mask_trace
        else None
    )
    started = time.perf_counter() if timed else 0.0
    for step, action in enumerate(actions):
        mask_kernel(
            st.zones,
            st.scalars,
            st.trade_row,
            st.trade_deck,
            st.mode,
            st.cur_player,
            st.done,
            st.cost,
            mask,
        )
        if trace is not None:
            trace[step] = mask
        step_kernel(
            st.zones,
            st.scalars,
            st.trade_row,
            st.trade_deck,
            st.mode,
            st.cur_player,
            st.done,
            st.rng_state,
            st.cost,
            st.eff_trade,
            st.eff_combat,
            st.eff_draw,
            st.choice_kind,
            st.initial_trade_row,
            action,
        )
    elapsed = time.perf_counter() - started if timed else 0.0
    mask_kernel(
        st.zones,
        st.scalars,
        st.trade_row,
        st.trade_deck,
        st.mode,
        st.cur_player,
        st.done,
        st.cost,
        mask,
    )
    return elapsed, _snapshot_numpy(st, mask, trace)


def _load_torch(device_name: str) -> tuple[Any, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch backend unavailable") from exc

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("torch-cuda backend unavailable: no CUDA/ROCm device")
    return torch, torch.device(device_name)


def _execute_torch(
    base: VecState,
    actions: np.ndarray,
    *,
    device_name: str,
    timed: bool,
    capture_mask_trace: bool,
) -> tuple[float, Snapshot]:
    torch, device = _load_torch(device_name)
    st = _to_torch_state(base, torch, device)
    action_tensor = torch.from_numpy(actions).to(device)
    mask = torch.zeros(
        (st.games, ACTION_DIM), dtype=torch.bool, device=device
    )
    trace = (
        np.empty((len(actions), st.games, ACTION_DIM), dtype=np.bool_)
        if capture_mask_trace
        else None
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter() if timed else 0.0
    for step, action in enumerate(action_tensor):
        build_mask_torch(st, mask, torch)
        if trace is not None:
            trace[step] = mask.detach().cpu().numpy()
        step_torch(st, action, torch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started if timed else 0.0
    build_mask_torch(st, mask, torch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return elapsed, _snapshot_torch(st, mask, trace)


def execute_backend(
    name: str,
    base: VecState,
    actions: np.ndarray,
    *,
    timed: bool,
    capture_mask_trace: bool = False,
) -> tuple[float, Snapshot]:
    if name == "numpy":
        return _execute_numpy(
            base,
            actions,
            timed=timed,
            capture_mask_trace=capture_mask_trace,
        )
    if name == "numba":
        return _execute_numba(
            base,
            actions,
            timed=timed,
            capture_mask_trace=capture_mask_trace,
        )
    if name == "torch-cpu":
        return _execute_torch(
            base,
            actions,
            device_name="cpu",
            timed=timed,
            capture_mask_trace=capture_mask_trace,
        )
    if name == "torch-cuda":
        return _execute_torch(
            base,
            actions,
            device_name="cuda",
            timed=timed,
            capture_mask_trace=capture_mask_trace,
        )
    raise ValueError(
        f"unknown backend {name!r}; expected one of {', '.join(KNOWN_BACKENDS)}"
    )


def backend_availability(name: str) -> tuple[bool, str | None]:
    if name not in KNOWN_BACKENDS:
        return False, f"unknown backend {name!r}"
    if name == "numpy":
        return True, None
    if name == "numba":
        try:
            import numba  # noqa: F401
        except ImportError:
            return False, "numba is not installed"
        return True, None
    if name.startswith("torch"):
        try:
            import torch
        except ImportError:
            return False, "torch is not installed"
        if name == "torch-cuda" and not torch.cuda.is_available():
            return False, "torch reports no CUDA/ROCm device"
        return True, None
    raise AssertionError(name)


def _assert_array_equal(
    backend: str, field_name: str, expected: np.ndarray, actual: np.ndarray
) -> None:
    if expected.shape != actual.shape:
        raise AssertionError(
            f"{backend}.{field_name}: shape {actual.shape} != {expected.shape}"
        )
    if expected.dtype != actual.dtype:
        raise AssertionError(
            f"{backend}.{field_name}: dtype {actual.dtype} != {expected.dtype}"
        )
    if np.array_equal(expected, actual):
        return

    differing = np.argwhere(expected != actual)
    first = tuple(int(index) for index in differing[0])
    raise AssertionError(
        f"{backend}.{field_name} diverged at {first}: "
        f"expected {expected[first]!r}, got {actual[first]!r}; "
        f"{len(differing)} elements differ"
    )


def assert_snapshot_invariants(snapshot: Snapshot) -> None:
    zones = snapshot.arrays["zones"]
    trade_deck = snapshot.arrays["trade_deck"]
    if (zones < 0).any():
        first = tuple(np.argwhere(zones < 0)[0])
        raise AssertionError(f"negative zone count at {first}: {zones[first]}")
    if (trade_deck < 0).any():
        first = tuple(np.argwhere(trade_deck < 0)[0])
        raise AssertionError(
            f"negative trade-deck count at {first}: {trade_deck[first]}"
        )
    if snapshot.mask.shape != (zones.shape[0], ACTION_DIM):
        raise AssertionError(f"wrong mask shape: {snapshot.mask.shape}")
    if snapshot.mask.dtype != np.bool_:
        raise AssertionError(f"wrong mask dtype: {snapshot.mask.dtype}")
    if not snapshot.mask.any(axis=1).all():
        game = int(np.nonzero(~snapshot.mask.any(axis=1))[0][0])
        raise AssertionError(f"game {game} has no legal action")


def verify_backends_agree(
    games: int, steps: int, backends: list[str]
) -> str:
    """Require exact backend equivalence before reporting any timing."""
    if not backends:
        raise ValueError("at least one available backend is required")

    base = VecState.create(games, seed=12_345)
    actions = make_action_schedule(base, steps)
    snapshots: dict[str, Snapshot] = {}
    for name in backends:
        _, snapshot = execute_backend(
            name,
            base,
            actions,
            timed=False,
            capture_mask_trace=True,
        )
        assert_snapshot_invariants(snapshot)
        snapshots[name] = snapshot

    reference_name = backends[0]
    reference = snapshots[reference_name]
    for name in backends[1:]:
        candidate = snapshots[name]
        for field_name in MUTABLE_FIELDS:
            _assert_array_equal(
                name,
                field_name,
                reference.arrays[field_name],
                candidate.arrays[field_name],
            )
        _assert_array_equal(name, "mask", reference.mask, candidate.mask)
        if reference.mask_trace is None or candidate.mask_trace is None:
            raise AssertionError("verification did not capture per-step masks")
        _assert_array_equal(
            name,
            "mask_trace",
            reference.mask_trace,
            candidate.mask_trace,
        )

    return reference.digest()


def warm_backend(name: str) -> None:
    """Exclude JIT compilation and first-device-use costs from timing."""
    base = VecState.create(64, seed=999)
    actions = make_action_schedule(base, 3)
    execute_backend(name, base, actions, timed=False)


def bench_backend(
    name: str, base: VecState, actions: np.ndarray
) -> Result:
    elapsed, snapshot = execute_backend(name, base, actions, timed=True)
    if elapsed <= 0:
        raise RuntimeError(f"{name}: non-positive elapsed time {elapsed}")
    decisions = base.games * len(actions)
    return Result(
        backend=name,
        decisions_per_sec=decisions / elapsed,
        step_ms=elapsed / len(actions) * 1_000,
        digest=snapshot.digest(),
        elapsed_s=elapsed,
    )


def _rotated_order(backends: list[str], repetition: int) -> list[str]:
    """A Latin-square-like rotation so no backend always runs first/last."""
    start = repetition % len(backends)
    order = backends[start:] + backends[:start]
    if (repetition // len(backends)) & 1:
        order.reverse()
    return order


def run_benchmarks(
    games: int,
    steps: int,
    repetitions: int,
    backends: list[str],
) -> dict[str, AggregateResult]:
    if repetitions <= 0:
        raise ValueError(f"repetitions must be positive, got {repetitions}")

    for name in backends:
        warm_backend(name)

    samples: dict[str, list[Result]] = {name: [] for name in backends}
    for repetition in range(repetitions):
        base = VecState.create(games, seed=50_000 + repetition)
        actions, profile = build_workload(base, steps)
        validate_workload_profile(profile)
        by_digest: dict[str, list[str]] = {}
        for name in _rotated_order(backends, repetition):
            result = bench_backend(name, base, actions)
            samples[name].append(result)
            by_digest.setdefault(result.digest, []).append(name)

        if len(by_digest) != 1:
            detail = "; ".join(
                f"{digest[:12]}: {','.join(names)}"
                for digest, names in by_digest.items()
            )
            raise AssertionError(
                f"backends diverged during timed repetition {repetition}: {detail}"
            )

    aggregates: dict[str, AggregateResult] = {}
    for name, values in samples.items():
        rates = tuple(v.decisions_per_sec for v in values)
        aggregates[name] = AggregateResult(
            backend=name,
            decisions_per_sec=statistics.median(rates),
            step_ms=statistics.median(v.step_ms for v in values),
            digest=values[-1].digest,
            repetitions=rates,
        )
    return aggregates


# ---------------------------------------------------------------------------
# CLI and reporting
# ---------------------------------------------------------------------------
def _resolve_backends(
    requested: list[str], *, require_all: bool
) -> tuple[list[str], dict[str, str]]:
    available: list[str] = []
    skipped: dict[str, str] = {}
    for name in requested:
        ok, reason = backend_availability(name)
        if ok:
            available.append(name)
        else:
            skipped[name] = reason or "unavailable"

    if require_all and skipped:
        detail = ", ".join(f"{name}: {reason}" for name, reason in skipped.items())
        raise RuntimeError(f"required backends unavailable: {detail}")
    if not available:
        raise RuntimeError("none of the requested backends is available")
    return available, skipped


def report(
    results: dict[str, AggregateResult],
    skipped: dict[str, str],
) -> None:
    print()
    print(
        f"{'backend':<14} {'decisions/sec':>15} {'step ms':>10} "
        f"{'ceiling/current':>16} {'spread':>14} {'gate':>10}"
    )
    print("-" * 86)
    for result in sorted(
        results.values(), key=lambda item: -item.decisions_per_sec
    ):
        ceiling = result.decisions_per_sec / BASELINE_DECISIONS_PER_SEC
        _, gate_eligible, gate = _gate_metadata(
            result.backend, result.decisions_per_sec
        )
        lo = min(result.repetitions)
        hi = max(result.repetitions)
        print(
            f"{result.backend:<14} {result.decisions_per_sec:>15,.0f} "
            f"{result.step_ms:>10.3f} {ceiling:>15.1f}x "
            f"{lo / 1e6:>5.2f}-{hi / 1e6:<5.2f}M {gate:>10}"
        )
    print("-" * 86)
    print(f"{'current rollout':<14} {BASELINE_DECISIONS_PER_SEC:>15,.0f}")
    if skipped:
        print()
        for name, reason in skipped.items():
            print(f"SKIPPED {name}: {reason}")
    print()
    print(
        "ceiling/current is an engine-only upper bound, not predicted training "
        "speedup:"
    )
    print(
        "the primitive benchmark excludes policy inference, while the current "
        "rollout includes it."
    )
    print(
        f"Gate: >= {GATE_SPEEDUP:.0f}x primitive headroom justifies a "
        "rules-correct vertical slice."
    )
    print(
        "Torch backends are eager lower bounds. They cannot reject a future "
        "fused GPU code generator."
    )


def _gate_metadata(
    backend: str, decisions_per_sec: float
) -> tuple[str, bool, str]:
    if backend.startswith("torch-"):
        return "eager", False, "EAGER ONLY"
    ceiling = decisions_per_sec / BASELINE_DECISIONS_PER_SEC
    verdict = (
        "PROCEED"
        if ceiling >= GATE_SPEEDUP
        else "MARGINAL"
        if ceiling >= 3
        else "STOP"
    )
    execution_model = "compiled-spmd" if backend == "numba" else "array-simd"
    return execution_model, True, verdict


def report_workload_profile(profile: WorkloadProfile) -> None:
    print()
    print("untimed workload branch coverage:")
    for label, count in (
        ("pending modes", profile.pending_modes),
        ("  mode 1", profile.mode_1),
        ("  mode 2", profile.mode_2),
        ("  mode 3", profile.mode_3),
        ("  mode 4", profile.mode_4),
        ("plays", profile.plays),
        ("draw-card plays", profile.draw_plays),
        ("buys", profile.buys),
        ("attacks", profile.attacks),
        ("end turns", profile.end_turns),
    ):
        print(f"  {label:<18} {count:>12,}  {profile.rate(count):>7.2%}")
    print(f"  {'terminal resets':<18} {profile.resets:>12,}")


def _result_payload(
    args: argparse.Namespace,
    results: dict[str, AggregateResult],
    skipped: dict[str, str],
    verification_digest: str,
    workload_profile: WorkloadProfile,
) -> dict[str, Any]:
    return {
        "games": args.games,
        "steps": args.steps,
        "repetitions": args.reps,
        "baseline_decisions_per_sec": BASELINE_DECISIONS_PER_SEC,
        "verification_digest": verification_digest,
        "workload_profile": workload_profile.as_dict(),
        "skipped": skipped,
        "results": {
            name: {
                "decisions_per_sec": result.decisions_per_sec,
                "step_ms": result.step_ms,
                "ceiling_vs_current": (
                    result.decisions_per_sec / BASELINE_DECISIONS_PER_SEC
                ),
                "execution_model": _gate_metadata(
                    name, result.decisions_per_sec
                )[0],
                "gate_eligible": _gate_metadata(
                    name, result.decisions_per_sec
                )[1],
                "gate_verdict": _gate_metadata(
                    name, result.decisions_per_sec
                )[2],
                "digest": result.digest,
                "repetitions": list(result.repetitions),
            }
            for name, result in results.items()
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--games", type=int, default=16_000)
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="deterministic candidate decisions per game",
    )
    parser.add_argument(
        "--reps", type=int, default=5, help="interleaved repetitions"
    )
    parser.add_argument(
        "--backends",
        default=",".join(KNOWN_BACKENDS),
        help="comma-separated backend names",
    )
    parser.add_argument(
        "--verify-games",
        type=int,
        default=257,
        help="games in the exact cross-backend verification",
    )
    parser.add_argument(
        "--verify-steps",
        type=int,
        default=127,
        help="steps in the exact cross-backend verification",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="check exact backend equivalence without timing",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip exact equivalence verification (not recommended)",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="fail instead of skipping an unavailable requested backend",
    )
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    requested = [name.strip() for name in args.backends.split(",") if name.strip()]
    available, skipped = _resolve_backends(
        requested, require_all=args.require_all
    )

    print(
        f"G={args.games:,}  steps={args.steps}  reps={args.reps}  "
        f"backends={','.join(available)}"
    )
    print(
        f"zones: {args.games * NUM_PLAYERS * NUM_ZONES * NUM_CARDS / 1e6:.1f} MB"
        f"  mask: {args.games * ACTION_DIM / 1e6:.1f} MB"
    )

    verification_digest = ""
    if not args.no_verify:
        verify_games = min(args.verify_games, args.games)
        verify_steps = min(args.verify_steps, args.steps)
        print(
            f"verifying exact equivalence: {verify_games} games x "
            f"{verify_steps} steps ..."
        )
        verification_digest = verify_backends_agree(
            verify_games, verify_steps, available
        )
        print(f"verification passed (digest={verification_digest[:16]}...)")

    if args.verify_only:
        if skipped:
            for name, reason in skipped.items():
                print(f"SKIPPED {name}: {reason}")
        return 0

    _, workload_profile = build_workload(
        VecState.create(args.games, seed=50_000), args.steps
    )
    report_workload_profile(workload_profile)

    results = run_benchmarks(
        args.games, args.steps, args.reps, available
    )
    report(results, skipped)

    if args.json_out is not None:
        payload = _result_payload(
            args,
            results,
            skipped,
            verification_digest,
            workload_profile,
        )
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
