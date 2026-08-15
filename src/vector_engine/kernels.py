"""Rules-correct Numba SPMD kernels for the Gate 2 supported subset."""
from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np
from numba import njit, prange

from src.encoding.action_encoder import (
    END_TURN_INDEX,
    get_action_space_layout,
    get_action_space_size,
)
from src.vector_engine.rules import (
    CompiledRules,
    EffectOpcode,
    EffectTrigger,
    MAX_BUNDLES,
    PENDING_DRAW_ON_COMPLETE,
    PENDING_MANDATORY,
    TARGET_DISCARD,
    TARGET_HAND,
    TARGET_TRADE,
)
from src.vector_engine.state import (
    INSTANCE_ACTIVE,
    INSTANCE_PLAYED_THIS_TURN,
    NO_CARD,
    NO_WINNER,
    NUM_PLAYERS,
    NUM_ZONES,
    MAX_PENDING_SETS,
    PENDING_DESTROY_BASE,
    PENDING_DISCARD,
    PENDING_NONE,
    PENDING_SCRAP,
    PENDING_TARGET_BASE,
    PENDING_TARGET_DISCARD,
    PENDING_TARGET_HAND,
    PENDING_TARGET_TRADE,
    VectorState,
    ZONE_BASES,
    ZONE_DECK,
    ZONE_DISCARD,
    ZONE_HAND,
    ZONE_PLAYED,
)


NUM_CARDS = 49
ACTION_DIM = get_action_space_size([""] * NUM_CARDS)
_LAYOUT = get_action_space_layout(NUM_CARDS)
_GROUPS = {name: offset for name, offset, _ in _LAYOUT["card_groups"]}

PLAY_OFFSET = _GROUPS["play_card"]
BUY_OFFSET = _GROUPS["buy_card"]
ATTACK_PLAYER_INDEX = _LAYOUT["global_indices"]["attack_player"]
ATTACK_BASE_OFFSET = _GROUPS["attack_base"]
DESTROY_BASE_OFFSET = _GROUPS["destroy_base"]
EFFECT_NONSCRAP_OFFSET = _GROUPS["effect_nonscrap"]
EFFECT_SCRAP_OFFSET = _GROUPS["effect_scrap"]
SCRAP_HAND_OFFSET = _GROUPS["scrap_hand"]
SCRAP_DISCARD_OFFSET = _GROUPS["scrap_discard"]
SCRAP_TRADE_OFFSET = _GROUPS["scrap_trade"]
DISCARD_OFFSET = _GROUPS["discard"]
SKIP_INDEX = _LAYOUT["global_indices"]["skip"]

LCG_MASK = 0x7FFF_FFFF
MAX_FORCED_DRAWS = 64
MAX_FORCED_REFILLS = 8
ACTIVE_BIT = int(INSTANCE_ACTIVE)
CURRENT_TURN_BIT = int(INSTANCE_PLAYED_THIS_TURN)

TRIGGER_AUTO = int(EffectTrigger.AUTO)
TRIGGER_ALLY = int(EffectTrigger.ALLY)
TRIGGER_SCRAP = int(EffectTrigger.SCRAP)
OP_ADD_TRADE = int(EffectOpcode.ADD_TRADE)
OP_ADD_COMBAT = int(EffectOpcode.ADD_COMBAT)
OP_ADD_HEALTH = int(EffectOpcode.ADD_HEALTH)
OP_DRAW = int(EffectOpcode.DRAW)
OP_DRAW_PER_FACTION = int(EffectOpcode.DRAW_PER_FACTION)
OP_DRAW_IF_BASES_GE_2 = int(EffectOpcode.DRAW_IF_BASES_GE_2)
OP_QUEUE_SCRAP = int(EffectOpcode.QUEUE_SCRAP)
OP_ADD_DISCARD_DEBT = int(EffectOpcode.ADD_DISCARD_DEBT)
OP_QUEUE_DESTROY_BASE = int(EffectOpcode.QUEUE_DESTROY_BASE)
OP_QUEUE_SCRAP_DRAW = int(EffectOpcode.QUEUE_SCRAP_DRAW)
OP_QUEUE_DISCARD_DRAW = int(EffectOpcode.QUEUE_DISCARD_DRAW)


@njit(inline="always")
def _advance_rng(value):
    """xorshift32-style state advance with useful low-bit distribution."""
    value ^= (value << 13) & LCG_MASK
    value ^= value >> 17
    value ^= (value << 5) & LCG_MASK
    value &= LCG_MASK
    return value if value != 0 else 1


@njit(inline="always")
def _bundle_bit(bundle):
    return 1 << bundle


@njit(inline="always")
def _add_effect_instance(
    effect_state,
    bundle_count,
    bundle_trigger,
    bundle_stage,
    game,
    player,
    card,
):
    insertion = effect_state.shape[3]
    active_count = 0
    for instance in range(effect_state.shape[3]):
        value = effect_state[game, player, card, instance]
        if (value & ACTIVE_BIT) == 0:
            if insertion == effect_state.shape[3]:
                insertion = instance
            break
        active_count += 1
        # Scalar iterates played_cards before established bases.
        if (
            insertion == effect_state.shape[3]
            and (value & CURRENT_TURN_BIT) == 0
        ):
            insertion = instance
    if active_count >= effect_state.shape[3]:
        return
    if insertion == effect_state.shape[3]:
        insertion = active_count
    for instance in range(active_count, insertion, -1):
        effect_state[game, player, card, instance] = effect_state[
            game, player, card, instance - 1
        ]
    value = ACTIVE_BIT | CURRENT_TURN_BIT
    for bundle in range(bundle_count[card]):
        if bundle_trigger[card, bundle] != TRIGGER_AUTO:
            value |= _bundle_bit(bundle)
    effect_state[game, player, card, insertion] = value


@njit(inline="always")
def _remove_effect_instance(effect_state, game, player, card, instance):
    for slot in range(instance, effect_state.shape[3] - 1):
        effect_state[game, player, card, slot] = effect_state[
            game, player, card, slot + 1
        ]
    effect_state[game, player, card, effect_state.shape[3] - 1] = 0


@njit(inline="always")
def _queue_pending(
    pending_mode,
    pending_remaining,
    pending_sources,
    pending_mandatory,
    pending_draw_on_complete,
    pending_resolved,
    pending_targets,
    pending_count,
    pending_overflow,
    zones,
    trade_row,
    game,
    player,
    mode,
    remaining,
    sources,
    flags,
):
    slot = pending_count[game, player]
    if slot >= pending_mode.shape[2]:
        pending_overflow[game] = True
        return
    if remaining <= 0:
        return
    pending_mode[game, player, slot] = mode
    pending_remaining[game, player, slot] = remaining
    pending_sources[game, player, slot] = sources
    pending_mandatory[game, player, slot] = (
        flags & PENDING_MANDATORY
    ) != 0
    pending_draw_on_complete[game, player, slot] = (
        flags & PENDING_DRAW_ON_COMPLETE
    ) != 0
    pending_resolved[game, player, slot] = 0
    pending_targets[game, player, slot] = 0
    if mode == PENDING_SCRAP:
        if (sources & TARGET_HAND) != 0:
            pending_targets[
                game, player, slot, PENDING_TARGET_HAND
            ] = zones[game, player, ZONE_HAND]
        if (sources & TARGET_DISCARD) != 0:
            pending_targets[
                game, player, slot, PENDING_TARGET_DISCARD
            ] = zones[game, player, ZONE_DISCARD]
        if (sources & TARGET_TRADE) != 0:
            for market_slot in range(trade_row.shape[1]):
                card = trade_row[game, market_slot]
                if card >= 0:
                    pending_targets[
                        game, player, slot, PENDING_TARGET_TRADE, card
                    ] += 1
    elif mode == PENDING_DESTROY_BASE:
        pending_targets[
            game, player, slot, PENDING_TARGET_BASE
        ] = zones[game, 1 - player, ZONE_BASES]
    elif mode == PENDING_DISCARD:
        pending_targets[
            game, player, slot, PENDING_TARGET_HAND
        ] = zones[game, player, ZONE_HAND]
    pending_count[game, player] += 1


@njit(inline="always")
def _pop_pending(
    pending_mode,
    pending_remaining,
    pending_sources,
    pending_mandatory,
    pending_draw_on_complete,
    pending_resolved,
    pending_targets,
    pending_count,
    game,
    player,
):
    count = pending_count[game, player]
    for slot in range(count - 1):
        pending_mode[game, player, slot] = pending_mode[
            game, player, slot + 1
        ]
        pending_remaining[game, player, slot] = pending_remaining[
            game, player, slot + 1
        ]
        pending_sources[game, player, slot] = pending_sources[
            game, player, slot + 1
        ]
        pending_mandatory[game, player, slot] = pending_mandatory[
            game, player, slot + 1
        ]
        pending_draw_on_complete[
            game, player, slot
        ] = pending_draw_on_complete[game, player, slot + 1]
        pending_resolved[game, player, slot] = pending_resolved[
            game, player, slot + 1
        ]
        pending_targets[game, player, slot] = pending_targets[
            game, player, slot + 1
        ]
    last = count - 1
    if last >= 0:
        pending_mode[game, player, last] = PENDING_NONE
        pending_remaining[game, player, last] = 0
        pending_sources[game, player, last] = 0
        pending_mandatory[game, player, last] = False
        pending_draw_on_complete[game, player, last] = False
        pending_resolved[game, player, last] = 0
        pending_targets[game, player, last] = 0
        pending_count[game, player] -= 1


@njit(inline="always")
def _active_count(zones, game, player, card):
    played = zones[game, player, ZONE_PLAYED, card]
    bases = zones[game, player, ZONE_BASES, card]
    return played if played >= bases else bases


@njit(inline="always")
def _faction_count(
    zones,
    card_faction,
    card_ally_factions,
    game,
    player,
    target,
):
    total = 0
    for card in range(zones.shape[3]):
        count = _active_count(zones, game, player, card)
        if count == 0:
            continue
        contributes = card_ally_factions[card]
        if contributes == 0:
            contributes = card_faction[card]
        if contributes == 15 or (contributes & target) != 0:
            total += count
    return total


@njit(inline="always")
def _bundle_eligible(
    zones,
    card_faction,
    card_ally_factions,
    bundle_trigger,
    bundle_faction,
    bundle_requirement,
    game,
    player,
    card,
    bundle,
):
    trigger = bundle_trigger[card, bundle]
    if trigger != TRIGGER_ALLY:
        return True
    target = bundle_faction[card, bundle]
    count = _faction_count(
        zones,
        card_faction,
        card_ally_factions,
        game,
        player,
        target,
    )
    # Match Effect.apply exactly: count <= requirement fails.
    return count > bundle_requirement[card, bundle]


@njit(inline="always")
def _draw_one(
    zones,
    rng_state,
    game,
    player,
    forced_draws,
    forced_draw_count,
    draw_position,
):
    total = 0
    for card in range(zones.shape[3]):
        total += zones[game, player, ZONE_DECK, card]
    if total == 0:
        for card in range(zones.shape[3]):
            zones[game, player, ZONE_DECK, card] += zones[
                game, player, ZONE_DISCARD, card
            ]
            zones[game, player, ZONE_DISCARD, card] = 0
            total += zones[game, player, ZONE_DECK, card]
    if total == 0:
        return draw_position

    if draw_position < forced_draw_count[game]:
        picked = forced_draws[game, draw_position]
        draw_position += 1
        if picked < 0 or zones[game, player, ZONE_DECK, picked] <= 0:
            # The differential harness supplied an impossible draw.
            return -1000 - draw_position
    else:
        next_rng = _advance_rng(rng_state[game])
        rng_state[game] = next_rng
        ticket = next_rng % total
        cumulative = 0
        picked = 0
        for card in range(zones.shape[3]):
            cumulative += zones[game, player, ZONE_DECK, card]
            if cumulative > ticket:
                picked = card
                break

    zones[game, player, ZONE_DECK, picked] -= 1
    zones[game, player, ZONE_HAND, picked] += 1
    return draw_position


@njit(inline="always")
def _apply_bundle(
    zones,
    trade,
    combat,
    health,
    pending_mode,
    pending_remaining,
    pending_sources,
    pending_mandatory,
    pending_draw_on_complete,
    pending_resolved,
    pending_targets,
    pending_count,
    pending_overflow,
    discard_debt,
    trade_row,
    rng_state,
    game,
    player,
    card,
    bundle,
    bundle_op_start,
    bundle_op_count,
    op_code,
    op_value,
    op_arg,
    card_faction,
    forced_draws,
    forced_draw_count,
    draw_position,
):
    start = bundle_op_start[card, bundle]
    length = bundle_op_count[card, bundle]
    for operation in range(start, start + length):
        opcode = op_code[card, operation]
        value = op_value[card, operation]
        if opcode == OP_ADD_TRADE:
            trade[game, player] += value
        elif opcode == OP_ADD_COMBAT:
            combat[game, player] += value
        elif opcode == OP_ADD_HEALTH:
            health[game, player] += value
        elif opcode == OP_DRAW:
            for _ in range(value):
                draw_position = _draw_one(
                    zones,
                    rng_state,
                    game,
                    player,
                    forced_draws,
                    forced_draw_count,
                    draw_position,
                )
        elif opcode == OP_DRAW_PER_FACTION:
            target = op_arg[card, operation]
            draws = 0
            for active_card in range(zones.shape[3]):
                if (card_faction[active_card] & target) != 0:
                    draws += zones[
                        game, player, ZONE_PLAYED, active_card
                    ]
            for _ in range(draws):
                draw_position = _draw_one(
                    zones,
                    rng_state,
                    game,
                    player,
                    forced_draws,
                    forced_draw_count,
                    draw_position,
                )
        elif opcode == OP_DRAW_IF_BASES_GE_2:
            bases = 0
            for active_card in range(zones.shape[3]):
                bases += zones[game, player, ZONE_BASES, active_card]
            if bases >= op_arg[card, operation]:
                for _ in range(value):
                    draw_position = _draw_one(
                        zones,
                        rng_state,
                        game,
                        player,
                        forced_draws,
                        forced_draw_count,
                        draw_position,
                    )
        elif opcode == OP_QUEUE_SCRAP or opcode == OP_QUEUE_SCRAP_DRAW:
            flags = op_arg[card, operation]
            sources = flags & (
                TARGET_HAND | TARGET_DISCARD | TARGET_TRADE
            )
            available = 0
            active_sources = 0
            if (sources & TARGET_HAND) != 0:
                source_count = 0
                for target in range(zones.shape[3]):
                    source_count += zones[
                        game, player, ZONE_HAND, target
                    ]
                if source_count > 0:
                    active_sources |= TARGET_HAND
                    available += source_count
            if (sources & TARGET_DISCARD) != 0:
                source_count = 0
                for target in range(zones.shape[3]):
                    source_count += zones[
                        game, player, ZONE_DISCARD, target
                    ]
                if source_count > 0:
                    active_sources |= TARGET_DISCARD
                    available += source_count
            if (sources & TARGET_TRADE) != 0:
                source_count = 0
                for slot in range(trade_row.shape[1]):
                    if trade_row[game, slot] >= 0:
                        source_count += 1
                if source_count > 0:
                    active_sources |= TARGET_TRADE
                    available += source_count
            if available > 0:
                if opcode == OP_QUEUE_SCRAP_DRAW:
                    flags |= PENDING_DRAW_ON_COMPLETE
                _queue_pending(
                    pending_mode,
                    pending_remaining,
                    pending_sources,
                    pending_mandatory,
                    pending_draw_on_complete,
                    pending_resolved,
                    pending_targets,
                    pending_count,
                    pending_overflow,
                    zones,
                    trade_row,
                    game,
                    player,
                    PENDING_SCRAP,
                    value,
                    active_sources,
                    flags,
                )
        elif opcode == OP_ADD_DISCARD_DEBT:
            discard_debt[game, 1 - player] += value
        elif opcode == OP_QUEUE_DESTROY_BASE:
            available = 0
            for target in range(zones.shape[3]):
                available += zones[game, 1 - player, ZONE_BASES, target]
            if available > 0:
                _queue_pending(
                    pending_mode,
                    pending_remaining,
                    pending_sources,
                    pending_mandatory,
                    pending_draw_on_complete,
                    pending_resolved,
                    pending_targets,
                    pending_count,
                    pending_overflow,
                    zones,
                    trade_row,
                    game,
                    player,
                    PENDING_DESTROY_BASE,
                    value,
                    0,
                    op_arg[card, operation],
                )
        elif opcode == OP_QUEUE_DISCARD_DRAW:
            available = 0
            for target in range(zones.shape[3]):
                available += zones[game, player, ZONE_HAND, target]
            if available > 0:
                _queue_pending(
                    pending_mode,
                    pending_remaining,
                    pending_sources,
                    pending_mandatory,
                    pending_draw_on_complete,
                    pending_resolved,
                    pending_targets,
                    pending_count,
                    pending_overflow,
                    zones,
                    trade_row,
                    game,
                    player,
                    PENDING_DISCARD,
                    value,
                    TARGET_HAND,
                    op_arg[card, operation]
                    | PENDING_DRAW_ON_COMPLETE,
                )
        # NOP intentionally has no branch.
    return draw_position


@njit(inline="always")
def _refill_market(
    trade_row,
    trade_deck,
    rng_state,
    game,
    slot,
    forced_refills,
    forced_refill_count,
    refill_position,
):
    total = 0
    for card in range(trade_deck.shape[1]):
        total += trade_deck[game, card]
    if total <= 0:
        trade_row[game, slot] = NO_CARD
        return refill_position

    if refill_position < forced_refill_count[game]:
        forced = forced_refills[game, refill_position]
        refill_position += 1
        picked = forced
        if trade_deck[game, picked] <= 0:
            trade_row[game, slot] = -1000
            return refill_position
    else:
        next_rng = _advance_rng(rng_state[game])
        rng_state[game] = next_rng
        ticket = next_rng % total
        cumulative = 0
        picked = 0
        for card in range(trade_deck.shape[1]):
            cumulative += trade_deck[game, card]
            if cumulative > ticket:
                picked = card
                break
    trade_deck[game, picked] -= 1
    trade_row[game, slot] = picked
    return refill_position


@njit(parallel=True, cache=True)
def build_action_masks_kernel(
    zones,
    trade,
    combat,
    trade_row,
    trade_deck,
    explorer_count,
    current_player,
    done,
    effect_state,
    pending_mode,
    pending_sources,
    pending_mandatory,
    pending_targets,
    pending_count,
    supported,
    cost,
    card_type,
    defense,
    card_faction,
    card_ally_factions,
    bundle_count,
    bundle_trigger,
    bundle_faction,
    bundle_requirement,
    masks,
):
    for game in prange(zones.shape[0]):
        for action in range(masks.shape[1]):
            masks[game, action] = False
        if done[game]:
            continue

        player = current_player[game]
        opponent = 1 - player

        if pending_count[game, player] > 0:
            mode = pending_mode[game, player, 0]
            sources = pending_sources[game, player, 0]
            if mode == PENDING_SCRAP:
                if (sources & TARGET_HAND) != 0:
                    for card in range(zones.shape[3]):
                        if pending_targets[
                            game, player, 0, PENDING_TARGET_HAND, card
                        ] > 0:
                            masks[game, SCRAP_HAND_OFFSET + card] = True
                if (sources & TARGET_DISCARD) != 0:
                    for card in range(zones.shape[3]):
                        if pending_targets[
                            game,
                            player,
                            0,
                            PENDING_TARGET_DISCARD,
                            card,
                        ] > 0:
                            masks[game, SCRAP_DISCARD_OFFSET + card] = True
                if (sources & TARGET_TRADE) != 0:
                    for card in range(zones.shape[3]):
                        if pending_targets[
                            game, player, 0, PENDING_TARGET_TRADE, card
                        ] > 0:
                            masks[game, SCRAP_TRADE_OFFSET + card] = True
            elif mode == PENDING_DESTROY_BASE:
                for card in range(zones.shape[3]):
                    if pending_targets[
                        game, player, 0, PENDING_TARGET_BASE, card
                    ] > 0:
                        masks[game, DESTROY_BASE_OFFSET + card] = True
            elif mode == PENDING_DISCARD:
                for card in range(zones.shape[3]):
                    if pending_targets[
                        game, player, 0, PENDING_TARGET_HAND, card
                    ] > 0:
                        masks[game, DISCARD_OFFSET + card] = True
            if not pending_mandatory[game, player, 0]:
                masks[game, SKIP_INDEX] = True
            continue

        for card in range(zones.shape[3]):
            if supported[card] and zones[game, player, ZONE_HAND, card] > 0:
                masks[game, PLAY_OFFSET + card] = True

        for slot in range(trade_row.shape[1]):
            card = trade_row[game, slot]
            if (
                card >= 0
                and supported[card]
                and cost[card] <= trade[game, player]
            ):
                masks[game, BUY_OFFSET + card] = True
        explorer = zones.shape[3] - 1
        if (
            explorer_count[game] > 0
            and supported[explorer]
            and cost[explorer] <= trade[game, player]
        ):
            masks[game, BUY_OFFSET + explorer] = True

        has_outpost = False
        for card in range(zones.shape[3]):
            if (
                zones[game, opponent, ZONE_BASES, card] > 0
                and card_type[card] == 2
            ):
                has_outpost = True
                break
        if combat[game, player] > 0:
            for card in range(zones.shape[3]):
                if zones[game, opponent, ZONE_BASES, card] <= 0:
                    continue
                if has_outpost and card_type[card] != 2:
                    continue
                if defense[card] <= combat[game, player]:
                    masks[game, ATTACK_BASE_OFFSET + card] = True
            if not has_outpost:
                masks[game, ATTACK_PLAYER_INDEX] = True

        for card in range(zones.shape[3]):
            for instance in range(effect_state.shape[3]):
                instance_state = effect_state[
                    game, player, card, instance
                ]
                if (instance_state & ACTIVE_BIT) == 0:
                    break
                for bundle in range(bundle_count[card]):
                    if (instance_state & _bundle_bit(bundle)) == 0:
                        continue
                    if not _bundle_eligible(
                        zones,
                        card_faction,
                        card_ally_factions,
                        bundle_trigger,
                        bundle_faction,
                        bundle_requirement,
                        game,
                        player,
                        card,
                        bundle,
                    ):
                        continue
                    if bundle_trigger[card, bundle] == TRIGGER_SCRAP:
                        masks[game, EFFECT_SCRAP_OFFSET + card] = True
                    else:
                        masks[game, EFFECT_NONSCRAP_OFFSET + card] = True

        masks[game, END_TURN_INDEX] = True


@njit(parallel=True, cache=True)
def build_action_weights_kernel(
    zones,
    trade_row,
    explorer_count,
    current_player,
    done,
    effect_state,
    pending_mode,
    pending_sources,
    pending_mandatory,
    pending_targets,
    pending_count,
    card_faction,
    card_ally_factions,
    bundle_count,
    bundle_trigger,
    bundle_faction,
    bundle_requirement,
    masks,
    weights,
):
    """Physical Action-object multiplicity for scalar RandomAgent parity."""
    for game in prange(zones.shape[0]):
        for action in range(weights.shape[1]):
            weights[game, action] = 0
        if done[game]:
            continue
        player = current_player[game]
        opponent = 1 - player

        if pending_count[game, player] > 0:
            mode = pending_mode[game, player, 0]
            sources = pending_sources[game, player, 0]
            if mode == PENDING_SCRAP:
                if (sources & TARGET_HAND) != 0:
                    for card in range(zones.shape[3]):
                        weights[game, SCRAP_HAND_OFFSET + card] = pending_targets[
                            game, player, 0, PENDING_TARGET_HAND, card
                        ]
                if (sources & TARGET_DISCARD) != 0:
                    for card in range(zones.shape[3]):
                        weights[game, SCRAP_DISCARD_OFFSET + card] = pending_targets[
                            game, player, 0, PENDING_TARGET_DISCARD, card
                        ]
                if (sources & TARGET_TRADE) != 0:
                    for card in range(zones.shape[3]):
                        weights[game, SCRAP_TRADE_OFFSET + card] = pending_targets[
                            game, player, 0, PENDING_TARGET_TRADE, card
                        ]
            elif mode == PENDING_DESTROY_BASE:
                for card in range(zones.shape[3]):
                    weights[game, DESTROY_BASE_OFFSET + card] = pending_targets[
                        game, player, 0, PENDING_TARGET_BASE, card
                    ]
            elif mode == PENDING_DISCARD:
                for card in range(zones.shape[3]):
                    weights[game, DISCARD_OFFSET + card] = pending_targets[
                        game, player, 0, PENDING_TARGET_HAND, card
                    ]
            if not pending_mandatory[game, player, 0]:
                weights[game, SKIP_INDEX] = 1
            continue

        for card in range(zones.shape[3]):
            weights[game, PLAY_OFFSET + card] = zones[
                game, player, ZONE_HAND, card
            ]
        for slot in range(trade_row.shape[1]):
            card = trade_row[game, slot]
            if card >= 0 and masks[game, BUY_OFFSET + card]:
                weights[game, BUY_OFFSET + card] += 1
        explorer = zones.shape[3] - 1
        if explorer_count[game] > 0 and masks[game, BUY_OFFSET + explorer]:
            weights[game, BUY_OFFSET + explorer] += 1
        if masks[game, ATTACK_PLAYER_INDEX]:
            weights[game, ATTACK_PLAYER_INDEX] = 1
        for card in range(zones.shape[3]):
            if masks[game, ATTACK_BASE_OFFSET + card]:
                weights[game, ATTACK_BASE_OFFSET + card] = zones[
                    game, opponent, ZONE_BASES, card
                ]
            for instance in range(effect_state.shape[3]):
                instance_value = effect_state[
                    game, player, card, instance
                ]
                if (instance_value & ACTIVE_BIT) == 0:
                    break
                for bundle in range(bundle_count[card]):
                    if (
                        instance_value & _bundle_bit(bundle)
                    ) == 0 or not _bundle_eligible(
                        zones,
                        card_faction,
                        card_ally_factions,
                        bundle_trigger,
                        bundle_faction,
                        bundle_requirement,
                        game,
                        player,
                        card,
                        bundle,
                    ):
                        continue
                    if bundle_trigger[card, bundle] == TRIGGER_SCRAP:
                        weights[game, EFFECT_SCRAP_OFFSET + card] += 1
                    else:
                        weights[game, EFFECT_NONSCRAP_OFFSET + card] += 1
        weights[game, END_TURN_INDEX] = 1


@njit(parallel=True, cache=True)
def step_kernel(
    zones,
    trade,
    combat,
    health,
    authority_gained,
    trade_row,
    trade_deck,
    explorer_count,
    current_player,
    turns,
    done,
    winner,
    effect_state,
    pending_mode,
    pending_remaining,
    pending_sources,
    pending_mandatory,
    pending_draw_on_complete,
    pending_resolved,
    pending_targets,
    pending_count,
    pending_overflow,
    discard_debt,
    rng_state,
    hand_size,
    turn_cap,
    supported,
    cost,
    card_type,
    defense,
    card_faction,
    card_ally_factions,
    bundle_count,
    bundle_trigger,
    bundle_faction,
    bundle_requirement,
    bundle_stage,
    bundle_next,
    bundle_op_start,
    bundle_op_count,
    op_code,
    op_value,
    op_arg,
    actions,
    action_variants,
    forced_draws,
    forced_draw_count,
    forced_refills,
    forced_refill_count,
):
    for game in prange(zones.shape[0]):
        if done[game]:
            continue
        player = current_player[game]
        opponent = 1 - player
        action = actions[game]
        action_variant = action_variants[game]
        draw_position = 0
        refill_position = 0

        if pending_count[game, player] > 0:
            mode = pending_mode[game, player, 0]
            sources = pending_sources[game, player, 0]
            matched = False
            scrapped_trade = False
            matched_group = -1
            matched_card = -1

            if (
                mode == PENDING_SCRAP
                and SCRAP_HAND_OFFSET
                <= action
                < SCRAP_HAND_OFFSET + zones.shape[3]
                and (sources & TARGET_HAND) != 0
            ):
                card = action - SCRAP_HAND_OFFSET
                if (
                    pending_targets[
                        game, player, 0, PENDING_TARGET_HAND, card
                    ] > 0
                    and zones[game, player, ZONE_HAND, card] > 0
                ):
                    zones[game, player, ZONE_HAND, card] -= 1
                    if card == zones.shape[3] - 1:
                        explorer_count[game] += 1
                    matched = True
                    matched_group = PENDING_TARGET_HAND
                    matched_card = card
            elif (
                mode == PENDING_SCRAP
                and SCRAP_DISCARD_OFFSET
                <= action
                < SCRAP_DISCARD_OFFSET + zones.shape[3]
                and (sources & TARGET_DISCARD) != 0
            ):
                card = action - SCRAP_DISCARD_OFFSET
                if (
                    pending_targets[
                        game, player, 0, PENDING_TARGET_DISCARD, card
                    ] > 0
                    and zones[game, player, ZONE_DISCARD, card] > 0
                ):
                    zones[game, player, ZONE_DISCARD, card] -= 1
                    if card == zones.shape[3] - 1:
                        explorer_count[game] += 1
                    matched = True
                    matched_group = PENDING_TARGET_DISCARD
                    matched_card = card
            elif (
                mode == PENDING_SCRAP
                and SCRAP_TRADE_OFFSET
                <= action
                < SCRAP_TRADE_OFFSET + zones.shape[3]
                and (sources & TARGET_TRADE) != 0
            ):
                card = action - SCRAP_TRADE_OFFSET
                row_length = 0
                selected_slot = -1
                for slot in range(trade_row.shape[1]):
                    if trade_row[game, slot] >= 0:
                        row_length += 1
                    if selected_slot < 0 and trade_row[game, slot] == card:
                        selected_slot = slot
                if (
                    selected_slot >= 0
                    and pending_targets[
                        game, player, 0, PENDING_TARGET_TRADE, card
                    ] > 0
                ):
                    for slot in range(selected_slot, row_length - 1):
                        trade_row[game, slot] = trade_row[game, slot + 1]
                    trade_row[game, row_length - 1] = NO_CARD
                    matched = True
                    scrapped_trade = True
                    matched_group = PENDING_TARGET_TRADE
                    matched_card = card
            elif (
                mode == PENDING_DESTROY_BASE
                and DESTROY_BASE_OFFSET
                <= action
                < DESTROY_BASE_OFFSET + zones.shape[3]
            ):
                card = action - DESTROY_BASE_OFFSET
                if (
                    pending_targets[
                        game, player, 0, PENDING_TARGET_BASE, card
                    ] > 0
                    and zones[game, opponent, ZONE_BASES, card] > 0
                ):
                    zones[game, opponent, ZONE_BASES, card] -= 1
                    zones[game, opponent, ZONE_DISCARD, card] += 1
                    active_variant = 0
                    for instance in range(effect_state.shape[3]):
                        if (
                            effect_state[
                                game, opponent, card, instance
                            ]
                            & ACTIVE_BIT
                        ) != 0:
                            if active_variant != action_variant:
                                active_variant += 1
                                continue
                            _remove_effect_instance(
                                effect_state,
                                game,
                                opponent,
                                card,
                                instance,
                            )
                            break
                    matched = True
                    matched_group = PENDING_TARGET_BASE
                    matched_card = card
            elif (
                mode == PENDING_DISCARD
                and DISCARD_OFFSET
                <= action
                < DISCARD_OFFSET + zones.shape[3]
            ):
                card = action - DISCARD_OFFSET
                if (
                    pending_targets[
                        game, player, 0, PENDING_TARGET_HAND, card
                    ] > 0
                    and zones[game, player, ZONE_HAND, card] > 0
                ):
                    zones[game, player, ZONE_HAND, card] -= 1
                    zones[game, player, ZONE_DISCARD, card] += 1
                    matched = True
                    matched_group = PENDING_TARGET_HAND
                    matched_card = card

            skipped = (
                action == SKIP_INDEX
                and not pending_mandatory[game, player, 0]
            )
            if matched:
                pending_targets[
                    game, player, 0, matched_group, matched_card
                ] -= 1
                pending_remaining[game, player, 0] -= 1
                pending_resolved[game, player, 0] += 1
                if (
                    SCRAP_HAND_OFFSET
                    <= action
                    < SCRAP_HAND_OFFSET + zones.shape[3]
                    or DISCARD_OFFSET
                    <= action
                    < DISCARD_OFFSET + zones.shape[3]
                ):
                    remaining_in_source = pending_targets[
                        game, player, 0, PENDING_TARGET_HAND
                    ].sum()
                    if remaining_in_source == 0:
                        pending_sources[game, player, 0] &= (
                            0xFFFF ^ TARGET_HAND
                        )
                elif (
                    SCRAP_DISCARD_OFFSET
                    <= action
                    < SCRAP_DISCARD_OFFSET + zones.shape[3]
                ):
                    remaining_in_source = pending_targets[
                        game, player, 0, PENDING_TARGET_DISCARD
                    ].sum()
                    if remaining_in_source == 0:
                        pending_sources[game, player, 0] &= (
                            0xFFFF ^ TARGET_DISCARD
                        )
                elif (
                    SCRAP_TRADE_OFFSET
                    <= action
                    < SCRAP_TRADE_OFFSET + zones.shape[3]
                ):
                    remaining_in_source = pending_targets[
                        game, player, 0, PENDING_TARGET_TRADE
                    ].sum()
                    if remaining_in_source == 0:
                        pending_sources[game, player, 0] &= (
                            0xFFFF ^ TARGET_TRADE
                        )
            completed = skipped or (
                matched and pending_remaining[game, player, 0] <= 0
            )
            if completed:
                if scrapped_trade:
                    for slot in range(trade_row.shape[1]):
                        if trade_row[game, slot] < 0:
                            refill_position = _refill_market(
                                trade_row,
                                trade_deck,
                                rng_state,
                                game,
                                slot,
                                forced_refills,
                                forced_refill_count,
                                refill_position,
                            )
                if pending_draw_on_complete[game, player, 0]:
                    draws = pending_resolved[game, player, 0]
                    for _ in range(draws):
                        draw_position = _draw_one(
                            zones,
                            rng_state,
                            game,
                            player,
                            forced_draws,
                            forced_draw_count,
                            draw_position,
                        )
                _pop_pending(
                    pending_mode,
                    pending_remaining,
                    pending_sources,
                    pending_mandatory,
                    pending_draw_on_complete,
                    pending_resolved,
                    pending_targets,
                    pending_count,
                    game,
                    player,
                )
            continue

        if PLAY_OFFSET <= action < PLAY_OFFSET + zones.shape[3]:
            card = action - PLAY_OFFSET
            if supported[card] and zones[game, player, ZONE_HAND, card] > 0:
                zones[game, player, ZONE_HAND, card] -= 1
                zones[game, player, ZONE_PLAYED, card] += 1
                if card_type[card] != 0:
                    zones[game, player, ZONE_BASES, card] += 1
                _add_effect_instance(
                    effect_state,
                    bundle_count,
                    bundle_trigger,
                    bundle_stage,
                    game,
                    player,
                    card,
                )
                for bundle in range(bundle_count[card]):
                    trigger = bundle_trigger[card, bundle]
                    if trigger == TRIGGER_AUTO:
                        draw_position = _apply_bundle(
                            zones,
                            trade,
                            combat,
                            health,
                            pending_mode,
                            pending_remaining,
                            pending_sources,
                            pending_mandatory,
                            pending_draw_on_complete,
                            pending_resolved,
                            pending_targets,
                            pending_count,
                            pending_overflow,
                            discard_debt,
                            trade_row,
                            rng_state,
                            game,
                            player,
                            card,
                            bundle,
                            bundle_op_start,
                            bundle_op_count,
                            op_code,
                            op_value,
                            op_arg,
                            card_faction,
                            forced_draws,
                            forced_draw_count,
                            draw_position,
                        )

        elif BUY_OFFSET <= action < BUY_OFFSET + zones.shape[3]:
            card = action - BUY_OFFSET
            if supported[card] and trade[game, player] >= cost[card]:
                bought = False
                explorer = zones.shape[3] - 1
                if card == explorer and explorer_count[game] > 0:
                    explorer_count[game] -= 1
                    bought = True
                else:
                    for slot in range(trade_row.shape[1]):
                        if trade_row[game, slot] == card:
                            row_length = 0
                            for market_slot in range(trade_row.shape[1]):
                                if trade_row[game, market_slot] >= 0:
                                    row_length += 1
                            for market_slot in range(slot, row_length - 1):
                                trade_row[game, market_slot] = trade_row[
                                    game, market_slot + 1
                                ]
                            refill_slot = row_length - 1
                            trade_row[game, refill_slot] = NO_CARD
                            refill_position = _refill_market(
                                trade_row,
                                trade_deck,
                                rng_state,
                                game,
                                refill_slot,
                                forced_refills,
                                forced_refill_count,
                                refill_position,
                            )
                            bought = True
                            break
                if bought:
                    trade[game, player] -= cost[card]
                    zones[game, player, ZONE_DISCARD, card] += 1

        elif action == ATTACK_PLAYER_INDEX:
            has_outpost = False
            for card in range(zones.shape[3]):
                if (
                    zones[game, opponent, ZONE_BASES, card] > 0
                    and card_type[card] == 2
                ):
                    has_outpost = True
                    break
            if combat[game, player] > 0 and not has_outpost:
                health[game, opponent] -= combat[game, player]
                combat[game, player] = 0
                if health[game, opponent] <= 0:
                    done[game] = True
                    winner[game] = player

        elif (
            ATTACK_BASE_OFFSET
            <= action
            < ATTACK_BASE_OFFSET + zones.shape[3]
        ):
            card = action - ATTACK_BASE_OFFSET
            has_outpost = False
            for active in range(zones.shape[3]):
                if (
                    zones[game, opponent, ZONE_BASES, active] > 0
                    and card_type[active] == 2
                ):
                    has_outpost = True
                    break
            if (
                zones[game, opponent, ZONE_BASES, card] > 0
                and defense[card] <= combat[game, player]
                and (not has_outpost or card_type[card] == 2)
            ):
                combat[game, player] -= defense[card]
                zones[game, opponent, ZONE_BASES, card] -= 1
                zones[game, opponent, ZONE_DISCARD, card] += 1
                active_variant = 0
                for instance in range(effect_state.shape[3]):
                    if (
                        effect_state[game, opponent, card, instance]
                        & ACTIVE_BIT
                    ) != 0:
                        if active_variant != action_variant:
                            active_variant += 1
                            continue
                        _remove_effect_instance(
                            effect_state,
                            game,
                            opponent,
                            card,
                            instance,
                        )
                        break

        elif (
            EFFECT_NONSCRAP_OFFSET
            <= action
            < EFFECT_NONSCRAP_OFFSET + zones.shape[3]
        ):
            card = action - EFFECT_NONSCRAP_OFFSET
            applied = False
            candidate_variant = 0
            for instance in range(effect_state.shape[3]):
                instance_state = effect_state[
                    game, player, card, instance
                ]
                if (instance_state & ACTIVE_BIT) == 0:
                    break
                for bundle in range(bundle_count[card]):
                    if (
                        (instance_state & _bundle_bit(bundle)) != 0
                        and bundle_trigger[card, bundle] != TRIGGER_SCRAP
                        and _bundle_eligible(
                            zones,
                            card_faction,
                            card_ally_factions,
                            bundle_trigger,
                            bundle_faction,
                            bundle_requirement,
                            game,
                            player,
                            card,
                            bundle,
                        )
                    ):
                        if candidate_variant != action_variant:
                            candidate_variant += 1
                            continue
                        draw_position = _apply_bundle(
                            zones,
                            trade,
                            combat,
                            health,
                            pending_mode,
                            pending_remaining,
                            pending_sources,
                            pending_mandatory,
                            pending_draw_on_complete,
                            pending_resolved,
                            pending_targets,
                            pending_count,
                            pending_overflow,
                            discard_debt,
                            trade_row,
                            rng_state,
                            game,
                            player,
                            card,
                            bundle,
                            bundle_op_start,
                            bundle_op_count,
                            op_code,
                            op_value,
                            op_arg,
                            card_faction,
                            forced_draws,
                            forced_draw_count,
                            draw_position,
                        )
                        next_state = instance_state & (
                            0xFFFF ^ _bundle_bit(bundle)
                        )
                        effect_state[
                            game, player, card, instance
                        ] = next_state
                        applied = True
                        break
                if applied:
                    break

        elif (
            EFFECT_SCRAP_OFFSET
            <= action
            < EFFECT_SCRAP_OFFSET + zones.shape[3]
        ):
            card = action - EFFECT_SCRAP_OFFSET
            scrapped = False
            candidate_variant = 0
            for instance in range(effect_state.shape[3]):
                instance_state = effect_state[
                    game, player, card, instance
                ]
                if (instance_state & ACTIVE_BIT) == 0:
                    break
                for bundle in range(bundle_count[card]):
                    if (
                        (instance_state & _bundle_bit(bundle)) != 0
                        and bundle_trigger[card, bundle] == TRIGGER_SCRAP
                    ):
                        if candidate_variant != action_variant:
                            candidate_variant += 1
                            continue
                        draw_position = _apply_bundle(
                            zones,
                            trade,
                            combat,
                            health,
                            pending_mode,
                            pending_remaining,
                            pending_sources,
                            pending_mandatory,
                            pending_draw_on_complete,
                            pending_resolved,
                            pending_targets,
                            pending_count,
                            pending_overflow,
                            discard_debt,
                            trade_row,
                            rng_state,
                            game,
                            player,
                            card,
                            bundle,
                            bundle_op_start,
                            bundle_op_count,
                            op_code,
                            op_value,
                            op_arg,
                            card_faction,
                            forced_draws,
                            forced_draw_count,
                            draw_position,
                        )
                        if zones[game, player, ZONE_PLAYED, card] > 0:
                            zones[game, player, ZONE_PLAYED, card] -= 1
                        if zones[game, player, ZONE_BASES, card] > 0:
                            zones[game, player, ZONE_BASES, card] -= 1
                        _remove_effect_instance(
                            effect_state,
                            game,
                            player,
                            card,
                            instance,
                        )
                        if card == zones.shape[3] - 1:
                            explorer_count[game] += 1
                        scrapped = True
                        break
                if scrapped:
                    break

        elif action == END_TURN_INDEX:
            for card in range(zones.shape[3]):
                ships = (
                    zones[game, player, ZONE_PLAYED, card]
                    - zones[game, player, ZONE_BASES, card]
                )
                if ships > 0:
                    zones[game, player, ZONE_DISCARD, card] += ships
                zones[game, player, ZONE_DISCARD, card] += zones[
                    game, player, ZONE_HAND, card
                ]
                zones[game, player, ZONE_HAND, card] = 0
                zones[game, player, ZONE_PLAYED, card] = 0
                if card_type[card] == 0:
                    for instance in range(effect_state.shape[3]):
                        effect_state[
                            game, player, card, instance
                        ] = 0
                else:
                    ready_bits = ACTIVE_BIT
                    for bundle in range(bundle_count[card]):
                        ready_bits |= _bundle_bit(bundle)
                    base_copies = zones[
                        game, player, ZONE_BASES, card
                    ]
                    for instance in range(effect_state.shape[3]):
                        effect_state[
                            game, player, card, instance
                        ] = (
                            ready_bits if instance < base_copies else 0
                        )
            for _ in range(hand_size[game]):
                draw_position = _draw_one(
                    zones,
                    rng_state,
                    game,
                    player,
                    forced_draws,
                    forced_draw_count,
                    draw_position,
                )
            trade[game, player] = 0
            combat[game, player] = 0
            authority_gained[game, player] = 0
            turns[game] += 1
            if turns[game] > turn_cap[game]:
                done[game] = True
                if health[game, 0] >= health[game, 1]:
                    winner[game] = 0
                else:
                    winner[game] = 1
            current_player[game] = opponent
            debt = discard_debt[game, opponent]
            if not done[game] and debt > 0:
                cards_in_hand = 0
                for card in range(zones.shape[3]):
                    cards_in_hand += zones[
                        game, opponent, ZONE_HAND, card
                    ]
                remaining = debt
                if remaining > cards_in_hand:
                    remaining = cards_in_hand
                discard_debt[game, opponent] = 0
                if remaining > 0:
                    _queue_pending(
                        pending_mode,
                        pending_remaining,
                        pending_sources,
                        pending_mandatory,
                        pending_draw_on_complete,
                        pending_resolved,
                        pending_targets,
                        pending_count,
                        pending_overflow,
                        zones,
                        trade_row,
                        game,
                        opponent,
                        PENDING_DISCARD,
                        remaining,
                        TARGET_HAND,
                        PENDING_MANDATORY,
                    )


@njit(parallel=True, cache=True)
def materialize_observations_kernel(
    zones,
    trade,
    combat,
    health,
    trade_row,
    current_player,
    first_player,
    training_player,
    masks,
    observations,
):
    for game in prange(zones.shape[0]):
        for column in range(observations.shape[1]):
            observations[game, column] = 0.0

        training = training_player[game]
        opponent = 1 - training
        observations[game, 0] = (
            1.0 if current_player[game] == training else 0.0
        )
        observations[game, 1] = (
            1.0 if first_player[game] == training else 0.0
        )
        can_buy = False
        for card in range(NUM_CARDS):
            if masks[game, BUY_OFFSET + card]:
                can_buy = True
                break
        observations[game, 2] = 1.0 if can_buy else 0.0
        has_actions = masks[game, ATTACK_PLAYER_INDEX]
        if not has_actions:
            for card in range(NUM_CARDS):
                if masks[game, PLAY_OFFSET + card]:
                    has_actions = True
                    break
        observations[game, 3] = 1.0 if has_actions else 0.0
        offset = 4

        for slot in range(trade_row.shape[1]):
            card = trade_row[game, slot]
            if card >= 0:
                value = observations[game, offset + card] + 0.125
                observations[game, offset + card] = (
                    1.0 if value > 1.0 else value
                )
        offset += NUM_CARDS

        observations[game, offset + 0] = trade[game, training] / 100.0
        observations[game, offset + 1] = combat[game, training] / 100.0
        observations[game, offset + 2] = health[game, training] / 100.0
        observations[game, offset + 3] = (
            zones[game, training, ZONE_DECK].sum() / 40.0
        )
        observations[game, offset + 4] = (
            zones[game, training, ZONE_DISCARD].sum() / 40.0
        )
        offset += 5
        for zone in (
            ZONE_HAND,
            ZONE_PLAYED,
            ZONE_DISCARD,
            ZONE_DECK,
            ZONE_BASES,
        ):
            for card in range(NUM_CARDS):
                value = zones[game, training, zone, card] * 0.125
                observations[game, offset + card] = (
                    1.0 if value > 1.0 else value
                )
            offset += NUM_CARDS

        observations[game, offset + 0] = trade[game, opponent] / 100.0
        observations[game, offset + 1] = combat[game, opponent] / 100.0
        observations[game, offset + 2] = health[game, opponent] / 100.0
        observations[game, offset + 3] = (
            zones[game, opponent, ZONE_DECK].sum() / 40.0
        )
        observations[game, offset + 4] = (
            zones[game, opponent, ZONE_HAND].sum() / 10.0
        )
        observations[game, offset + 5] = (
            zones[game, opponent, ZONE_DISCARD].sum() / 40.0
        )
        offset += 6
        for card in range(NUM_CARDS):
            value = (
                zones[game, opponent, ZONE_HAND, card]
                + zones[game, opponent, ZONE_DECK, card]
            ) * 0.125
            observations[game, offset + card] = (
                1.0 if value > 1.0 else value
            )
        offset += NUM_CARDS
        for zone in (ZONE_PLAYED, ZONE_DISCARD, ZONE_BASES):
            for card in range(NUM_CARDS):
                value = zones[game, opponent, zone, card] * 0.125
                observations[game, offset + card] = (
                    1.0 if value > 1.0 else value
                )
            offset += NUM_CARDS


@dataclass
class VectorEngine:
    state: VectorState
    rules: CompiledRules

    def __post_init__(self):
        if self.state.num_cards != self.rules.num_cards:
            raise ValueError("state and rules card widths differ")
        if (
            self.rules.num_cards != NUM_CARDS
            or self.rules.card_names[-3:]
            != ("Scout", "Viper", "Explorer")
        ):
            raise ValueError(
                "vector engine requires the canonical 49-card ordering "
                "ending in Scout, Viper, Explorer"
            )
        if not self.rules.supported.all():
            unsupported = [
                self.rules.card_names[index]
                for index in range(self.rules.num_cards)
                if not self.rules.supported[index]
            ]
            raise ValueError(
                f"vector rules do not cover every card: {unsupported}"
            )
        self.masks = np.zeros(
            (self.state.games, ACTION_DIM), dtype=np.bool_
        )
        self.action_weights = np.zeros(
            (self.state.games, ACTION_DIM), dtype=np.int16
        )
        self.observations = np.zeros(
            (self.state.games, 505), dtype=np.float32
        )
        self._empty_forced_draws = np.full(
            (self.state.games, MAX_FORCED_DRAWS),
            NO_CARD,
            dtype=np.int16,
        )
        self._empty_forced_draw_count = np.zeros(
            self.state.games, dtype=np.int8
        )
        self._empty_forced_refills = np.full(
            (self.state.games, MAX_FORCED_REFILLS),
            NO_CARD,
            dtype=np.int16,
        )
        self._empty_forced_refill_count = np.zeros(
            self.state.games, dtype=np.int8
        )

    def build_masks(self) -> np.ndarray:
        s, r = self.state, self.rules
        build_action_masks_kernel(
            s.zones,
            s.trade,
            s.combat,
            s.trade_row,
            s.trade_deck,
            s.explorer_count,
            s.current_player,
            s.done,
            s.effect_state,
            s.pending_mode,
            s.pending_sources,
            s.pending_mandatory,
            s.pending_targets,
            s.pending_count,
            r.supported,
            r.cost,
            r.card_type,
            r.defense,
            r.faction,
            r.ally_factions,
            r.bundle_count,
            r.bundle_trigger,
            r.bundle_faction,
            r.bundle_requirement,
            self.masks,
        )
        return self.masks

    def materialize_observations(self) -> np.ndarray:
        materialize_observations_kernel(
            self.state.zones,
            self.state.trade,
            self.state.combat,
            self.state.health,
            self.state.trade_row,
            self.state.current_player,
            self.state.first_player,
            self.state.training_player,
            self.masks,
            self.observations,
        )
        return self.observations

    def build_action_weights(self) -> np.ndarray:
        """Physical scalar-Action multiplicities for random opponents."""
        s, r = self.state, self.rules
        build_action_weights_kernel(
            s.zones,
            s.trade_row,
            s.explorer_count,
            s.current_player,
            s.done,
            s.effect_state,
            s.pending_mode,
            s.pending_sources,
            s.pending_mandatory,
            s.pending_targets,
            s.pending_count,
            r.faction,
            r.ally_factions,
            r.bundle_count,
            r.bundle_trigger,
            r.bundle_faction,
            r.bundle_requirement,
            self.masks,
            self.action_weights,
        )
        return self.action_weights

    def step(
        self,
        actions: np.ndarray,
        *,
        action_variants: np.ndarray | None = None,
        forced_draws: np.ndarray | None = None,
        forced_draw_count: np.ndarray | None = None,
        forced_refill: np.ndarray | None = None,
    ) -> None:
        games = self.state.games
        if actions.shape != (games,):
            raise ValueError(f"actions must have shape ({games},)")
        if action_variants is None:
            action_variants = np.zeros(games, dtype=np.int16)
        elif action_variants.shape != (games,):
            raise ValueError(f"action_variants must have shape ({games},)")
        if forced_draws is None:
            forced_draws = self._empty_forced_draws
        elif forced_draws.shape[0] != games:
            raise ValueError("forced_draws batch size differs from state")
        if forced_draw_count is None:
            forced_draw_count = self._empty_forced_draw_count
        elif (
            forced_draw_count.shape != (games,)
            or np.any(forced_draw_count > forced_draws.shape[1])
        ):
            raise ValueError("forced_draw_count exceeds tape width")
        if forced_refill is None:
            forced_refills = self._empty_forced_refills
            forced_refill_count = self._empty_forced_refill_count
        elif forced_refill.ndim == 1:
            forced_refills = np.full(
                (games, MAX_FORCED_REFILLS), NO_CARD, dtype=np.int16
            )
            forced_refills[:, 0] = forced_refill
            forced_refill_count = (forced_refill >= 0).astype(np.int8)
        else:
            if forced_refill.shape[0] != games:
                raise ValueError(
                    "forced_refill batch size differs from state"
                )
            forced_refills = forced_refill
            forced_refill_count = (forced_refill >= 0).sum(axis=1).astype(
                np.int8
            )
        s, r = self.state, self.rules
        step_kernel(
            s.zones,
            s.trade,
            s.combat,
            s.health,
            s.authority_gained,
            s.trade_row,
            s.trade_deck,
            s.explorer_count,
            s.current_player,
            s.turns,
            s.done,
            s.winner,
            s.effect_state,
            s.pending_mode,
            s.pending_remaining,
            s.pending_sources,
            s.pending_mandatory,
            s.pending_draw_on_complete,
            s.pending_resolved,
            s.pending_targets,
            s.pending_count,
            s.pending_overflow,
            s.discard_debt,
            s.rng_state,
            s.hand_size,
            s.turn_cap,
            r.supported,
            r.cost,
            r.card_type,
            r.defense,
            r.faction,
            r.ally_factions,
            r.bundle_count,
            r.bundle_trigger,
            r.bundle_faction,
            r.bundle_requirement,
            r.bundle_stage,
            r.bundle_next,
            r.bundle_op_start,
            r.bundle_op_count,
            r.op_code,
            r.op_value,
            r.op_arg,
            actions.astype(np.int32, copy=False),
            action_variants.astype(np.int16, copy=False),
            forced_draws,
            forced_draw_count,
            forced_refills,
            forced_refill_count,
        )
        if self.state.pending_overflow.any():
            games = np.flatnonzero(self.state.pending_overflow)
            raise RuntimeError(
                f"vector pending queue overflow in games {games[:8].tolist()}"
            )


def configure_numba_threads(threads: int = 16) -> None:
    maximum = numba.config.NUMBA_NUM_THREADS
    if threads <= 0 or threads > maximum:
        raise ValueError(
            f"threads must be in 1..{maximum}"
        )
    numba.set_num_threads(threads)
