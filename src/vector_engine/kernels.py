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
)
from src.vector_engine.state import (
    INSTANCE_ACTIVE,
    INSTANCE_PLAYED_THIS_TURN,
    NO_CARD,
    NO_WINNER,
    NUM_PLAYERS,
    NUM_ZONES,
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
EFFECT_NONSCRAP_OFFSET = _GROUPS["effect_nonscrap"]
EFFECT_SCRAP_OFFSET = _GROUPS["effect_scrap"]

LCG_MULT = 1_664_525
LCG_ADD = 1_013_904_223
LCG_MASK = 0x7FFF_FFFF
MAX_FORCED_DRAWS = 16
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


@njit(inline="always")
def _bundle_bit(bundle):
    return 1 << bundle


@njit(inline="always")
def _add_effect_instance(
    effect_state,
    bundle_count,
    bundle_trigger,
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
        next_rng = (rng_state[game] * LCG_MULT + LCG_ADD) & LCG_MASK
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
        # NOP intentionally has no branch.
    return draw_position


@njit(inline="always")
def _refill_market(
    trade_row,
    trade_deck,
    rng_state,
    game,
    slot,
    forced_refill,
):
    total = 0
    for card in range(trade_deck.shape[1]):
        total += trade_deck[game, card]
    if total <= 0:
        trade_row[game, slot] = NO_CARD
        return

    forced = forced_refill[game]
    if forced >= 0:
        picked = forced
        if trade_deck[game, picked] <= 0:
            trade_row[game, slot] = -1000
            return
    else:
        next_rng = (rng_state[game] * LCG_MULT + LCG_ADD) & LCG_MASK
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
    bundle_op_start,
    bundle_op_count,
    op_code,
    op_value,
    op_arg,
    actions,
    forced_draws,
    forced_draw_count,
    forced_refill,
):
    for game in prange(zones.shape[0]):
        if done[game]:
            continue
        player = current_player[game]
        opponent = 1 - player
        action = actions[game]
        draw_position = 0

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
                            _refill_market(
                                trade_row,
                                trade_deck,
                                rng_state,
                                game,
                                refill_slot,
                                forced_refill,
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
                for instance in range(effect_state.shape[3]):
                    if (
                        effect_state[game, opponent, card, instance]
                        & ACTIVE_BIT
                    ) != 0:
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
                        draw_position = _apply_bundle(
                            zones,
                            trade,
                            combat,
                            health,
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
                        effect_state[
                            game, player, card, instance
                        ] = instance_state & (
                            0xFFFF ^ _bundle_bit(bundle)
                        )
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
                        draw_position = _apply_bundle(
                            zones,
                            trade,
                            combat,
                            health,
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
        self.masks = np.zeros(
            (self.state.games, ACTION_DIM), dtype=np.bool_
        )
        self.observations = np.zeros(
            (self.state.games, 505), dtype=np.float32
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

    def step(
        self,
        actions: np.ndarray,
        *,
        forced_draws: np.ndarray | None = None,
        forced_draw_count: np.ndarray | None = None,
        forced_refill: np.ndarray | None = None,
    ) -> None:
        games = self.state.games
        if actions.shape != (games,):
            raise ValueError(f"actions must have shape ({games},)")
        if forced_draws is None:
            forced_draws = np.full(
                (games, MAX_FORCED_DRAWS), NO_CARD, dtype=np.int16
            )
        if forced_draw_count is None:
            forced_draw_count = np.zeros(games, dtype=np.int8)
        if forced_refill is None:
            forced_refill = np.full(games, NO_CARD, dtype=np.int16)
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
            r.bundle_op_start,
            r.bundle_op_count,
            r.op_code,
            r.op_value,
            r.op_arg,
            actions.astype(np.int32, copy=False),
            forced_draws,
            forced_draw_count,
            forced_refill,
        )


def configure_numba_threads(threads: int = 16) -> None:
    if threads <= 0 or threads > numba.config.NUMBA_DEFAULT_NUM_THREADS:
        raise ValueError(
            f"threads must be in 1..{numba.config.NUMBA_DEFAULT_NUM_THREADS}"
        )
    numba.set_num_threads(threads)
