"""Batched fixed-opponent policies for the vector engine."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numba import njit, prange

from src.cards.effects import CardEffectType
from src.rules.ir import EffectOpcode
from src.vector_engine.kernels import (
    ATTACK_BASE_OFFSET,
    ATTACK_PLAYER_INDEX,
    BUY_OFFSET,
    DESTROY_BASE_OFFSET,
    DISCARD_OFFSET,
    EFFECT_NONSCRAP_OFFSET,
    EFFECT_SCRAP_OFFSET,
    END_TURN_INDEX,
    NUM_CARDS,
    PLAY_OFFSET,
    SCRAP_DISCARD_OFFSET,
    SCRAP_HAND_OFFSET,
    SCRAP_TRADE_OFFSET,
    SKIP_INDEX,
    TRIGGER_SCRAP,
    _advance_rng,
    _bundle_bit,
    _bundle_eligible,
)
from src.vector_engine.rules import CompiledRules, MAX_BUNDLES
from src.vector_engine.state import (
    INSTANCE_ACTIVE,
    PENDING_ACQUIRE_SHIP_TOP,
    PENDING_COPY_SHIP,
    PENDING_DESTROY_BASE,
    PENDING_DISCARD,
    PENDING_SCRAP,
    PENDING_TARGET_BASE,
    PENDING_TARGET_DISCARD,
    PENDING_TARGET_HAND,
    PENDING_TARGET_TRADE,
    SOURCE_DISCARD,
    SOURCE_HAND,
    SOURCE_TRADE,
    ZONE_BASES,
    ZONE_DECK,
    ZONE_DISCARD,
    ZONE_HAND,
    ZONE_PLAYED,
)


class FixedOpponent(IntEnum):
    RANDOM = 0
    HEURISTIC = 1
    SIMPLE = 2


OPPONENT_CODE = {
    "random": int(FixedOpponent.RANDOM),
    "heuristic": int(FixedOpponent.HEURISTIC),
    "simple": int(FixedOpponent.SIMPLE),
}
OPPONENT_LABEL = {
    int(FixedOpponent.RANDOM): "Random",
    int(FixedOpponent.HEURISTIC): "Heuristic",
    int(FixedOpponent.SIMPLE): "Simple",
}
OPPONENT_RANDOM = int(FixedOpponent.RANDOM)
OPPONENT_HEURISTIC = int(FixedOpponent.HEURISTIC)
OPPONENT_SIMPLE = int(FixedOpponent.SIMPLE)
SNAPSHOT_OFFSET = len(FixedOpponent)


@dataclass(frozen=True)
class OpponentPolicyData:
    play_priority: np.ndarray
    buy_score_base: np.ndarray
    bundle_priority: np.ndarray
    bundle_is_or: np.ndarray
    discard_value: np.ndarray
    scout: int
    viper: int
    explorer: int


def _effect_priority(effect) -> int:
    value = effect.value or 0
    if effect.effect_type == CardEffectType.DRAW:
        return 100 + value
    if effect.effect_type == CardEffectType.TRADE:
        return 80 + value
    if effect.effect_type == CardEffectType.COMBAT:
        return 60 + value
    if effect.effect_type == CardEffectType.TARGET_DISCARD:
        return 55
    if effect.effect_type == CardEffectType.DESTROY_BASE:
        return 50
    if effect.effect_type == CardEffectType.HEAL:
        return 40 + value
    if effect.effect_type == CardEffectType.SCRAP:
        return 30
    return 10


def _priority_from_opcode(rules: CompiledRules, card: int, bundle: int) -> int:
    start = int(rules.bundle_op_start[card, bundle])
    if int(rules.bundle_op_count[card, bundle]) == 0:
        return 10
    opcode = int(rules.op_code[card, start])
    value = int(rules.op_value[card, start])
    if opcode == int(EffectOpcode.DRAW):
        return 100 + value
    if opcode == int(EffectOpcode.ADD_TRADE):
        return 80 + value
    if opcode == int(EffectOpcode.ADD_COMBAT):
        return 60 + value
    if opcode == int(EffectOpcode.ADD_DISCARD_DEBT):
        return 55
    if opcode == int(EffectOpcode.QUEUE_DESTROY_BASE):
        return 50
    if opcode == int(EffectOpcode.ADD_HEALTH):
        return 40 + value
    if opcode == int(EffectOpcode.QUEUE_SCRAP):
        return 30
    return 10


def compile_opponent_policy_data(
    cards,
    registry,
    rules: CompiledRules,
) -> OpponentPolicyData:
    """Compile static card scores used by the scalar HeuristicAgent."""
    templates = {}
    for card in cards:
        templates.setdefault(card.name, card)

    play_priority = rules.cost.astype(np.int16, copy=True)
    buy_score_base = (rules.cost.astype(np.int16) * 4).astype(np.int16)
    bundle_priority = np.full(
        (rules.num_cards, MAX_BUNDLES), 10, dtype=np.int16
    )
    bundle_is_or = np.zeros(
        (rules.num_cards, MAX_BUNDLES), dtype=np.bool_
    )
    discard_value = np.full(rules.num_cards, 10, dtype=np.int8)

    scout = registry.card_index_map["Scout"]
    viper = registry.card_index_map["Viper"]
    explorer = registry.card_index_map["Explorer"]
    discard_value[scout] = 1
    discard_value[viper] = 2
    discard_value[explorer] = 3

    for card_index, name in enumerate(rules.card_names):
        card = templates.get(name)
        if card is not None:
            draw_value = sum(
                effect.value * 10
                for effect in card.effects
                if effect.effect_type == CardEffectType.DRAW
            )
            play_priority[card_index] = draw_value + card.cost

            score = card.cost * 4
            for effect in card.effects:
                value = effect.value or 0
                if effect.effect_type == CardEffectType.DRAW:
                    score += value * 8
                elif effect.effect_type == CardEffectType.COMBAT:
                    score += value * 3
                elif (
                    effect.effect_type == CardEffectType.SCRAP
                    and not effect.is_scrap_effect
                ):
                    score += 6
                elif effect.effect_type == CardEffectType.TARGET_DISCARD:
                    score += 4
                elif effect.effect_type == CardEffectType.DESTROY_BASE:
                    score += 4
            if int(rules.card_type[card_index]) == 2:
                score += 8 + int(rules.defense[card_index])
            elif int(rules.card_type[card_index]) == 1:
                score += 6 + int(rules.defense[card_index])
            buy_score_base[card_index] = score

        for bundle in range(int(rules.bundle_count[card_index])):
            if card is None:
                priority = _priority_from_opcode(rules, card_index, bundle)
            else:
                source_index = int(
                    rules.bundle_source_effect[card_index, bundle]
                )
                source = card.effects[source_index]
                if source.is_or_effect and source.child_effects:
                    bundle_is_or[card_index, bundle] = True
                    stage = int(rules.bundle_stage[card_index, bundle])
                    source = source.child_effects[
                        min(stage, len(source.child_effects) - 1)
                    ]
                priority = _effect_priority(source)
            bundle_priority[card_index, bundle] = priority

    return OpponentPolicyData(
        play_priority=play_priority,
        buy_score_base=buy_score_base,
        bundle_priority=bundle_priority,
        bundle_is_or=bundle_is_or,
        discard_value=discard_value,
        scout=scout,
        viper=viper,
        explorer=explorer,
    )


def sample_opponent_kinds(
    entries: list[tuple[str, float]],
    games: int,
    seed: int,
) -> np.ndarray:
    """Sample one fixed opponent type per resident game."""
    names = [name for name, _ in entries]
    probabilities = np.asarray([weight for _, weight in entries], dtype=np.float64)
    rng = np.random.default_rng(seed ^ 0x5EED_0FF5)
    choices = rng.choice(len(entries), size=games, p=probabilities)
    return np.asarray(
        [OPPONENT_CODE[names[index]] for index in choices],
        dtype=np.int8,
    )


def sample_mixed_opponent_kinds(
    entries: list[tuple[str, float]],
    snapshot_count: int,
    self_play_ratio: float,
    pfsp_weights: list[float] | None,
    games: int,
    seed: int,
) -> np.ndarray:
    """Sample stable fixed or snapshot assignments for resident games."""
    if snapshot_count < 0:
        raise ValueError("snapshot_count must be non-negative")
    if not 0.0 <= self_play_ratio <= 1.0:
        raise ValueError("self_play_ratio must be in [0, 1]")
    if snapshot_count == 0:
        return sample_opponent_kinds(entries, games, seed).astype(
            np.int16
        )

    weights = (
        np.ones(snapshot_count, dtype=np.float64)
        if pfsp_weights is None
        else np.asarray(pfsp_weights, dtype=np.float64)
    )
    if weights.shape != (snapshot_count,):
        raise ValueError("PFSP weights must match snapshot count")
    if np.any(weights <= 0) or not np.isfinite(weights).all():
        raise ValueError("PFSP weights must be finite and positive")
    weights /= weights.sum()

    fixed_names = [name for name, _ in entries]
    fixed_weights = np.asarray(
        [weight for _, weight in entries], dtype=np.float64
    )
    rng = np.random.default_rng(seed ^ 0x5EED_0FF5)
    use_snapshot = rng.random(games) < self_play_ratio
    assignments = np.empty(games, dtype=np.int16)

    fixed_count = int((~use_snapshot).sum())
    if fixed_count:
        fixed_choices = rng.choice(
            len(entries), size=fixed_count, p=fixed_weights
        )
        assignments[~use_snapshot] = np.asarray(
            [OPPONENT_CODE[fixed_names[index]] for index in fixed_choices],
            dtype=np.int16,
        )
    snapshot_games = int(use_snapshot.sum())
    if snapshot_games:
        snapshot_choices = rng.choice(
            snapshot_count, size=snapshot_games, p=weights
        )
        assignments[use_snapshot] = SNAPSHOT_OFFSET + snapshot_choices
    return assignments


@njit(inline="always")
def _weighted_action(weights, rng_state, game, start, end):
    count = 0
    for action in range(start, end):
        count += weights[game, action]
    if count <= 0:
        return -1, 0
    next_rng = _advance_rng(rng_state[game])
    rng_state[game] = next_rng
    wanted = next_rng % count
    seen = 0
    for action in range(start, end):
        weight = weights[game, action]
        if weight > 0:
            if seen + weight > wanted:
                return action, wanted - seen
            seen += weight
    return -1, 0


@njit(inline="always")
def _select_random(weights, rng_state, game):
    action, variant = _weighted_action(
        weights,
        rng_state,
        game,
        PLAY_OFFSET,
        PLAY_OFFSET + NUM_CARDS,
    )
    if action >= 0:
        return action, variant
    if weights[game, ATTACK_PLAYER_INDEX] > 0:
        return ATTACK_PLAYER_INDEX, 0
    action, variant = _weighted_action(
        weights, rng_state, game, 0, weights.shape[1]
    )
    if action >= 0:
        return action, variant
    return END_TURN_INDEX, 0


@njit(inline="always")
def _first_target(pending_targets, game, player, group):
    for card in range(pending_targets.shape[4]):
        if pending_targets[game, player, 0, group, card] > 0:
            return card
    return -1


@njit(inline="always")
def _first_pending_action(
    pending_mode,
    pending_sources,
    pending_mandatory,
    pending_targets,
    trade_row,
    game,
    player,
    explorer,
):
    mode = pending_mode[game, player, 0]
    sources = pending_sources[game, player, 0]
    card = -1
    if mode == PENDING_SCRAP:
        if sources & SOURCE_DISCARD:
            card = _first_target(
                pending_targets, game, player, PENDING_TARGET_DISCARD
            )
            if card >= 0:
                return SCRAP_DISCARD_OFFSET + card, 0
        if sources & SOURCE_HAND:
            card = _first_target(
                pending_targets, game, player, PENDING_TARGET_HAND
            )
            if card >= 0:
                return SCRAP_HAND_OFFSET + card, 0
        if sources & SOURCE_TRADE:
            for slot in range(trade_row.shape[1]):
                target = trade_row[game, slot]
                if (
                    target >= 0
                    and pending_targets[
                        game, player, 0, PENDING_TARGET_TRADE, target
                    ]
                    > 0
                ):
                    return SCRAP_TRADE_OFFSET + target, 0
    elif mode == PENDING_DESTROY_BASE:
        card = _first_target(
            pending_targets, game, player, PENDING_TARGET_BASE
        )
        if card >= 0:
            return DESTROY_BASE_OFFSET + card, 0
    elif mode == PENDING_DISCARD:
        card = _first_target(
            pending_targets, game, player, PENDING_TARGET_HAND
        )
        if card >= 0:
            return DISCARD_OFFSET + card, 0
    elif mode == PENDING_ACQUIRE_SHIP_TOP:
        for slot in range(trade_row.shape[1]):
            target = trade_row[game, slot]
            if (
                target >= 0
                and pending_targets[
                    game, player, 0, PENDING_TARGET_HAND, target
                ]
                > 0
            ):
                return BUY_OFFSET + target, 0
        if pending_targets[
            game, player, 0, PENDING_TARGET_HAND, explorer
        ] > 0:
            return BUY_OFFSET + explorer, 0
    elif mode == PENDING_COPY_SHIP:
        card = _first_target(
            pending_targets, game, player, PENDING_TARGET_HAND
        )
        if card >= 0:
            return EFFECT_NONSCRAP_OFFSET + card, 0
    if not pending_mandatory[game, player, 0]:
        return SKIP_INDEX, 0
    return END_TURN_INDEX, 0


@njit(inline="always")
def _preferred_scrap_target(
    pending_targets,
    game,
    player,
    card,
):
    if pending_targets[
        game, player, 0, PENDING_TARGET_DISCARD, card
    ] > 0:
        return SCRAP_DISCARD_OFFSET + card
    if pending_targets[game, player, 0, PENDING_TARGET_HAND, card] > 0:
        return SCRAP_HAND_OFFSET + card
    if pending_targets[game, player, 0, PENDING_TARGET_TRADE, card] > 0:
        return SCRAP_TRADE_OFFSET + card
    return -1


@njit(cache=True)
def _select_heuristic_pending(
    zones,
    trade_row,
    pending_mode,
    pending_sources,
    pending_mandatory,
    pending_targets,
    cost,
    discard_value,
    scout,
    viper,
    explorer,
    game,
    player,
):
    mode = pending_mode[game, player, 0]
    if mode == PENDING_SCRAP:
        deck_size = 0
        for card in range(zones.shape[3]):
            deck_size += zones[game, player, ZONE_HAND, card]
            deck_size += zones[game, player, ZONE_DECK, card]
            deck_size += zones[game, player, ZONE_DISCARD, card]
            deck_size += zones[game, player, ZONE_PLAYED, card]
        if deck_size <= 14:
            priorities = (viper, scout, explorer)
        else:
            priorities = (scout, viper, explorer)
        for card in priorities:
            action = _preferred_scrap_target(
                pending_targets, game, player, card
            )
            if action >= 0:
                return action, 0

        best_card = -1
        best_cost = -1
        for slot in range(trade_row.shape[1]):
            card = trade_row[game, slot]
            if (
                card >= 0
                and pending_targets[
                    game, player, 0, PENDING_TARGET_TRADE, card
                ]
                > 0
                and cost[card] > best_cost
            ):
                best_card = card
                best_cost = cost[card]
        if best_card >= 0 and best_cost >= 5:
            return SCRAP_TRADE_OFFSET + best_card, 0
        if not pending_mandatory[game, player, 0]:
            return SKIP_INDEX, 0
        return _first_pending_action(
            pending_mode,
            pending_sources,
            pending_mandatory,
            pending_targets,
            trade_row,
            game,
            player,
            explorer,
        )

    if mode == PENDING_DISCARD:
        best_card = -1
        best_value = 127
        for card in range(zones.shape[3]):
            if (
                pending_targets[
                    game, player, 0, PENDING_TARGET_HAND, card
                ]
                > 0
                and discard_value[card] < best_value
            ):
                best_card = card
                best_value = discard_value[card]
        if best_card >= 0:
            return DISCARD_OFFSET + best_card, 0

    return _first_pending_action(
        pending_mode,
        pending_sources,
        pending_mandatory,
        pending_targets,
        trade_row,
        game,
        player,
        explorer,
    )


@njit(cache=True)
def _select_effect(
    zones,
    effect_state,
    instance_rule_card,
    instance_faction,
    card_faction,
    card_ally_factions,
    bundle_count,
    bundle_trigger,
    bundle_faction,
    bundle_requirement,
    bundle_source_effect,
    bundle_stage,
    bundle_priority,
    bundle_is_or,
    game,
    player,
    prefer_highest,
):
    best_action = -1
    best_variant = 0
    best_priority = -1
    for card in range(zones.shape[3]):
        nonscrap_variant = 0
        scrap_variant = 0
        for instance in range(effect_state.shape[3]):
            instance_value = effect_state[game, player, card, instance]
            if (instance_value & int(INSTANCE_ACTIVE)) == 0:
                break
            rule_card = instance_rule_card[game, player, card, instance]
            if rule_card < 0:
                rule_card = card
            for bundle in range(bundle_count[rule_card]):
                if (
                    instance_value & _bundle_bit(bundle)
                ) == 0 or not _bundle_eligible(
                    zones,
                    card_faction,
                    card_ally_factions,
                    effect_state,
                    instance_faction,
                    bundle_trigger,
                    bundle_faction,
                    bundle_requirement,
                    game,
                    player,
                    rule_card,
                    bundle,
                ):
                    continue
                scrap = bundle_trigger[rule_card, bundle] == TRIGGER_SCRAP
                group_first = True
                for previous in range(bundle):
                    if (
                        (instance_value & _bundle_bit(previous)) != 0
                        and (
                            bundle_trigger[rule_card, previous]
                            == TRIGGER_SCRAP
                        )
                        == scrap
                        and bundle_source_effect[rule_card, previous]
                        == bundle_source_effect[rule_card, bundle]
                        and bundle_stage[rule_card, previous]
                        == bundle_stage[rule_card, bundle]
                        and _bundle_eligible(
                            zones,
                            card_faction,
                            card_ally_factions,
                            effect_state,
                            instance_faction,
                            bundle_trigger,
                            bundle_faction,
                            bundle_requirement,
                            game,
                            player,
                            rule_card,
                            previous,
                        )
                    ):
                        group_first = False
                        break
                if not group_first:
                    continue
                if scrap:
                    action = EFFECT_SCRAP_OFFSET + card
                    variant = scrap_variant
                    scrap_variant += 1
                else:
                    action = EFFECT_NONSCRAP_OFFSET + card
                    variant = nonscrap_variant
                    nonscrap_variant += 1
                priority = bundle_priority[rule_card, bundle]
                if bundle_is_or[rule_card, bundle]:
                    source = bundle_source_effect[rule_card, bundle]
                    for sibling in range(bundle_count[rule_card]):
                        if (
                            sibling != bundle
                            and bundle_source_effect[rule_card, sibling]
                            == source
                            and (
                                instance_value & _bundle_bit(sibling)
                            )
                            == 0
                        ):
                            priority = 10
                            break
                if best_action < 0 or (
                    prefer_highest and priority > best_priority
                ):
                    best_action = action
                    best_variant = variant
                    best_priority = priority
                    if not prefer_highest:
                        return best_action, best_variant
    return best_action, best_variant


@njit(cache=True)
def _select_simple(
    zones,
    trade_row,
    current_player,
    effect_state,
    instance_rule_card,
    instance_faction,
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
    bundle_source_effect,
    bundle_stage,
    bundle_priority,
    bundle_is_or,
    masks,
    weights,
    explorer,
    game,
):
    player = current_player[game]
    if pending_count[game, player] > 0:
        return _first_pending_action(
            pending_mode,
            pending_sources,
            pending_mandatory,
            pending_targets,
            trade_row,
            game,
            player,
            explorer,
        )
    for card in range(zones.shape[3]):
        if weights[game, PLAY_OFFSET + card] > 0:
            return PLAY_OFFSET + card, 0
    for slot in range(trade_row.shape[1]):
        card = trade_row[game, slot]
        if card >= 0 and masks[game, BUY_OFFSET + card]:
            return BUY_OFFSET + card, 0
    if masks[game, BUY_OFFSET + explorer]:
        return BUY_OFFSET + explorer, 0
    for card in range(zones.shape[3]):
        if masks[game, ATTACK_BASE_OFFSET + card]:
            return ATTACK_BASE_OFFSET + card, 0
    if masks[game, ATTACK_PLAYER_INDEX]:
        return ATTACK_PLAYER_INDEX, 0
    action, variant = _select_effect(
        zones,
        effect_state,
        instance_rule_card,
        instance_faction,
        card_faction,
        card_ally_factions,
        bundle_count,
        bundle_trigger,
        bundle_faction,
        bundle_requirement,
        bundle_source_effect,
        bundle_stage,
        bundle_priority,
        bundle_is_or,
        game,
        player,
        False,
    )
    if action >= 0:
        return action, variant
    return END_TURN_INDEX, 0


@njit(cache=True)
def _select_heuristic(
    zones,
    combat,
    health,
    trade_row,
    current_player,
    effect_state,
    instance_rule_card,
    instance_faction,
    pending_mode,
    pending_sources,
    pending_mandatory,
    pending_targets,
    pending_count,
    cost,
    defense,
    card_faction,
    card_ally_factions,
    bundle_count,
    bundle_trigger,
    bundle_faction,
    bundle_requirement,
    bundle_source_effect,
    bundle_stage,
    masks,
    weights,
    play_priority,
    buy_score_base,
    bundle_priority,
    bundle_is_or,
    discard_value,
    scout,
    viper,
    explorer,
    game,
):
    player = current_player[game]
    opponent = 1 - player
    if pending_count[game, player] > 0:
        return _select_heuristic_pending(
            zones,
            trade_row,
            pending_mode,
            pending_sources,
            pending_mandatory,
            pending_targets,
            cost,
            discard_value,
            scout,
            viper,
            explorer,
            game,
            player,
        )

    best_card = -1
    best_priority = -1
    for card in range(zones.shape[3]):
        if (
            weights[game, PLAY_OFFSET + card] > 0
            and play_priority[card] > best_priority
        ):
            best_card = card
            best_priority = play_priority[card]
    if best_card >= 0:
        return PLAY_OFFSET + best_card, 0

    action, variant = _select_effect(
        zones,
        effect_state,
        instance_rule_card,
        instance_faction,
        card_faction,
        card_ally_factions,
        bundle_count,
        bundle_trigger,
        bundle_faction,
        bundle_requirement,
        bundle_source_effect,
        bundle_stage,
        bundle_priority,
        bundle_is_or,
        game,
        player,
        True,
    )
    if action >= 0:
        return action, variant

    if (
        masks[game, ATTACK_PLAYER_INDEX]
        and combat[game, player] >= health[game, opponent]
    ):
        return ATTACK_PLAYER_INDEX, 0
    best_card = -1
    best_defense = 32767
    for card in range(zones.shape[3]):
        if (
            masks[game, ATTACK_BASE_OFFSET + card]
            and defense[card] < best_defense
        ):
            best_card = card
            best_defense = defense[card]
    if best_card >= 0:
        return ATTACK_BASE_OFFSET + best_card, 0
    if masks[game, ATTACK_PLAYER_INDEX]:
        return ATTACK_PLAYER_INDEX, 0

    faction_counts = np.zeros(16, dtype=np.int16)
    deck_size = 0
    for card in range(zones.shape[3]):
        count = 0
        for zone in (
            ZONE_HAND,
            ZONE_DECK,
            ZONE_DISCARD,
            ZONE_PLAYED,
            ZONE_BASES,
        ):
            count += zones[game, player, zone, card]
        deck_size += (
            zones[game, player, ZONE_HAND, card]
            + zones[game, player, ZONE_DECK, card]
            + zones[game, player, ZONE_DISCARD, card]
            + zones[game, player, ZONE_PLAYED, card]
        )
        faction = card_faction[card]
        if faction > 0:
            faction_counts[faction] += count

    best_action = -1
    best_score = -32768
    for slot in range(trade_row.shape[1] + 1):
        if slot < trade_row.shape[1]:
            card = trade_row[game, slot]
            if card < 0 or not masks[game, BUY_OFFSET + card]:
                continue
        else:
            card = explorer
            if not masks[game, BUY_OFFSET + card]:
                continue
        if card == explorer:
            score = 4 if deck_size <= 14 else 0
        else:
            score = buy_score_base[card]
            faction = card_faction[card]
            if faction > 0:
                score += faction_counts[faction] * 3
        if score > best_score:
            best_score = score
            best_action = BUY_OFFSET + card
    if best_action >= 0 and best_score > 2:
        return best_action, 0
    return END_TURN_INDEX, 0


@njit(parallel=True, cache=True)
def select_fixed_actions_kernel(
    zones,
    combat,
    health,
    trade_row,
    current_player,
    done,
    effect_state,
    instance_rule_card,
    instance_faction,
    pending_mode,
    pending_sources,
    pending_mandatory,
    pending_targets,
    pending_count,
    rng_state,
    cost,
    defense,
    card_faction,
    card_ally_factions,
    bundle_count,
    bundle_trigger,
    bundle_faction,
    bundle_requirement,
    bundle_source_effect,
    bundle_stage,
    masks,
    weights,
    opponent_kind,
    selected_rows,
    play_priority,
    buy_score_base,
    bundle_priority,
    bundle_is_or,
    discard_value,
    scout,
    viper,
    explorer,
    actions,
    action_variants,
):
    for position in prange(selected_rows.shape[0]):
        game = selected_rows[position]
        if done[game]:
            continue
        kind = opponent_kind[game]
        if kind == OPPONENT_RANDOM:
            action, variant = _select_random(weights, rng_state, game)
        elif kind == OPPONENT_SIMPLE:
            action, variant = _select_simple(
                zones,
                trade_row,
                current_player,
                effect_state,
                instance_rule_card,
                instance_faction,
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
                bundle_source_effect,
                bundle_stage,
                bundle_priority,
                bundle_is_or,
                masks,
                weights,
                explorer,
                game,
            )
        elif kind == OPPONENT_HEURISTIC:
            action, variant = _select_heuristic(
                zones,
                combat,
                health,
                trade_row,
                current_player,
                effect_state,
                instance_rule_card,
                instance_faction,
                pending_mode,
                pending_sources,
                pending_mandatory,
                pending_targets,
                pending_count,
                cost,
                defense,
                card_faction,
                card_ally_factions,
                bundle_count,
                bundle_trigger,
                bundle_faction,
                bundle_requirement,
                bundle_source_effect,
                bundle_stage,
                masks,
                weights,
                play_priority,
                buy_score_base,
                bundle_priority,
                bundle_is_or,
                discard_value,
                scout,
                viper,
                explorer,
                game,
            )
        else:
            action, variant = END_TURN_INDEX, 0
        actions[game] = action
        action_variants[game] = variant


def select_fixed_actions(
    engine,
    policy_data: OpponentPolicyData,
    opponent_kind: np.ndarray,
    selected_rows: np.ndarray,
    actions: np.ndarray,
    action_variants: np.ndarray,
) -> None:
    """Select one scalar-policy-equivalent action for each opponent row."""
    state = engine.state
    rules = engine.rules
    weights = engine.build_action_weights()
    select_fixed_actions_kernel(
        state.zones,
        state.combat,
        state.health,
        state.trade_row,
        state.current_player,
        state.done,
        state.effect_state,
        state.instance_rule_card,
        state.instance_faction,
        state.pending_mode,
        state.pending_sources,
        state.pending_mandatory,
        state.pending_targets,
        state.pending_count,
        state.rng_state,
        rules.cost,
        rules.defense,
        rules.faction,
        rules.ally_factions,
        rules.bundle_count,
        rules.bundle_trigger,
        rules.bundle_faction,
        rules.bundle_requirement,
        rules.bundle_source_effect,
        rules.bundle_stage,
        engine.masks,
        weights,
        opponent_kind,
        selected_rows.astype(np.int64, copy=False),
        policy_data.play_priority,
        policy_data.buy_score_base,
        policy_data.bundle_priority,
        policy_data.bundle_is_or,
        policy_data.discard_value,
        policy_data.scout,
        policy_data.viper,
        policy_data.explorer,
        actions,
        action_variants,
    )


@njit(parallel=True, cache=True)
def _select_random_legal(
    weights, rng_state, selected_rows, actions, action_variants
):
    """Compatibility wrapper for the original random-only vector selector."""
    for position in prange(selected_rows.shape[0]):
        game = selected_rows[position]
        action, variant = _select_random(weights, rng_state, game)
        actions[game] = action
        action_variants[game] = variant
