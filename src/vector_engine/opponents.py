"""Batched fixed-opponent policies for the vector engine."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numba import njit, prange

from src.ai.heuristic_agent import on_play_trade as heuristic_on_play_trade
from src.cards.effects import CardEffectType
from src.cards.factions import parse_faction
from src.rules.ir import EffectOpcode
from src.vector_engine.kernels import (
    ATTACK_BASE_OFFSET,
    ATTACK_PLAYER_INDEX,
    BUY_OFFSET,
    DESTROY_BASE_OFFSET,
    DISCARD_OFFSET,
    EFFECT_ALT_OFFSET,
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
    _faction_slot,
)
from src.vector_engine.rules import CompiledRules, MAX_BUNDLES, _templates_by_name
from src.vector_engine.state import (
    INSTANCE_ACTIVE,
    PENDING_ACQUIRE_SHIP_TOP,
    PENDING_CHOOSE_OPTION,
    PENDING_COPY_SHIP,
    PENDING_DESTROY_BASE,
    PENDING_DISCARD,
    PENDING_SCRAP,
    PENDING_TARGET_BASE,
    PENDING_TARGET_DISCARD,
    PENDING_TARGET_FIRST_OPTION,
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


# Kinds of an OR option for the Heuristic's option rule
# (HeuristicAgent._choose_or_option).
OPTION_OTHER = 0
OPTION_COMBAT = 1
OPTION_TRADE = 2
OPTION_HEAL = 3
OPTION_DRAW_PER_FACTION = 4
OPTION_DISCARD_DRAW = 5

_OPTION_KIND = {
    CardEffectType.COMBAT: OPTION_COMBAT,
    CardEffectType.TRADE: OPTION_TRADE,
    CardEffectType.HEAL: OPTION_HEAL,
    CardEffectType.DRAW_PER_FACTION: OPTION_DRAW_PER_FACTION,
    CardEffectType.DISCARD_DRAW: OPTION_DISCARD_DRAW,
}


@dataclass(frozen=True)
class OpponentPolicyData:
    play_priority: np.ndarray
    buy_score_base: np.ndarray
    bundle_priority: np.ndarray
    # Per OR option bundle: its kind (OPTION_*), value, and for a
    # draw-per-faction option the faction bits it counts.
    bundle_option_kind: np.ndarray
    bundle_option_value: np.ndarray
    bundle_option_faction: np.ndarray
    # Trade each card adds as soon as it is played (heuristic_agent.on_play_trade).
    on_play_trade: np.ndarray
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
    all_templates = _templates_by_name(cards, registry)

    play_priority = rules.cost.astype(np.int16, copy=True)
    buy_score_base = (rules.cost.astype(np.int16) * 4).astype(np.int16)
    bundle_priority = np.full(
        (rules.num_cards, MAX_BUNDLES), 10, dtype=np.int16
    )
    bundle_option_kind = np.zeros(
        (rules.num_cards, MAX_BUNDLES), dtype=np.int8
    )
    bundle_option_value = np.zeros(
        (rules.num_cards, MAX_BUNDLES), dtype=np.int16
    )
    bundle_option_faction = np.zeros(
        (rules.num_cards, MAX_BUNDLES), dtype=np.int8
    )
    on_play_trade = np.zeros(rules.num_cards, dtype=np.int16)
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
                option = int(rules.bundle_option[card_index, bundle])
                if option:
                    source = source.child_effects[option - 1]
                    bundle_option_kind[card_index, bundle] = _OPTION_KIND.get(
                        source.effect_type, OPTION_OTHER
                    )
                    bundle_option_value[card_index, bundle] = source.value or 0
                    if source.effect_type == CardEffectType.DRAW_PER_FACTION:
                        bundle_option_faction[card_index, bundle] = int(
                            parse_faction(source.faction_target)
                        )
                priority = _effect_priority(source)
            bundle_priority[card_index, bundle] = priority

    for card_index, name in enumerate(rules.card_names):
        card = all_templates.get(name)
        if card is not None:
            on_play_trade[card_index] = heuristic_on_play_trade(card)

    return OpponentPolicyData(
        play_priority=play_priority,
        buy_score_base=buy_score_base,
        bundle_priority=bundle_priority,
        bundle_option_kind=bundle_option_kind,
        bundle_option_value=bundle_option_value,
        bundle_option_faction=bundle_option_faction,
        on_play_trade=on_play_trade,
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
    max_active_snapshots: int = 0,
) -> np.ndarray:
    """Sample stable fixed or snapshot assignments for resident games."""
    if snapshot_count < 0:
        raise ValueError("snapshot_count must be non-negative")
    if max_active_snapshots < 0:
        raise ValueError("max_active_snapshots must be non-negative")
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
        active_snapshots = np.arange(snapshot_count, dtype=np.int16)
        if (
            max_active_snapshots > 0
            and snapshot_count > max_active_snapshots
        ):
            active_snapshots = rng.choice(
                snapshot_count,
                size=max_active_snapshots,
                replace=False,
                p=weights,
            ).astype(np.int16)
        active_weights = weights[active_snapshots]
        active_weights /= active_weights.sum()
        snapshot_choices = rng.choice(
            len(active_snapshots),
            size=snapshot_games,
            p=active_weights,
        )
        assignments[use_snapshot] = (
            SNAPSHOT_OFFSET + active_snapshots[snapshot_choices]
        )
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
    elif mode == PENDING_CHOOSE_OPTION:
        card = _first_target(
            pending_targets, game, player, PENDING_TARGET_FIRST_OPTION
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
    trade,
    combat,
    health,
    trade_row,
    explorer_count,
    faction_plays,
    effect_state,
    instance_rule_card,
    instance_faction,
    pending_mode,
    pending_sources,
    pending_mandatory,
    pending_targets,
    pending_source_card,
    pending_source_instance,
    cost,
    card_type,
    card_faction,
    bundle_count,
    bundle_option,
    bundle_option_kind,
    bundle_option_value,
    bundle_option_faction,
    buy_score_base,
    on_play_trade,
    discard_value,
    scout,
    viper,
    explorer,
    game,
    player,
):
    mode = pending_mode[game, player, 0]
    if mode == PENDING_CHOOSE_OPTION:
        card = pending_source_card[game, player, 0]
        rule_card = instance_rule_card[
            game, player, card, pending_source_instance[game, player, 0]
        ]
        if rule_card < 0:
            rule_card = card
        option = _heuristic_option(
            zones,
            trade,
            combat,
            health,
            trade_row,
            explorer_count,
            faction_plays,
            card_faction,
            card_type,
            cost,
            bundle_count,
            bundle_option,
            bundle_option_kind,
            bundle_option_value,
            bundle_option_faction,
            buy_score_base,
            on_play_trade,
            explorer,
            game,
            player,
            rule_card,
        )
        if option == 2:
            return EFFECT_ALT_OFFSET + card, 0
        return EFFECT_NONSCRAP_OFFSET + card, 0
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


@njit(inline="always")
def _option_of_kind(kind_first, kind_second, kind):
    if kind_first == kind:
        return 1
    if kind_second == kind:
        return 2
    return 0


@njit(cache=True)
def _heuristic_faction_counts(zones, card_faction, game, player):
    """Faction counts of every owned card and the deck size, as HeuristicAgent counts them."""
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
    return faction_counts, deck_size


@njit(inline="always")
def _heuristic_buy_score(
    card,
    explorer,
    deck_size,
    faction_counts,
    card_faction,
    buy_score_base,
):
    """Twice HeuristicAgent._score_card, so that it stays an integer."""
    if card == explorer:
        return 4 if deck_size <= 14 else 0
    score = buy_score_base[card]
    faction = card_faction[card]
    if faction > 0:
        score += faction_counts[faction] * 3
    return score


@njit(cache=True)
def _heuristic_trade_enables_purchase(
    zones,
    trade,
    trade_row,
    explorer_count,
    card_faction,
    cost,
    buy_score_base,
    on_play_trade,
    explorer,
    game,
    player,
    extra,
):
    """HeuristicAgent._trade_enables_purchase."""
    expected = 0
    expected += trade[game, player]
    for card in range(zones.shape[3]):
        expected += zones[game, player, ZONE_HAND, card] * on_play_trade[card]
    faction_counts, deck_size = _heuristic_faction_counts(
        zones, card_faction, game, player
    )
    for slot in range(trade_row.shape[1] + 1):
        if slot < trade_row.shape[1]:
            card = trade_row[game, slot]
            if card < 0:
                continue
        else:
            if explorer_count[game] <= 0:
                continue
            card = explorer
        if (
            expected < cost[card] <= expected + extra
            and _heuristic_buy_score(
                card,
                explorer,
                deck_size,
                faction_counts,
                card_faction,
                buy_score_base,
            )
            > 2
        ):
            return True
    return False


@njit(cache=True)
def _heuristic_option(
    zones,
    trade,
    combat,
    health,
    trade_row,
    explorer_count,
    faction_plays,
    card_faction,
    card_type,
    cost,
    bundle_count,
    bundle_option,
    bundle_option_kind,
    bundle_option_value,
    bundle_option_faction,
    buy_score_base,
    on_play_trade,
    explorer,
    game,
    player,
    rule_card,
):
    """Option (1 or 2) of *rule_card*'s OR ability that HeuristicAgent picks.

    Mirrors HeuristicAgent._choose_or_option.
    """
    first = -1
    second = -1
    for bundle in range(bundle_count[rule_card]):
        if bundle_option[rule_card, bundle] == 1:
            first = bundle
        elif bundle_option[rule_card, bundle] == 2:
            second = bundle
    kind_first = bundle_option_kind[rule_card, first]
    kind_second = bundle_option_kind[rule_card, second]
    opponent = 1 - player

    option = _option_of_kind(kind_first, kind_second, OPTION_COMBAT)
    if option:
        bundle = first if option == 1 else second
        has_outpost = False
        for card in range(zones.shape[3]):
            if (
                zones[game, opponent, ZONE_BASES, card] > 0
                and card_type[card] == 2
            ):
                has_outpost = True
                break
        if (
            not has_outpost
            and combat[game, player] + bundle_option_value[rule_card, bundle]
            >= health[game, opponent]
        ):
            return option
    option = _option_of_kind(kind_first, kind_second, OPTION_TRADE)
    if option:
        bundle = first if option == 1 else second
        if _heuristic_trade_enables_purchase(
            zones,
            trade,
            trade_row,
            explorer_count,
            card_faction,
            cost,
            buy_score_base,
            on_play_trade,
            explorer,
            game,
            player,
            bundle_option_value[rule_card, bundle],
        ):
            return option
    option = _option_of_kind(kind_first, kind_second, OPTION_DRAW_PER_FACTION)
    if option:
        bundle = first if option == 1 else second
        # Every card of the faction played this turn counts, scrapped or not.
        if (
            faction_plays[
                game,
                player,
                _faction_slot(bundle_option_faction[rule_card, bundle]),
            ]
            >= 2
        ):
            return option
    for kind in (OPTION_COMBAT, OPTION_HEAL, OPTION_TRADE):
        option = _option_of_kind(kind_first, kind_second, kind)
        if option:
            return option
    return 1


@njit(cache=True)
def _select_effect(
    zones,
    trade,
    combat,
    health,
    trade_row,
    explorer_count,
    faction_plays,
    effect_state,
    instance_rule_card,
    instance_faction,
    card_faction,
    card_ally_factions,
    card_type,
    cost,
    bundle_count,
    bundle_trigger,
    bundle_faction,
    bundle_requirement,
    bundle_source_effect,
    bundle_option,
    bundle_priority,
    bundle_option_kind,
    bundle_option_value,
    bundle_option_faction,
    buy_score_base,
    on_play_trade,
    explorer,
    game,
    player,
    prefer_highest,
):
    """First available ability (Simple) or the highest-priority one (Heuristic).

    The Heuristic considers only the OR option HeuristicAgent picks; Simple
    reaches an OR ability's first option before its second.
    """
    best_action = -1
    best_variant = 0
    best_priority = -1
    for card in range(zones.shape[3]):
        nonscrap_variant = 0
        scrap_variant = 0
        alt_variant = 0
        for instance in range(effect_state.shape[3]):
            instance_value = effect_state[game, player, card, instance]
            if (instance_value & int(INSTANCE_ACTIVE)) == 0:
                break
            rule_card = instance_rule_card[game, player, card, instance]
            if rule_card < 0:
                rule_card = card
            chosen_option = 0
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
                option = bundle_option[rule_card, bundle]
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
                        and bundle_option[rule_card, previous] == option
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
                elif option == 2:
                    action = EFFECT_ALT_OFFSET + card
                    variant = alt_variant
                    alt_variant += 1
                else:
                    action = EFFECT_NONSCRAP_OFFSET + card
                    variant = nonscrap_variant
                    nonscrap_variant += 1
                if option != 0 and prefer_highest:
                    if chosen_option == 0:
                        chosen_option = _heuristic_option(
                            zones,
                            trade,
                            combat,
                            health,
                            trade_row,
                            explorer_count,
                            faction_plays,
                            card_faction,
                            card_type,
                            cost,
                            bundle_count,
                            bundle_option,
                            bundle_option_kind,
                            bundle_option_value,
                            bundle_option_faction,
                            buy_score_base,
                            on_play_trade,
                            explorer,
                            game,
                            player,
                            rule_card,
                        )
                    if option != chosen_option:
                        continue
                priority = bundle_priority[rule_card, bundle]
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
    trade,
    combat,
    health,
    trade_row,
    explorer_count,
    faction_plays,
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
    card_type,
    card_faction,
    card_ally_factions,
    bundle_count,
    bundle_trigger,
    bundle_faction,
    bundle_requirement,
    bundle_source_effect,
    bundle_option,
    bundle_priority,
    bundle_option_kind,
    bundle_option_value,
    bundle_option_faction,
    buy_score_base,
    on_play_trade,
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
        trade,
        combat,
        health,
        trade_row,
        explorer_count,
        faction_plays,
        effect_state,
        instance_rule_card,
        instance_faction,
        card_faction,
        card_ally_factions,
        card_type,
        cost,
        bundle_count,
        bundle_trigger,
        bundle_faction,
        bundle_requirement,
        bundle_source_effect,
        bundle_option,
        bundle_priority,
        bundle_option_kind,
        bundle_option_value,
        bundle_option_faction,
        buy_score_base,
        on_play_trade,
        explorer,
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
    trade,
    combat,
    health,
    trade_row,
    explorer_count,
    faction_plays,
    current_player,
    effect_state,
    instance_rule_card,
    instance_faction,
    pending_mode,
    pending_sources,
    pending_mandatory,
    pending_targets,
    pending_source_card,
    pending_source_instance,
    pending_count,
    cost,
    defense,
    card_type,
    card_faction,
    card_ally_factions,
    bundle_count,
    bundle_trigger,
    bundle_faction,
    bundle_requirement,
    bundle_source_effect,
    bundle_option,
    masks,
    weights,
    play_priority,
    buy_score_base,
    bundle_priority,
    bundle_option_kind,
    bundle_option_value,
    bundle_option_faction,
    on_play_trade,
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
            trade,
            combat,
            health,
            trade_row,
            explorer_count,
            faction_plays,
            effect_state,
            instance_rule_card,
            instance_faction,
            pending_mode,
            pending_sources,
            pending_mandatory,
            pending_targets,
            pending_source_card,
            pending_source_instance,
            cost,
            card_type,
            card_faction,
            bundle_count,
            bundle_option,
            bundle_option_kind,
            bundle_option_value,
            bundle_option_faction,
            buy_score_base,
            on_play_trade,
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
        trade,
        combat,
        health,
        trade_row,
        explorer_count,
        faction_plays,
        effect_state,
        instance_rule_card,
        instance_faction,
        card_faction,
        card_ally_factions,
        card_type,
        cost,
        bundle_count,
        bundle_trigger,
        bundle_faction,
        bundle_requirement,
        bundle_source_effect,
        bundle_option,
        bundle_priority,
        bundle_option_kind,
        bundle_option_value,
        bundle_option_faction,
        buy_score_base,
        on_play_trade,
        explorer,
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

    faction_counts, deck_size = _heuristic_faction_counts(
        zones, card_faction, game, player
    )
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
        score = _heuristic_buy_score(
            card,
            explorer,
            deck_size,
            faction_counts,
            card_faction,
            buy_score_base,
        )
        if score > best_score:
            best_score = score
            best_action = BUY_OFFSET + card
    if best_action >= 0 and best_score > 2:
        return best_action, 0
    return END_TURN_INDEX, 0


@njit(parallel=True, cache=True)
def select_fixed_actions_kernel(
    zones,
    trade,
    combat,
    health,
    trade_row,
    explorer_count,
    faction_plays,
    current_player,
    done,
    effect_state,
    instance_rule_card,
    instance_faction,
    pending_mode,
    pending_sources,
    pending_mandatory,
    pending_targets,
    pending_source_card,
    pending_source_instance,
    pending_count,
    rng_state,
    cost,
    defense,
    card_type,
    card_faction,
    card_ally_factions,
    bundle_count,
    bundle_trigger,
    bundle_faction,
    bundle_requirement,
    bundle_source_effect,
    bundle_option,
    masks,
    weights,
    opponent_kind,
    selected_rows,
    play_priority,
    buy_score_base,
    bundle_priority,
    bundle_option_kind,
    bundle_option_value,
    bundle_option_faction,
    on_play_trade,
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
                trade,
                combat,
                health,
                trade_row,
                explorer_count,
                faction_plays,
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
                card_type,
                card_faction,
                card_ally_factions,
                bundle_count,
                bundle_trigger,
                bundle_faction,
                bundle_requirement,
                bundle_source_effect,
                bundle_option,
                bundle_priority,
                bundle_option_kind,
                bundle_option_value,
                bundle_option_faction,
                buy_score_base,
                on_play_trade,
                masks,
                weights,
                explorer,
                game,
            )
        elif kind == OPPONENT_HEURISTIC:
            action, variant = _select_heuristic(
                zones,
                trade,
                combat,
                health,
                trade_row,
                explorer_count,
                faction_plays,
                current_player,
                effect_state,
                instance_rule_card,
                instance_faction,
                pending_mode,
                pending_sources,
                pending_mandatory,
                pending_targets,
                pending_source_card,
                pending_source_instance,
                pending_count,
                cost,
                defense,
                card_type,
                card_faction,
                card_ally_factions,
                bundle_count,
                bundle_trigger,
                bundle_faction,
                bundle_requirement,
                bundle_source_effect,
                bundle_option,
                masks,
                weights,
                play_priority,
                buy_score_base,
                bundle_priority,
                bundle_option_kind,
                bundle_option_value,
                bundle_option_faction,
                on_play_trade,
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
        state.trade,
        state.combat,
        state.health,
        state.trade_row,
        state.explorer_count,
        state.faction_plays,
        state.current_player,
        state.done,
        state.effect_state,
        state.instance_rule_card,
        state.instance_faction,
        state.pending_mode,
        state.pending_sources,
        state.pending_mandatory,
        state.pending_targets,
        state.pending_source_card,
        state.pending_source_instance,
        state.pending_count,
        state.rng_state,
        rules.cost,
        rules.defense,
        rules.card_type,
        rules.faction,
        rules.ally_factions,
        rules.bundle_count,
        rules.bundle_trigger,
        rules.bundle_faction,
        rules.bundle_requirement,
        rules.bundle_source_effect,
        rules.bundle_option,
        engine.masks,
        weights,
        opponent_kind,
        selected_rows.astype(np.int64, copy=False),
        policy_data.play_priority,
        policy_data.buy_score_base,
        policy_data.bundle_priority,
        policy_data.bundle_option_kind,
        policy_data.bundle_option_value,
        policy_data.bundle_option_faction,
        policy_data.on_play_trade,
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
