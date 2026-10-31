"""Step-by-step scalar oracle for the Gate 2 vector engine."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from src.ai.random_agent import RandomAgent
from src.cards.card import Card
from src.config import DataConfig
from src.encoding.action_context import build_action_context
from src.encoding.action_encoder import (
    SKIP_INDEX,
    card_group_of_index,
    get_action_offsets,
    get_action_space_size,
)
from src.encoding.state_encoder import encode_state
from src.engine.actions import Action, ActionType
from src.engine.game import Game
from src.engine.player import Player
from src.utils.logger import set_disabled
from src.vector_engine.kernels import (
    MAX_FORCED_DRAWS,
    VectorEngine,
)
from src.vector_engine.rules import CompiledRules
from src.vector_engine.state import (
    NO_CARD,
    assert_canonical_equal,
    canonical_scalar_snapshot,
    canonical_vector_snapshot,
    from_games,
)


@dataclass(frozen=True)
class DifferentialResult:
    steps: int
    games_completed: int
    action_counts: dict[str, int]
    # Ability actions by "<group>:<card>" (groups effect, effect_alt and
    # scrap), "play_choice:<ship>" for every ship whose play opened a pending
    # decision, and "copy_choice:<ship>" for every copy of a ship that opened
    # one.
    card_action_counts: dict[str, int] = field(default_factory=dict)


def make_supported_game(
    cards: list[Card],
    registry,
    rules: CompiledRules,
    seed: int,
) -> Game:
    """Start a scalar game whose trade deck cannot leave Gate 2's subset."""
    set_disabled(True)
    supported_cards = [
        card for card in cards if bool(rules.supported[card.index])
    ]
    random.seed(seed)
    game = Game(
        supported_cards,
        card_index_map=registry.card_index_map,
    )
    game.add_player("P0", RandomAgent("P0"))
    game.add_player("P1", RandomAgent("P1"))
    game.start_game()
    return game


def _capture_scalar_step(
    game: Game,
    action: Action,
) -> tuple[list[int], list[int]]:
    """Apply one scalar action and capture nondeterministic outputs.

    The vector engine consumes the realized draw/refill card ids, making the
    state transition exact without changing the scalar engine's global RNG.
    """
    drawn: list[int] = []
    refills: list[int] = []
    original_draw = Player.draw_card
    original_fill = Game.fill_trade_row

    def recording_draw(player):
        card = original_draw(player)
        if card is not None:
            drawn.append(int(card.index))
        return card

    def recording_fill(target_game):
        before = len(target_game.trade_row)
        original_fill(target_game)
        refills.extend(
            int(card.index) for card in target_game.trade_row[before:]
        )

    if (
        action.type == ActionType.BUY_CARD
        and action.card is not None
        and action.card in game.trade_row
        and game.trade_deck
    ):
        refills.append(int(game.trade_deck[-1].index))

    Player.draw_card = recording_draw
    Game.fill_trade_row = recording_fill
    try:
        game.apply_decision(action)
    finally:
        Player.draw_card = original_draw
        Game.fill_trade_row = original_fill
    return drawn, refills


# Trace labels of the action layout's card-indexed groups.
_GROUP_LABELS = {
    "play_card": "play",
    "buy_card": "buy",
    "attack_base": "attack_base",
    "destroy_base": "destroy_base",
    "effect_nonscrap": "effect",
    "effect_scrap": "scrap",
    "scrap_hand": "scrap_hand",
    "scrap_discard": "scrap_discard",
    "scrap_trade": "scrap_trade",
    "discard": "discard",
    "effect_alt": "effect_alt",
}


def _action_group(index: int, num_cards: int) -> str:
    if index == get_action_offsets(num_cards).attack_player:
        return "attack_player"
    if index == SKIP_INDEX:
        return "skip"
    slot = card_group_of_index(index, num_cards)
    if slot is None:
        return "end"
    return _GROUP_LABELS[slot[0]]


def choose_differential_action(
    context,
    step: int,
    num_cards: int,
    *,
    pending: bool = False,
) -> tuple[int, Action]:
    """Pick a legal action by a priority order that rotates with *step*.

    Outside a pending decision the class tried first rotates with *step*,
    and the card within the class rotates too, so the trace covers every
    supported action class and every card offering one. Ending the turn is
    chosen only when it is the class tried first or nothing else is legal,
    so a turn usually plays several cards and their ally abilities and
    copies come into play. A pending decision is resolved by a legal action
    picked by a hash of *step*, so its skip and every kind of target are
    covered although the decision always follows the action that opened it.
    """
    legal = np.flatnonzero(context.mask)
    if pending:
        mixed = (step * 2654435761) & 0xFFFFFFFF
        index = int(legal[(mixed >> 8) % len(legal)])
        return index, context.resolvers[index]
    groups = (
        "play",
        "effect",
        "buy",
        "attack_base",
        "attack_player",
        "scrap",
        "effect_alt",
        "scrap_hand",
        "scrap_discard",
        "scrap_trade",
        "destroy_base",
        "discard",
        "skip",
        "end",
    )
    by_group: dict[str, list[int]] = {}
    for index in legal:
        group = _action_group(int(index), num_cards)
        by_group.setdefault(group, []).append(int(index))
    start = step % len(groups)
    order = [
        groups[(start + offset) % len(groups)]
        for offset in range(len(groups))
    ]
    if order[0] != "end":
        order.remove("end")
        order.append("end")
    for wanted in order:
        candidates = by_group.get(wanted)
        if candidates:
            index = candidates[(step // len(groups)) % len(candidates)]
            return index, context.resolvers[index]
    raise AssertionError("scalar context produced no resolvable legal action")


def assert_masks_equal(
    scalar_mask: np.ndarray,
    vector_mask: np.ndarray,
    *,
    game_index: int,
    step: int,
) -> None:
    if np.array_equal(scalar_mask, vector_mask):
        return
    differing = np.flatnonzero(scalar_mask != vector_mask)
    index = int(differing[0])
    raise AssertionError(
        f"mask divergence game={game_index} step={step} action={index}: "
        f"scalar={bool(scalar_mask[index])} "
        f"vector={bool(vector_mask[index])}; "
        f"{len(differing)} bits differ"
    )


def run_differential_trace(
    games: list[Game],
    rules: CompiledRules,
    registry,
    *,
    max_steps: int,
) -> DifferentialResult:
    """Run scalar and vector engines in lockstep until done or max_steps."""
    training = [
        game.players.index(game.current_player) for game in games
    ]
    engine = VectorEngine(
        from_games(games, rules, training_players=training),
        rules,
    )
    action_dim = get_action_space_size(registry.card_names)
    action_counts: dict[str, int] = {}
    card_action_counts: dict[str, int] = {}
    completed = 0
    steps_run = 0

    for step in range(max_steps):
        vector_masks = engine.build_masks().copy()
        vector_observations = engine.materialize_observations().copy()
        actions = np.full(len(games), 0, dtype=np.int32)
        forced_draws = np.full(
            (len(games), MAX_FORCED_DRAWS), NO_CARD, dtype=np.int16
        )
        forced_draw_count = np.zeros(len(games), dtype=np.int8)
        forced_refill = np.full(
            (len(games), 8), NO_CARD, dtype=np.int16
        )
        active = False

        for game_index, game in enumerate(games):
            if game.is_game_over:
                continue
            active = True
            context = build_action_context(
                game,
                game.current_player,
                registry.card_index_map,
                action_dim,
            )
            assert_masks_equal(
                context.mask,
                vector_masks[game_index],
                game_index=game_index,
                step=step,
            )
            training_index = int(engine.state.training_player[game_index])
            current_index = game.players.index(game.current_player)
            scalar_observation = encode_state(
                game,
                is_current_player_training=(
                    current_index == training_index
                ),
                cards=registry.card_names,
                card_index_map=registry.card_index_map,
                can_buy=context.can_buy,
                has_actions=context.has_meaningful,
                return_numpy=True,
            )
            if not np.array_equal(
                scalar_observation, vector_observations[game_index]
            ):
                differing = np.flatnonzero(
                    scalar_observation
                    != vector_observations[game_index]
                )
                column = int(differing[0])
                raise AssertionError(
                    f"observation divergence game={game_index} step={step} "
                    f"column={column}: scalar={scalar_observation[column]!r} "
                    f"vector={vector_observations[game_index, column]!r}"
                )
            action_index, action = choose_differential_action(
                context,
                step + game_index,
                rules.num_cards,
                pending=(
                    game.current_player.get_current_pending_set() is not None
                ),
            )
            actions[game_index] = action_index
            group = _action_group(action_index, rules.num_cards)
            action_counts[group] = action_counts.get(group, 0) + 1
            if group in ("effect", "effect_alt", "scrap") and action.card:
                key = f"{group}:{action.card.name}"
                card_action_counts[key] = card_action_counts.get(key, 0) + 1
            played_ship = (
                action.type == ActionType.PLAY_CARD
                and action.card is not None
                and not action.card.is_base()
            )
            copied_ship = bool(
                action.additional_params
                and action.additional_params.get("copy_source") is not None
            )
            drawn, refills = _capture_scalar_step(game, action)
            if played_ship and game.current_player.get_current_pending_set():
                key = f"play_choice:{action.card.name}"
                card_action_counts[key] = card_action_counts.get(key, 0) + 1
            if copied_ship and game.current_player.get_current_pending_set():
                key = f"copy_choice:{action.card.name}"
                card_action_counts[key] = card_action_counts.get(key, 0) + 1
            if len(drawn) > MAX_FORCED_DRAWS:
                raise AssertionError(
                    f"{len(drawn)} draws exceed tape capacity"
                )
            forced_draws[game_index, : len(drawn)] = drawn
            forced_draw_count[game_index] = len(drawn)
            if len(refills) > forced_refill.shape[1]:
                raise AssertionError(
                    f"{len(refills)} refills exceed tape capacity"
                )
            forced_refill[game_index, : len(refills)] = refills

        if not active:
            break

        engine.step(
            actions,
            forced_draws=forced_draws,
            forced_draw_count=forced_draw_count,
            forced_refill=forced_refill,
        )
        steps_run += 1

        for game_index, game in enumerate(games):
            try:
                assert_canonical_equal(
                    canonical_scalar_snapshot(game, rules),
                    canonical_vector_snapshot(engine.state, game_index),
                )
            except AssertionError as exc:
                raise AssertionError(
                    f"state divergence game={game_index} step={step} "
                    f"action={int(actions[game_index])}: {exc}"
                ) from exc

    completed = sum(game.is_game_over for game in games)
    return DifferentialResult(
        steps=steps_run,
        games_completed=completed,
        action_counts=action_counts,
        card_action_counts=card_action_counts,
    )
