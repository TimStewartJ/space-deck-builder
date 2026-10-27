"""How often a checkpoint uses three ways the engine's rules differ from the rulebook.

The official rulebook (Star Realms Box Set 1.1E, pages 4 and 5) says a ship's
primary ability happens immediately as the ship is played, a base's primary
ability may be used at any time during the main phase, and an OR ability
gives one of its options. The engine differs:

1. An OR ability gives both options: its first option is offered on the
   card's ``effect_nonscrap`` action, and once it is used the same action
   applies the OR parent, which applies the remaining option.
2. A choice in a ship's primary ability (for example Trade Bot's "may scrap")
   is offered as an action usable at any time in the main phase.
3. On the turn Machine Base is played, its draw happens at once and its scrap
   is left as an optional action.

This plays greedy self-play games of a checkpoint on CPU (END_TURN suppressed
while another action is available, as in production) and counts, per card,
OR abilities whose second option was applied after the first, and how many of
those second options had an effect (Recycling Station's discard can be
skipped or impossible, and Blob World can draw nothing); ship choices taken on
the next decision or later in the turn; and Machine Base plays whose scrap was
used or skipped. A ship choice is tracked through the ship whose action the
policy takes, so the choice of a second copy of the same ship, a choice opened
by Stealth Needle's copy, and choices still open when the game ends are not
counted.

Usage, from the repository root:
    python results/strength_yardstick/rules_probe.py <checkpoint> [games] [deal seed]
"""
from __future__ import annotations

import collections
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.config import DataConfig  # noqa: E402
from src.encoding.action_context import build_action_context  # noqa: E402
from src.encoding.action_encoder import (  # noqa: E402
    END_TURN_INDEX,
    get_action_space_layout,
    get_action_space_size,
)
from src.encoding.state_encoder import encode_state  # noqa: E402
from src.engine.actions import ActionType  # noqa: E402
from src.engine.game import Game  # noqa: E402
from src.search.turn_search import PolicyValueEvaluator  # noqa: E402
from src.search.uplift import _load_model, game_seed  # noqa: E402
from src.utils.logger import set_disabled  # noqa: E402

SEATS = ("A", "B")
STEP_LIMIT = 20_000


class _External:
    def __init__(self, name):
        self.name = name

    def make_decision(self, state):
        raise RuntimeError("decisions are made by the probe")


def probe(checkpoint: str, games: int, seed: int) -> dict:
    torch.set_num_threads(2)
    set_disabled(True)
    data = DataConfig()
    cards = data.load_cards()
    registry = data.build_registry(cards)
    names = registry.card_names
    dim = get_action_space_size(names)
    effect_offset = {
        group: offset for group, offset, _ in get_action_space_layout(len(names))["card_groups"]
    }["effect_nonscrap"]
    evaluator = PolicyValueEvaluator(_load_model(checkpoint, registry, "cpu"), "cpu")

    slots = []
    for index in range(games):
        game = Game(cards, card_names=names, card_index_map=registry.card_index_map)
        game.rng = random.Random(game_seed(seed, index))
        for name in SEATS:
            game.add_player(name, _External(name))
        game.start_game(first_player_name=SEATS[index % 2])
        slots.append({"game": game, "index": index, "open_ship_choices": {}, "open_or": None, "steps": 0})

    or_first = collections.Counter()
    or_second = collections.Counter()
    or_second_effective = collections.Counter()
    games_with_or = set()
    games_with_both = set()
    ship_choices = collections.Counter()
    machine_base = collections.Counter()
    active = list(slots)
    while active:
        contexts = [
            build_action_context(slot["game"], slot["game"].current_player, registry.card_index_map, dim)
            for slot in active
        ]
        observations = np.stack([
            encode_state(slot["game"], True, cards=names, card_index_map=registry.card_index_map,
                         can_buy=context.can_buy, has_actions=context.has_meaningful, return_numpy=True)
            for slot, context in zip(active, contexts)
        ])
        masks = np.stack([context.mask for context in contexts])
        for row, context in enumerate(contexts):
            if context.has_meaningful:
                masks[row, END_TURN_INDEX] = False
        logits = evaluator.logits(observations, masks)
        remaining = []
        for row, (slot, context) in enumerate(zip(active, contexts)):
            game = slot["game"]
            player = game.current_player
            action = context.resolvers[int(np.argmax(logits[row]))]
            open_choices = slot["open_ship_choices"]
            for key in open_choices:
                open_choices[key] += 1
            second_option = None
            if action.type == ActionType.APPLY_EFFECT and action.card_effect is not None:
                effect, card = action.card_effect, action.card
                if any(parent.is_or_effect and effect in (parent.child_effects or []) for parent in card.effects):
                    or_first[card.name] += 1
                    games_with_or.add(slot["index"])
                elif effect.is_or_effect:
                    or_second[card.name] += 1
                    second_option = (card.name, _resources(player))
                if id(card) in open_choices and not effect.faction_requirement:
                    age = open_choices.pop(id(card))
                    ship_choices["next decision" if age <= 1 else "later in the turn"] += 1
            if slot["open_or"] is not None and action.type == ActionType.DISCARD_CARDS:
                slot["open_or"]["discards"] += 1
            if action.type == ActionType.END_TURN:
                ship_choices["never used"] += len(open_choices)
                open_choices.clear()
                for base in player.bases:
                    if base.name == "Machine Base" and base in player.played_cards:
                        scrap = base.effects[0].child_effects[1]
                        machine_base["scrap used" if scrap.applied else "scrap skipped"] += 1
            game.apply_decision(action)
            if second_option is not None:
                name, before = second_option
                if player.get_current_pending_set() is not None and _resources(player) == before:
                    # A discard-then-draw option opens a choice; its effect is
                    # known once the choice closes.
                    slot["open_or"] = {"card": name, "discards": 0}
                elif _resources(player) != before:
                    or_second_effective[name] += 1
                    games_with_both.add(slot["index"])
            elif slot["open_or"] is not None and (
                game.current_player is not player or player.get_current_pending_set() is None
            ):
                if slot["open_or"]["discards"]:
                    or_second_effective[slot["open_or"]["card"]] += 1
                    games_with_both.add(slot["index"])
                slot["open_or"] = None
            if (
                action.type == ActionType.PLAY_CARD
                and action.card is not None
                and not action.card.is_base()
                and not game.is_game_over
                and game.current_player is player
            ):
                ship = action.card
                after = build_action_context(game, player, registry.card_index_map, dim)
                index = effect_offset + registry.card_index_map[ship.name]
                if after.mask[index]:
                    resolver = after.resolvers[index]
                    if resolver.card is ship and not resolver.card_effect.faction_requirement:
                        open_choices[id(ship)] = 0
            slot["steps"] += 1
            if slot["steps"] > STEP_LIMIT:
                raise RuntimeError(f"game {slot['index']} exceeded {STEP_LIMIT} decisions")
            if not game.is_game_over:
                remaining.append(slot)
        active = remaining

    return {
        "checkpoint": checkpoint,
        "games": games,
        "deal_seed": seed,
        "games_with_or": len(games_with_or),
        "games_with_both_or_options_effective": len(games_with_both),
        "or_by_card": {
            name: {
                "first_option": or_first[name],
                "second_option_applied": or_second[name],
                "second_option_with_effect": or_second_effective[name],
            }
            for name in sorted(set(or_first) | set(or_second))
        },
        "ship_choices": dict(ship_choices),
        "machine_base": dict(machine_base),
    }


def _resources(player) -> tuple:
    return (player.trade, player.combat, player.health, len(player.hand))


def main(argv: list[str]) -> int:
    import json

    games = int(argv[2]) if len(argv) > 2 else 400
    seed = int(argv[3]) if len(argv) > 3 else 20261027
    print(json.dumps(probe(argv[1], games, seed), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
