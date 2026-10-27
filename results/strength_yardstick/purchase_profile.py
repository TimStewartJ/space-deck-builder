"""What two checkpoints buy when they play each other greedily.

Plays head-to-head games (deal seed fixed, seats alternating, both
checkpoints choosing their highest-scoring legal action with END_TURN
suppressed while another action is available) on CPU and reports each
checkpoint's win rate and its purchases per game, by card.

Exploratory; not one of decision 0013's pre-registered measures.

Usage, from the repository root:
    python results/strength_yardstick/purchase_profile.py <checkpoint A> <checkpoint B> [games] [deal seed]
"""
from __future__ import annotations

import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.config import DataConfig  # noqa: E402
from src.encoding.action_context import build_action_context  # noqa: E402
from src.encoding.action_encoder import END_TURN_INDEX, get_action_space_size  # noqa: E402
from src.encoding.state_encoder import encode_state  # noqa: E402
from src.engine.actions import ActionType  # noqa: E402
from src.engine.game import Game  # noqa: E402
from src.search.turn_search import PolicyValueEvaluator  # noqa: E402
from src.search.uplift import _load_model, game_seed  # noqa: E402
from src.utils.logger import set_disabled  # noqa: E402


class _External:
    def __init__(self, name):
        self.name = name

    def make_decision(self, state):
        raise RuntimeError


def main(path_a: str, path_b: str, games: int, seed: int) -> None:
    torch.set_num_threads(2)
    set_disabled(True)
    data = DataConfig()
    cards = data.load_cards()
    registry = data.build_registry(cards)
    names = registry.card_names
    dim = get_action_space_size(names)
    evaluators = {
        "A": PolicyValueEvaluator(_load_model(path_a, registry, "cpu"), "cpu"),
        "B": PolicyValueEvaluator(_load_model(path_b, registry, "cpu"), "cpu"),
    }
    slots = []
    for index in range(games):
        game = Game(cards, card_names=names, card_index_map=registry.card_index_map)
        game.rng = random.Random(game_seed(seed, index))
        for name in ("A", "B"):
            game.add_player(name, _External(name))
        game.start_game(first_player_name="A" if index % 2 == 0 else "B")
        slots.append({"game": game, "steps": 0})

    buys = {"A": collections.Counter(), "B": collections.Counter()}
    scraps = collections.Counter()
    wins = collections.Counter()
    turns = []
    active = list(slots)
    while active:
        actions = {}
        for seat in ("A", "B"):
            group = [slot for slot in active if slot["game"].current_player.name == seat]
            if not group:
                continue
            contexts = [build_action_context(s["game"], s["game"].current_player, registry.card_index_map, dim) for s in group]
            observations = np.stack([
                encode_state(s["game"], True, cards=names, card_index_map=registry.card_index_map,
                             can_buy=c.can_buy, has_actions=c.has_meaningful, return_numpy=True)
                for s, c in zip(group, contexts)
            ])
            masks = np.stack([c.mask for c in contexts])
            for row, context in enumerate(contexts):
                if context.has_meaningful:
                    masks[row, END_TURN_INDEX] = False
            logits = evaluators[seat].logits(observations, masks)
            for row, (slot, context) in enumerate(zip(group, contexts)):
                actions[id(slot)] = context.resolvers[int(np.argmax(logits[row]))]
        remaining = []
        for slot in active:
            game = slot["game"]
            seat = game.current_player.name
            action = actions[id(slot)]
            if action.type == ActionType.BUY_CARD and action.card is not None:
                buys[seat][action.card.name] += 1
            elif action.type == ActionType.SCRAP_CARD:
                scraps[seat] += 1
            game.apply_decision(action)
            slot["steps"] += 1
            if game.is_game_over:
                wins[game.get_winner()] += 1
                turns.append(game.stats.total_turns)
            elif slot["steps"] < 20000:
                remaining.append(slot)
        active = remaining

    per_game = lambda counter, card: counter[card] / games
    rows = sorted(
        set(buys["A"]) | set(buys["B"]),
        key=lambda card: -abs(per_game(buys["A"], card) - per_game(buys["B"], card)),
    )
    report = {
        "a": os.path.basename(path_a),
        "b": os.path.basename(path_b),
        "games": games,
        "deal_seed": seed,
        "a_win_rate": wins["A"] / games,
        "mean_turns": float(np.mean(turns)),
        "scraps_per_game": {"a": scraps["A"] / games, "b": scraps["B"] / games},
        "buys_per_game": {
            card: {"a": per_game(buys["A"], card), "b": per_game(buys["B"], card)} for card in rows
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 400,
         int(sys.argv[4]) if len(sys.argv) > 4 else 20261027)
