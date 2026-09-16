"""Evaluate top-two candidate values on fixed positions with one implementation.

Usage: python compare_search_values.py <repo> <output.json> [positions]

Run once against each repository and compare the JSON files. Positions come
from greedy self-play with fixed seeds, so both runs search identical games.
"""
import json
import random
import sys

REPO, OUTPUT = sys.argv[1], sys.argv[2]
POSITIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 150
sys.path.insert(0, REPO)

import os  # noqa: E402

os.chdir(REPO)

import torch  # noqa: E402

from src.ai.policy_agent import PolicyAgent  # noqa: E402
from src.config import DataConfig, ModelConfig, load_checkpoint  # noqa: E402
from src.encoding.action_context import build_action_context  # noqa: E402
from src.encoding.action_encoder import END_TURN_INDEX, get_action_space_size  # noqa: E402
from src.encoding.state_encoder import get_state_size  # noqa: E402
from src.engine.game import Game  # noqa: E402
from src.ppo.ppo_actor_critic import PPOActorCritic  # noqa: E402
from src.search.turn_search import TurnSearch, TurnSearchConfig  # noqa: E402
from src.utils.logger import set_disabled  # noqa: E402
from src.vector_engine.kernels import configure_numba_threads  # noqa: E402
from src.vector_engine.rules import compile_rules  # noqa: E402

set_disabled(True)
configure_numba_threads(4)
torch.set_num_threads(4)
data = DataConfig()
cards = data.load_cards()
registry = data.build_registry(cards)
rules = compile_rules(cards, registry)
names = registry.card_names
checkpoint = load_checkpoint("results/optimized_selfplay_baseline/checkpoints/selfplay_seed0.pth", map_location="cpu")
model = PPOActorCritic(
    get_state_size(names), get_action_space_size(names), len(names),
    model_config=ModelConfig.from_dict(checkpoint["config"]["model"]), card_registry=registry,
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
agent = PolicyAgent("P", model, names, registry.card_index_map, device="cpu")
action_dim = get_action_space_size(names)
search = TurnSearch(
    model, rules, registry.card_index_map, device="cpu",
    config=TurnSearchConfig(budget=256, max_candidates=2, seed=3),
)

records = []
index = 0
while len(records) < POSITIONS:
    game = Game(cards, card_names=names, card_index_map=registry.card_index_map)
    game.rng = random.Random(5000 + index)
    game.add_player("P0", agent)
    game.add_player("P1", agent)
    game.start_game(first_player_name="P0" if index % 2 == 0 else "P1")
    target = random.Random(index).randint(10, 200)
    index += 1
    steps = 0
    while not game.is_game_over:
        context = build_action_context(game, game.current_player, registry.card_index_map, action_dim)
        mask = context.mask.copy()
        if context.has_meaningful:
            mask[END_TURN_INDEX] = False
        if steps >= target and mask.sum() >= 2:
            result = search.choose(game)
            records.append({
                "position": index - 1,
                "forced": result.forced,
                "candidates": [[c.action, c.rollouts, c.value] for c in result.candidates],
            })
            break
        game.apply_decision(agent.make_decision(game))
        steps += 1

with open(OUTPUT, "w", encoding="utf-8") as fh:
    json.dump(records, fh)
print(f"wrote {len(records)} positions to {OUTPUT}")
