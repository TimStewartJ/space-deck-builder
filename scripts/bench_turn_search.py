"""Benchmark batched turn-bounded search.

``decisions`` mode times :meth:`TurnSearch.choose_many` on realistic search
roots at several batch sizes. ``games`` mode plays complete search-versus-greedy
games through the batched uplift arena. Launch outside the agent process tree,
for example::

    .\\scripts\\launch_training.ps1 -Name sdb_bench_search -Command \\
        "{python} -u scripts/bench_turn_search.py --mode decisions --output logs/bench_search.json"
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.ai.policy_agent import PolicyAgent  # noqa: E402
from src.config import DataConfig, DeviceConfig  # noqa: E402
from src.encoding.action_context import build_action_context  # noqa: E402
from src.encoding.action_encoder import END_TURN_INDEX, get_action_space_size  # noqa: E402
from src.engine.game import Game  # noqa: E402
from src.search.turn_search import SearchTimings, TurnSearch, TurnSearchConfig  # noqa: E402
from src.search.uplift import UpliftConfig, _load_model  # noqa: E402
from src.utils.logger import set_disabled  # noqa: E402
from src.vector_engine.kernels import configure_numba_threads  # noqa: E402
from src.vector_engine.rules import compile_rules  # noqa: E402

DEFAULT_CHECKPOINT = "results/optimized_selfplay_baseline/checkpoints/selfplay_seed0.pth"


def _positions(cards, registry, model, count: int, seed: int) -> list[Game]:
    """Greedy self-play positions at multi-action decisions."""
    action_dim = get_action_space_size(registry.card_names)
    agent = PolicyAgent("P", model, registry.card_names, registry.card_index_map, device="cpu")
    rng = random.Random(seed)
    games = []
    index = 0
    while len(games) < count:
        game = Game(cards, card_names=registry.card_names, card_index_map=registry.card_index_map)
        game.rng = random.Random(seed * 100_003 + index)
        index += 1
        game.add_player("P0", agent)
        game.add_player("P1", agent)
        game.start_game(first_player_name="P0" if index % 2 else "P1")
        target = rng.randint(10, 220)
        steps = 0
        while not game.is_game_over:
            context = build_action_context(game, game.current_player, registry.card_index_map, action_dim)
            mask = context.mask.copy()
            if context.has_meaningful:
                mask[END_TURN_INDEX] = False
            if steps >= target and mask.sum() >= 2:
                games.append(game)
                break
            game.apply_decision(agent.make_decision(game))
            steps += 1
    return games


def bench_decisions(args, cards, registry, rules, device) -> dict:
    cpu_model = _load_model(args.model, registry, "cpu")
    started = time.perf_counter()
    games = _positions(cards, registry, cpu_model, max(args.batch_sizes), args.seed)
    positions_s = time.perf_counter() - started
    model = _load_model(args.model, registry, device)
    search = TurnSearch(
        model, rules, registry.card_index_map, device=device,
        config=TurnSearchConfig(budget=args.budget, max_candidates=args.max_candidates, seed=args.seed),
    )
    search.choose_many(games[: min(8, len(games))])
    rows = []
    for batch_size in args.batch_sizes:
        batch = games[:batch_size]
        repeats = max(1, args.min_decisions // batch_size)
        search.timings = SearchTimings()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        searched = 0
        for _ in range(repeats):
            results = search.choose_many(batch)
            searched += sum(not result.forced for result in results)
        elapsed = time.perf_counter() - started
        timings = search.timings.as_dict()
        decisions = repeats * len(batch)
        row = {
            "batch_size": batch_size,
            "repeats": repeats,
            "decisions": decisions,
            "searched": searched,
            "elapsed_s": elapsed,
            "decisions_per_s": decisions / elapsed,
            "ms_per_searched_decision": 1000 * elapsed / max(1, searched),
            "timings_per_call_ms": {
                key: 1000 * value / repeats
                for key, value in timings.items()
                if key.endswith("_s")
            },
            "rollout_steps_per_call": timings["rollout_steps"] / repeats,
            "rollout_rows_per_step": timings["rollout_rows"] / max(1, timings["rollout_steps"]),
        }
        rows.append(row)
        print(
            f"batch {batch_size:>5}: {row['decisions_per_s']:8.1f} decisions/s, "
            f"{row['ms_per_searched_decision']:7.2f} ms/searched decision, "
            f"{row['rollout_steps_per_call']:.0f} steps/call, "
            f"{row['rollout_rows_per_step']:.0f} rows/step",
            flush=True,
        )
        print("   per call ms: " + ", ".join(
            f"{key[:-2]}={value:.1f}" for key, value in row["timings_per_call_ms"].items()
        ), flush=True)
    return {"positions_s": positions_s, "rows": rows}


def bench_games(args, device) -> dict:
    from src.search.batched_games import RunnerTimings
    from src.search.uplift import _Arena

    rows = []
    for batch_size in args.batch_sizes:
        config = UpliftConfig(
            model_path=args.model, budgets=(args.budget,), games=args.games,
            baseline="greedy", max_candidates=args.max_candidates, seed=args.seed,
            device=device, workers=1, numba_threads=args.numba_threads, batch_size=batch_size,
        )
        arena = _Arena(config)
        arena.runner.timings = RunnerTimings()
        arena.search.timings = SearchTimings()
        played = []
        started = time.perf_counter()
        arena.play_many([(args.budget, index) for index in range(args.games)], played.append)
        elapsed = time.perf_counter() - started
        searched = sum(row["search"]["decisions"] - row["search"]["forced"] for row in played)
        wins = sum(row["search_won"] for row in played)
        row = {
            "batch_size": batch_size,
            "games": len(played),
            "elapsed_s": elapsed,
            "games_per_min": 60 * len(played) / elapsed,
            "searched_decisions": searched,
            "searched_decisions_per_s": searched / elapsed,
            "win_rate": wins / len(played),
            "runner_timings": arena.runner.timings.as_dict(),
            "search_timings": arena.search.timings.as_dict(),
        }
        rows.append(row)
        print(
            f"batch {batch_size:>5}: {row['games_per_min']:.1f} games/min, "
            f"{row['searched_decisions_per_s']:.1f} searched decisions/s "
            f"({elapsed:.0f}s, search won {100 * row['win_rate']:.1f}%)",
            flush=True,
        )
        print("   runner s: " + ", ".join(
            f"{key[:-2]}={value:.1f}" for key, value in row["runner_timings"].items() if key.endswith("_s")
        ), flush=True)
        print("   search s: " + ", ".join(
            f"{key[:-2]}={value:.1f}" for key, value in row["search_timings"].items() if key.endswith("_s")
        ), flush=True)
    return {"rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["decisions", "games"], default="decisions")
    parser.add_argument("--model", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16, 64, 256, 1024])
    parser.add_argument("--min-decisions", type=int, default=256)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--numba-threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    set_disabled(True)
    configure_numba_threads(args.numba_threads)
    device = DeviceConfig.resolve(args.device)
    data = DataConfig()
    cards = data.load_cards()
    registry = data.build_registry(cards)
    rules = compile_rules(cards, registry)
    print(f"mode={args.mode} device={device} numba_threads={args.numba_threads} budget={args.budget}", flush=True)
    if args.mode == "decisions":
        result = bench_decisions(args, cards, registry, rules, device)
    else:
        result = bench_games(args, device)
    payload = {"config": vars(args), "device": device, "result": result}
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)


if __name__ == "__main__":
    main()
