"""Measure how much turn-bounded search improves a PPO policy's play.

Every arm plays the same seeded games with exact seat alternation: game
``index`` seats the searcher first when ``index`` is even. The baseline is the
same checkpoint without search (greedy by default), so budget 0, where the
searcher also plays greedily, is a fairness control. Game RNG streams depend
only on the master seed and game index, so arms are paired by deal.
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import queue as queue_module
import random
import time
import traceback
from dataclasses import asdict, dataclass

SEARCH_NAME = "Search"
BASELINE_NAME = "Baseline"
BASELINES = ("greedy", "sampled", "heuristic")
_STEP_LIMIT = 20_000


@dataclass(frozen=True)
class UpliftConfig:
    model_path: str
    budgets: tuple[int, ...]
    games: int
    baseline: str = "greedy"
    max_candidates: int = 8
    seed: int = 0
    device: str = "cuda"
    workers: int = 1
    numba_threads: int = 2

    def __post_init__(self):
        if not self.budgets:
            raise ValueError("at least one budget is required")
        if len(set(self.budgets)) != len(self.budgets):
            raise ValueError(f"duplicate budgets: {self.budgets}")
        if any(budget < 0 for budget in self.budgets):
            raise ValueError(f"budgets must be non-negative: {self.budgets}")
        if self.games <= 0 or self.games % 2:
            raise ValueError(f"games must be positive and even, got {self.games}")
        if self.baseline not in BASELINES:
            raise ValueError(
                f"baseline must be one of {', '.join(BASELINES)}, got {self.baseline!r}"
            )
        if self.max_candidates < 1 or self.workers < 1 or self.numba_threads < 1:
            raise ValueError("max_candidates, workers, and numba_threads must be positive")


def game_seed(seed: int, index: int) -> int:
    """63-bit game RNG seed shared by every arm for game *index*."""
    digest = hashlib.sha256(f"uplift:{int(seed)}:{int(index)}".encode("ascii"))
    return int.from_bytes(digest.digest()[:8], "little") & ((1 << 63) - 1)


def _load_model(model_path: str, registry, device: str):
    import torch

    from src.config import ModelConfig, load_checkpoint
    from src.encoding.action_encoder import get_action_space_size
    from src.encoding.state_encoder import get_state_size
    from src.ppo.ppo_actor_critic import PPOActorCritic

    names = registry.card_names
    checkpoint = load_checkpoint(model_path, map_location="cpu")
    saved = checkpoint.get("config", {}).get("model")
    model = PPOActorCritic(
        get_state_size(names), get_action_space_size(names), len(names),
        model_config=ModelConfig.from_dict(saved) if saved else ModelConfig(),
        card_registry=registry,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(torch.device(device)).eval()


class _Arena:
    """Per-process cards, rules, policy, and one search per budget."""

    def __init__(self, config: UpliftConfig):
        from src.config import DataConfig, DeviceConfig
        from src.search.turn_search import TurnSearch, TurnSearchConfig
        from src.utils.logger import set_disabled
        from src.vector_engine.kernels import configure_numba_threads
        from src.vector_engine.rules import compile_rules

        set_disabled(True)
        configure_numba_threads(config.numba_threads)
        self.config = config
        self.device = DeviceConfig.resolve(config.device)
        data = DataConfig()
        self.cards = data.load_cards()
        self.registry = data.build_registry(self.cards)
        self.rules = compile_rules(self.cards, self.registry)
        self.model = _load_model(config.model_path, self.registry, self.device)
        self.searches = {
            budget: TurnSearch(
                self.model, self.rules, self.registry.card_index_map,
                device=self.device,
                config=TurnSearchConfig(
                    budget=budget,
                    max_candidates=config.max_candidates,
                    seed=config.seed,
                ),
            )
            for budget in config.budgets
            if budget > 0
        }

    def _policy_agent(self, name: str, temperature: float):
        from src.ai.policy_agent import PolicyAgent

        return PolicyAgent(
            name, self.model, self.registry.card_names,
            self.registry.card_index_map, device=self.device,
            temperature=temperature,
        )

    def play(self, budget: int, index: int) -> dict:
        import torch

        from src.ai.heuristic_agent import HeuristicAgent
        from src.engine.game import Game
        from src.search.agent import SearchAgent

        seed = game_seed(self.config.seed, index)
        torch.manual_seed(seed)
        game = Game(
            self.cards,
            card_names=self.registry.card_names,
            card_index_map=self.registry.card_index_map,
        )
        game.rng = random.Random(seed)
        searcher = (
            SearchAgent(SEARCH_NAME, self.searches[budget])
            if budget > 0
            else self._policy_agent(SEARCH_NAME, 0.0)
        )
        if self.config.baseline == "heuristic":
            baseline = HeuristicAgent(BASELINE_NAME)
        else:
            baseline = self._policy_agent(
                BASELINE_NAME, 0.0 if self.config.baseline == "greedy" else 1.0
            )
        game.add_player(SEARCH_NAME, searcher)
        game.add_player(BASELINE_NAME, baseline)
        search_first = index % 2 == 0
        game.start_game(
            first_player_name=SEARCH_NAME if search_first else BASELINE_NAME
        )
        started = time.perf_counter()
        steps = 0
        while not game.is_game_over:
            action = game.current_player.make_decision(game)
            game.apply_decision(action)
            steps += 1
            if steps > _STEP_LIMIT:
                raise RuntimeError(f"game {index} exceeded {_STEP_LIMIT} decisions")
        statistics = (
            searcher.statistics.to_dict()
            if budget > 0
            else {"decisions": 0, "forced": 0, "deviations": 0, "rollouts": 0, "elapsed_s": 0.0}
        )
        return {
            "budget": budget,
            "index": index,
            "search_first": search_first,
            "search_won": game.get_winner() == SEARCH_NAME,
            "turns": int(game.stats.total_turns),
            "decisions": steps,
            "elapsed_s": time.perf_counter() - started,
            "search": statistics,
        }


def _tasks(config: UpliftConfig, completed: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return [
        (budget, index)
        for index in range(config.games)
        for budget in config.budgets
        if (budget, index) not in completed
    ]


def _worker_main(worker_id: int, config_dict: dict, tasks: list, results) -> None:
    try:
        config = UpliftConfig(**{**config_dict, "budgets": tuple(config_dict["budgets"])})
        arena = _Arena(config)
        for budget, index in tasks:
            results.put(("result", worker_id, arena.play(budget, index)))
        results.put(("done", worker_id, None))
    except BaseException:
        results.put(("error", worker_id, traceback.format_exc()))


def wilson_interval(wins: int, games: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 1.0
    p = wins / games
    denominator = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denominator
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def elo_from_score(score: float) -> float:
    score = min(max(score, 1e-6), 1 - 1e-6)
    return 400.0 * math.log10(score / (1 - score))


def sprt_llr(wins: int, games: int, elo0: float = 0.0, elo1: float = 20.0) -> float:
    """Bernoulli log-likelihood ratio of H1 (elo1) against H0 (elo0)."""
    p0 = 1.0 / (1.0 + 10 ** (-elo0 / 400.0))
    p1 = 1.0 / (1.0 + 10 ** (-elo1 / 400.0))
    losses = games - wins
    return wins * math.log(p1 / p0) + losses * math.log((1 - p1) / (1 - p0))


def summarize(results: list[dict], budgets: tuple[int, ...]) -> list[dict]:
    """Per-budget win rate, Wilson interval, Elo delta, and search cost."""
    summary = []
    for budget in budgets:
        rows = [row for row in results if row["budget"] == budget]
        games = len(rows)
        wins = sum(row["search_won"] for row in rows)
        first = [row for row in rows if row["search_first"]]
        second = [row for row in rows if not row["search_first"]]
        low, high = wilson_interval(wins, games)
        searched = sum(row["search"]["decisions"] - row["search"]["forced"] for row in rows)
        elapsed = sum(row["search"]["elapsed_s"] for row in rows)
        summary.append({
            "budget": budget,
            "games": games,
            "wins": wins,
            "win_rate": wins / games if games else None,
            "win_rate_ci95": [low, high],
            "elo": elo_from_score(wins / games) if games else None,
            "elo_ci95": [elo_from_score(low), elo_from_score(high)] if games else None,
            "sprt_llr_0_20": sprt_llr(wins, games) if games else None,
            "first_seat": {"games": len(first), "wins": sum(r["search_won"] for r in first)},
            "second_seat": {"games": len(second), "wins": sum(r["search_won"] for r in second)},
            "mean_turns": sum(r["turns"] for r in rows) / games if games else None,
            "searched_decisions": searched,
            "forced_decisions": sum(r["search"]["forced"] for r in rows),
            "deviations_from_greedy": sum(r["search"]["deviations"] for r in rows),
            "rollouts": sum(r["search"]["rollouts"] for r in rows),
            "mean_search_ms": 1000.0 * elapsed / searched if searched else None,
        })
    return summary


def _write_json(path: str, payload: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    os.replace(temporary, path)


def run_uplift(config: UpliftConfig, output_json: str | None = None, log_fn=print) -> dict:
    """Run every (budget, game) pairing and return per-game results and a summary.

    When *output_json* is given, completed games are appended to
    ``<output_json>.games.jsonl`` as they finish and a matching journal is
    resumed, so an interrupted run continues where it stopped.
    """
    config_dict = asdict(config)
    # Worker topology does not change outcomes, so a journal can resume with
    # a different process or thread count.
    identity = {
        key: (list(value) if key == "budgets" else value)
        for key, value in config_dict.items()
        if key not in ("workers", "numba_threads")
    }
    journal = f"{output_json}.games.jsonl" if output_json else None
    results: list[dict] = []
    if journal and os.path.exists(journal):
        with open(journal, encoding="utf-8") as fh:
            lines = [json.loads(line) for line in fh if line.strip()]
        if not lines or lines[0].get("config") != identity:
            raise ValueError(f"existing journal {journal} was written for a different configuration")
        results = [line for line in lines[1:]]
    elif journal:
        os.makedirs(os.path.dirname(os.path.abspath(journal)), exist_ok=True)
        with open(journal, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"config": identity}) + "\n")
    completed = {(row["budget"], row["index"]) for row in results}
    tasks = _tasks(config, completed)
    log_fn(
        f"search uplift: budgets={list(config.budgets)} games={config.games} "
        f"baseline={config.baseline} remaining={len(tasks)} workers={config.workers}"
    )
    started = time.perf_counter()

    def record(row: dict) -> None:
        results.append(row)
        if journal:
            with open(journal, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        done = len(results)
        if done % max(1, len(config.budgets) * 10) == 0 or done == config.games * len(config.budgets):
            log_fn(
                f"  {done}/{config.games * len(config.budgets)} games "
                f"({time.perf_counter() - started:.0f}s)"
            )

    if tasks and config.workers == 1:
        arena = _Arena(config)
        for budget, index in tasks:
            record(arena.play(budget, index))
    elif tasks:
        context = mp.get_context("spawn")
        results_queue = context.Queue()
        workers = min(config.workers, len(tasks))
        # Deterministically mix budgets and deals across workers so slow and
        # fast arms progress together and no worker runs a single budget.
        schedule = list(tasks)
        random.Random(config.seed).shuffle(schedule)
        processes = [
            context.Process(
                target=_worker_main,
                args=(worker, config_dict, schedule[worker::workers], results_queue),
            )
            for worker in range(workers)
        ]
        for process in processes:
            process.start()
        finished = 0
        try:
            while finished < workers:
                try:
                    kind, worker, payload = results_queue.get(timeout=30.0)
                except queue_module.Empty:
                    dead = [p for p in processes if p.exitcode not in (None, 0)]
                    if dead:
                        raise RuntimeError(f"uplift worker exited with code {dead[0].exitcode}")
                    continue
                if kind == "result":
                    record(payload)
                elif kind == "done":
                    finished += 1
                else:
                    raise RuntimeError(f"uplift worker {worker} failed:\n{payload}")
        finally:
            for process in processes:
                process.join(timeout=10.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5.0)

    results.sort(key=lambda row: (row["budget"], row["index"]))
    payload = {
        "config": {**config_dict, "budgets": list(config.budgets)},
        "summary": summarize(results, config.budgets),
        "elapsed_s": time.perf_counter() - started,
        "games": results,
    }
    if output_json:
        _write_json(output_json, payload)
    for row in payload["summary"]:
        log_fn(
            f"  budget {row['budget']:>4}: {row['wins']}/{row['games']} "
            f"({100 * row['win_rate']:.1f}%, 95% CI {100 * row['win_rate_ci95'][0]:.1f}-"
            f"{100 * row['win_rate_ci95'][1]:.1f}%, Elo {row['elo']:+.0f}), "
            f"deviations {row['deviations_from_greedy']}, "
            f"mean search {row['mean_search_ms'] or 0:.1f} ms"
        )
    return payload
