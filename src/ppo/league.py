"""Held-out league: checkpoints against scripted opponents no agent trains on.

The protocol is fixed by docs/decisions/0013-strength-yardstick.md. A
checkpoint plays each held-out opponent (:mod:`src.ai.held_out_agents`) at
temperature 0 for 4,000 games. Game ``index`` uses the deal seed
``game_seed(101, index)`` and the checkpoint moves first on even indices, so
it plays exactly half the games in each seat. One process plays one
opponent's games in index order through :class:`BatchedGameRunner` with 256
games in flight; the league score is the pooled win rate over the six
opponents.

Calibration plays each held-out opponent against Heuristic (2,000 games, deal
seed 103, the held-out opponent first on even indices) with no network.

Finished matchups are appended to ``<output>.journal.jsonl``, so an
interrupted league resumes where it stopped.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import random
import time
import traceback
from dataclasses import asdict, dataclass

from src.ai.held_out_agents import HELD_OUT_STYLES, held_out_agent_factory

LEAGUE_GAMES = 4000
LEAGUE_SEED = 101
LEAGUE_BATCH_SIZE = 256
CALIBRATION_GAMES = 2000
CALIBRATION_SEED = 103
CALIBRATION_OPPONENT = "heuristic"
CHECKPOINT_NAME = "Checkpoint"
STEP_LIMIT = 20_000


@dataclass(frozen=True)
class LeagueConfig:
    models: tuple[str, ...] = ()
    opponents: tuple[str, ...] = HELD_OUT_STYLES
    games: int = LEAGUE_GAMES
    seed: int = LEAGUE_SEED
    batch_size: int = LEAGUE_BATCH_SIZE
    calibration: bool = False
    calibration_games: int = CALIBRATION_GAMES
    calibration_seed: int = CALIBRATION_SEED
    workers: int = 8
    device: str = "cuda"

    def __post_init__(self):
        unknown = [name for name in self.opponents if name not in HELD_OUT_STYLES]
        if unknown or not self.opponents:
            raise ValueError(
                f"opponents must be held-out styles ({', '.join(HELD_OUT_STYLES)}), "
                f"got {list(self.opponents)}"
            )
        if not self.models and not self.calibration:
            raise ValueError("give at least one model or ask for calibration")
        for name in ("games", "calibration_games"):
            value = getattr(self, name)
            if value <= 0 or value % 2:
                raise ValueError(f"{name} must be positive and even, got {value}")
        if self.batch_size < 1 or self.workers < 1:
            raise ValueError("batch_size and workers must be positive")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_name(index: int, name_a: str, name_b: str) -> str:
    """Seat *name_a* first on even game indices and *name_b* on odd ones."""
    return name_a if index % 2 == 0 else name_b


_PROCESS_STATE: dict = {}


def _cards_and_registry():
    if "cards" not in _PROCESS_STATE:
        import torch

        from src.config import DataConfig
        from src.utils.logger import set_disabled

        set_disabled(True)
        # Matchups run in parallel processes; one CPU thread each avoids
        # oversubscription. Inference runs on the league's device.
        torch.set_num_threads(1)
        data = DataConfig()
        cards = data.load_cards()
        _PROCESS_STATE["cards"] = cards
        _PROCESS_STATE["registry"] = data.build_registry(cards)
    return _PROCESS_STATE["cards"], _PROCESS_STATE["registry"]


def play_scripted_games(cards, registry, factory_a, factory_b, games: int, seed: int,
                        name_a: str = "A", name_b: str = "B") -> list[dict]:
    """Play scripted agents against each other on the league's deals.

    Every game is built the way :class:`BatchedGameRunner` builds one, so a
    scripted pairing gives the same games as when the runner plays it.
    """
    from src.engine.game import Game
    from src.search.uplift import game_seed

    rows = []
    for index in range(games):
        game = Game(cards, card_names=registry.card_names, card_index_map=registry.card_index_map)
        game.rng = random.Random(game_seed(seed, index))
        game.add_player(name_a, factory_a(name_a))
        game.add_player(name_b, factory_b(name_b))
        game.start_game(first_player_name=first_name(index, name_a, name_b))
        steps = 0
        while not game.is_game_over:
            game.apply_decision(game.current_player.make_decision(game))
            steps += 1
            if steps > STEP_LIMIT:
                raise RuntimeError(f"game {index} exceeded {STEP_LIMIT} decisions")
        rows.append({
            "index": index,
            "a_first": index % 2 == 0,
            "a_won": game.get_winner() == name_a,
            "turns": int(game.stats.total_turns),
        })
    return rows


def play_checkpoint_games(model, cards, registry, device, factory, games: int, seed: int,
                          batch_size: int, opponent_name: str = "Opponent") -> list[dict]:
    """Play a checkpoint at temperature 0 against one scripted opponent."""
    from src.search.batched_games import AgentSeat, BatchedGameRunner, GameSpec, PolicySeat
    from src.search.uplift import game_seed

    runner = BatchedGameRunner(
        cards=cards, registry=registry, model=model, search=None,
        device=device, batch_size=batch_size,
    )
    rows: dict[int, dict] = {}

    def specs():
        for index in range(games):
            yield GameSpec(
                key=index,
                seed=game_seed(seed, index),
                players=((CHECKPOINT_NAME, PolicySeat(0.0)), (opponent_name, AgentSeat(factory))),
                first_player=first_name(index, CHECKPOINT_NAME, opponent_name),
            )

    def finish(outcome):
        rows[outcome.key] = {
            "index": outcome.key,
            "a_first": outcome.key % 2 == 0,
            "a_won": outcome.winner == CHECKPOINT_NAME,
            "turns": outcome.turns,
        }

    runner.run(specs(), finish)
    return [rows[index] for index in range(games)]


def _load_model(path: str, registry, device: str):
    key = ("model", path, device)
    if key not in _PROCESS_STATE:
        from src.search.uplift import _load_model as load

        _PROCESS_STATE[key] = load(path, registry, device)
    return _PROCESS_STATE[key]


def _summarize_rows(rows: list[dict]) -> dict:
    first = [row for row in rows if row["a_first"]]
    return {
        "games": len(rows),
        "wins": sum(row["a_won"] for row in rows),
        "first_seat_games": len(first),
        "first_seat_wins": sum(row["a_won"] for row in first),
        "mean_turns": sum(row["turns"] for row in rows) / len(rows),
    }


def run_task(task: dict) -> dict:
    """Play one matchup and return its journal record."""
    from src.config import DeviceConfig

    cards, registry = _cards_and_registry()
    started = time.perf_counter()
    if task["kind"] == "checkpoint":
        device = DeviceConfig.resolve(task["device"])
        model = _load_model(task["model"], registry, device)
        rows = play_checkpoint_games(
            model, cards, registry, device, held_out_agent_factory(task["opponent"]),
            task["games"], task["seed"], task["batch_size"], opponent_name=task["opponent"],
        )
    elif task["kind"] == "calibration":
        from src.ai.heuristic_agent import HeuristicAgent

        rows = play_scripted_games(
            cards, registry, held_out_agent_factory(task["opponent"]), HeuristicAgent,
            task["games"], task["seed"], name_a=task["opponent"], name_b="Heuristic",
        )
    else:
        raise ValueError(f"unknown league task kind {task['kind']!r}")
    record = {key: value for key, value in task.items() if key != "device"}
    record.update(_summarize_rows(rows))
    record["elapsed_s"] = time.perf_counter() - started
    return record


def _task_key(task: dict) -> tuple:
    return (task["kind"], task.get("sha256"), task["opponent"], task["games"], task["seed"], task.get("batch_size"))


def _worker(task: dict) -> dict:
    try:
        return run_task(task)
    except BaseException:
        return {"error": traceback.format_exc(), "task": task}


def wilson(wins: int, games: int) -> list[float]:
    from src.search.uplift import wilson_interval

    return list(wilson_interval(wins, games))


def summarize(config: LeagueConfig, records: list[dict], models: list[dict]) -> dict:
    """League table per checkpoint plus the optional calibration table."""
    by_key = {_task_key(record): record for record in records}
    checkpoints = []
    for model in models:
        opponents = {}
        wins = games = 0
        for opponent in config.opponents:
            record = by_key.get(("checkpoint", model["sha256"], opponent, config.games, config.seed, config.batch_size))
            if record is None:
                continue
            opponents[opponent] = {
                **{key: record[key] for key in ("wins", "games", "first_seat_games", "first_seat_wins", "mean_turns")},
                "win_rate": record["wins"] / record["games"],
                "win_rate_ci95": wilson(record["wins"], record["games"]),
            }
            wins += record["wins"]
            games += record["games"]
        complete = len(opponents) == len(config.opponents)
        checkpoints.append({
            "model": model["path"],
            "label": model["label"],
            "sha256": model["sha256"],
            "opponents": opponents,
            "pooled": {
                "wins": wins,
                "games": games,
                "win_rate": wins / games if games else None,
                "win_rate_ci95": wilson(wins, games) if games else None,
                "complete": complete,
            },
        })
    calibration = {}
    if config.calibration:
        for opponent in config.opponents:
            record = by_key.get(("calibration", None, opponent, config.calibration_games, config.calibration_seed, None))
            if record is not None:
                calibration[opponent] = {
                    **{key: record[key] for key in ("wins", "games", "first_seat_games", "first_seat_wins", "mean_turns")},
                    "opponent": CALIBRATION_OPPONENT,
                    "win_rate": record["wins"] / record["games"],
                    "win_rate_ci95": wilson(record["wins"], record["games"]),
                }
    return {
        "protocol": "docs/decisions/0013-strength-yardstick.md",
        "config": {key: (list(value) if isinstance(value, tuple) else value) for key, value in asdict(config).items()},
        "checkpoints": checkpoints,
        "calibration": calibration,
    }


def _write_json(path: str, payload: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    os.replace(temporary, path)


def run_league(config: LeagueConfig, output_json: str, log_fn=print) -> dict:
    """Play every missing matchup, journal it, and write the league summary."""
    models = []
    for path in config.models:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"checkpoint not found: {path}")
        models.append({
            "path": os.path.abspath(path),
            "label": os.path.splitext(os.path.basename(path))[0],
            "sha256": file_sha256(path),
        })
    journal = f"{output_json}.journal.jsonl"
    records: list[dict] = []
    if os.path.exists(journal):
        with open(journal, encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
    done = {_task_key(record) for record in records}

    tasks = []
    if config.calibration:
        for opponent in config.opponents:
            tasks.append({
                "kind": "calibration", "opponent": opponent, "sha256": None,
                "games": config.calibration_games, "seed": config.calibration_seed,
                "batch_size": None, "device": "cpu",
            })
    for model in models:
        for opponent in config.opponents:
            tasks.append({
                "kind": "checkpoint", "model": model["path"], "label": model["label"],
                "sha256": model["sha256"], "opponent": opponent, "games": config.games,
                "seed": config.seed, "batch_size": config.batch_size, "device": config.device,
            })
    tasks = [task for task in tasks if _task_key(task) not in done]
    log_fn(
        f"league: {len(models)} checkpoint(s) x {len(config.opponents)} opponent(s), "
        f"calibration={config.calibration}, remaining matchups={len(tasks)}, workers={config.workers}"
    )
    started = time.perf_counter()

    def record(result: dict) -> None:
        if "error" in result:
            raise RuntimeError(f"league task failed: {result['task']}\n{result['error']}")
        records.append(result)
        os.makedirs(os.path.dirname(os.path.abspath(journal)), exist_ok=True)
        with open(journal, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(result) + "\n")
        name = result.get("label", "calibration")
        log_fn(
            f"  {name:>24} vs {result['opponent']:<9} {result['wins']}/{result['games']} "
            f"= {result['wins'] / result['games']:.4f}  ({time.perf_counter() - started:.0f}s)"
        )
        _write_json(output_json, summarize(config, records, models))

    if tasks and config.workers == 1:
        for task in tasks:
            record(_worker(task))
    elif tasks:
        context = mp.get_context("spawn")
        with context.Pool(processes=min(config.workers, len(tasks)), maxtasksperchild=1) as pool:
            for result in pool.imap_unordered(_worker, tasks):
                record(result)
    summary = summarize(config, records, models)
    _write_json(output_json, summary)
    for checkpoint in summary["checkpoints"]:
        pooled = checkpoint["pooled"]
        if pooled["games"]:
            low, high = pooled["win_rate_ci95"]
            log_fn(
                f"{checkpoint['label']}: league {pooled['wins']}/{pooled['games']} = "
                f"{pooled['win_rate']:.4f} (95% CI {low:.4f}-{high:.4f})"
            )
    return summary
