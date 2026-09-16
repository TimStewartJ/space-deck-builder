"""Record search self-play games as training shards.

Both seats of every game are the same checkpoint searching at a fixed rollout
budget. Every searched decision (forced decisions included) becomes one
training position:

- ``observations``: the acting player's policy input;
- ``masks``: legal actions with production END_TURN suppression, stored as
  packed bits;
- ``policy``: the completed-Q improved policy from :class:`SearchTargets`;
- ``outcome``: +1 if the acting player won the game, otherwise −1;
- ``root_value``: the critic estimate at the root;
- ``action``, ``greedy``, ``forced``: the searched move, the policy's greedy
  move, and whether only one action was legal;
- ``game``, ``step``, ``turn``, ``player``: position provenance.

Shards contain complete games only and are written atomically, so an
interrupted generation resumes by replaying the games missing from every
shard. Game deals depend only on the master seed and game index.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import queue as queue_module
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

SHARD_FORMAT = "search-selfplay-v1"
SEAT_NAMES = ("A", "B")


@dataclass(frozen=True)
class SelfPlayConfig:
    model_path: str
    games: int
    budget: int = 64
    max_candidates: int = 8
    seed: int = 0
    device: str = "cuda"
    workers: int = 2
    numba_threads: int = 4
    batch_size: int = 1024
    shard_positions: int = 65536

    def __post_init__(self):
        if self.games <= 0:
            raise ValueError(f"games must be positive, got {self.games}")
        for name in ("budget", "max_candidates", "workers", "numba_threads", "batch_size", "shard_positions"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")


def selfplay_game_seed(seed: int, index: int) -> int:
    """63-bit deal seed for self-play game *index*."""
    digest = hashlib.sha256(f"selfplay:{int(seed)}:{int(index)}".encode("ascii"))
    return int.from_bytes(digest.digest()[:8], "little") & ((1 << 63) - 1)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class _Record:
    player: int
    step: int
    turn: int
    action: int
    greedy: int
    forced: bool
    observation: np.ndarray
    mask: np.ndarray
    policy: np.ndarray
    root_value: float


class ShardWriter:
    """Buffer complete games and write them as compressed shards."""

    def __init__(self, directory: Path, prefix: str, shard_positions: int):
        self.directory = Path(directory)
        self.prefix = prefix
        self.shard_positions = shard_positions
        self._games: list[tuple[int, list[_Record], np.ndarray]] = []
        self._positions = 0
        self._shards = 0
        self.written_games = 0
        self.written_positions = 0

    def add_game(self, game: int, records: list[_Record], winner_seat: int) -> None:
        outcomes = np.asarray(
            [1.0 if record.player == winner_seat else -1.0 for record in records],
            dtype=np.float32,
        )
        self._games.append((game, records, outcomes))
        self._positions += len(records)
        if self._positions >= self.shard_positions:
            self.flush()

    def flush(self) -> None:
        if not self._games:
            return
        records = [record for _, game_records, _ in self._games for record in game_records]
        arrays = {
            "format": np.asarray(SHARD_FORMAT),
            "games": np.asarray([game for game, _, _ in self._games], dtype=np.int64),
            "game": np.concatenate([
                np.full(len(game_records), game, dtype=np.int64)
                for game, game_records, _ in self._games
            ]),
            "outcome": np.concatenate([outcomes for _, _, outcomes in self._games]),
            "observations": np.stack([record.observation for record in records]).astype(np.float32),
            "mask_bits": np.packbits(np.stack([record.mask for record in records]).astype(np.bool_), axis=1),
            "action_dim": np.asarray(len(records[0].mask), dtype=np.int64),
            "policy": np.stack([record.policy for record in records]).astype(np.float32),
            "root_value": np.asarray([record.root_value for record in records], dtype=np.float32),
            "action": np.asarray([record.action for record in records], dtype=np.int16),
            "greedy": np.asarray([record.greedy for record in records], dtype=np.int16),
            "forced": np.asarray([record.forced for record in records], dtype=np.bool_),
            "step": np.asarray([record.step for record in records], dtype=np.int32),
            "turn": np.asarray([record.turn for record in records], dtype=np.int32),
            "player": np.asarray([record.player for record in records], dtype=np.uint8),
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        final = self.directory / f"shard_{self.prefix}_{self._shards:05d}.npz"
        temporary = final.with_name(final.stem + ".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, final)
        self._shards += 1
        self.written_games += len(self._games)
        self.written_positions += len(records)
        self._games = []
        self._positions = 0


class GameRecorder:
    """Collect searched decisions per game and hand finished games to a writer."""

    def __init__(self, writer: ShardWriter):
        self.writer = writer
        self._pending: dict[int, list[_Record]] = {}

    def on_search(self, decision) -> None:
        targets = decision.result.targets
        if targets is None:
            raise ValueError("self-play recording requires search targets")
        self._pending.setdefault(decision.key, []).append(_Record(
            player=SEAT_NAMES.index(decision.player),
            step=decision.step,
            turn=decision.turn,
            action=decision.result.action,
            greedy=decision.result.greedy_action,
            forced=decision.result.forced,
            observation=np.array(targets.observation, dtype=np.float32, copy=True),
            mask=np.array(targets.mask, dtype=np.bool_, copy=True),
            policy=np.array(targets.policy, dtype=np.float32, copy=True),
            root_value=targets.root_value,
        ))

    def on_finish(self, outcome) -> None:
        records = self._pending.pop(outcome.key, [])
        if not records:
            raise RuntimeError(f"game {outcome.key!r} finished without searched decisions")
        self.writer.add_game(outcome.key, records, SEAT_NAMES.index(outcome.winner))


def finished_shards(directory: Path) -> list[Path]:
    """Complete shard files in *directory*, excluding interrupted temporary writes."""
    return [
        path for path in sorted(Path(directory).glob("shard_*.npz"))
        if not path.name.endswith(".tmp.npz")
    ]


def completed_games(directory: Path) -> set[int]:
    games: set[int] = set()
    for path in finished_shards(directory):
        with np.load(path) as shard:
            if str(shard["format"]) != SHARD_FORMAT:
                raise ValueError(f"{path} is not a {SHARD_FORMAT} shard")
            contained = set(int(game) for game in shard["games"])
        overlap = games & contained
        if overlap:
            raise ValueError(f"{path} repeats games already recorded: {sorted(overlap)[:5]}")
        games |= contained
    return games


@dataclass
class SelfPlayData:
    observations: np.ndarray
    masks: np.ndarray
    policy: np.ndarray
    outcome: np.ndarray
    root_value: np.ndarray
    action: np.ndarray
    greedy: np.ndarray
    forced: np.ndarray
    game: np.ndarray
    step: np.ndarray
    turn: np.ndarray
    player: np.ndarray

    @property
    def positions(self) -> int:
        return len(self.outcome)


def load_selfplay_data(directory: str | Path) -> SelfPlayData:
    """Load and validate every shard in *directory*."""
    paths = finished_shards(Path(directory))
    if not paths:
        raise FileNotFoundError(f"no self-play shards in {directory}")
    parts: dict[str, list[np.ndarray]] = {name: [] for name in SelfPlayData.__dataclass_fields__}
    seen: set[int] = set()
    action_dim = None
    for path in paths:
        with np.load(path) as shard:
            if str(shard["format"]) != SHARD_FORMAT:
                raise ValueError(f"{path} is not a {SHARD_FORMAT} shard")
            dim = int(shard["action_dim"])
            if action_dim is not None and dim != action_dim:
                raise ValueError("shards disagree on action dimension")
            action_dim = dim
            games = set(int(game) for game in shard["games"])
            if games & seen or set(int(game) for game in np.unique(shard["game"])) != games:
                raise ValueError(f"{path} has inconsistent or repeated games")
            seen |= games
            parts["masks"].append(np.unpackbits(shard["mask_bits"], axis=1, count=dim).astype(np.bool_))
            for name in parts:
                if name != "masks":
                    parts[name].append(shard[name])
    data = SelfPlayData(**{name: np.concatenate(values) for name, values in parts.items()})
    rows = np.arange(data.positions)
    if not data.masks[rows, data.action].all():
        raise ValueError("recorded actions must be legal")
    if (data.policy[~data.masks] != 0).any():
        raise ValueError("improved policies must be zero on illegal actions")
    return data


class _Generator:
    """One process's model, search, runner, and recorder."""

    def __init__(self, config: SelfPlayConfig, directory: Path, prefix: str, model=None):
        from src.config import DataConfig, DeviceConfig
        from src.search.batched_games import BatchedGameRunner
        from src.search.turn_search import TurnSearch, TurnSearchConfig
        from src.search.uplift import _load_model
        from src.utils.logger import set_disabled
        from src.vector_engine.kernels import configure_numba_threads
        from src.vector_engine.rules import compile_rules

        set_disabled(True)
        configure_numba_threads(config.numba_threads)
        self.config = config
        device = DeviceConfig.resolve(config.device)
        data = DataConfig()
        cards = data.load_cards()
        registry = data.build_registry(cards)
        rules = compile_rules(cards, registry)
        model = model if model is not None else _load_model(config.model_path, registry, device)
        search = TurnSearch(
            model, rules, registry.card_index_map, device=device,
            config=TurnSearchConfig(
                budget=config.budget, max_candidates=config.max_candidates, seed=config.seed,
            ),
        )
        self.runner = BatchedGameRunner(
            cards=cards, registry=registry, model=model, search=search,
            device=device, batch_size=config.batch_size,
        )
        self.writer = ShardWriter(directory, prefix, config.shard_positions)
        self.recorder = GameRecorder(self.writer)

    def play(self, games: list[int], on_game: Callable[[int, int], None]) -> None:
        from src.search.batched_games import GameSpec, SearchSeat

        seat = SearchSeat(self.config.budget)
        specs = (
            GameSpec(
                key=index,
                seed=selfplay_game_seed(self.config.seed, index),
                players=((SEAT_NAMES[0], seat), (SEAT_NAMES[1], seat)),
                first_player=SEAT_NAMES[index % 2],
            )
            for index in games
        )

        def finish(outcome) -> None:
            positions = len(self.recorder._pending.get(outcome.key, []))
            self.recorder.on_finish(outcome)
            on_game(outcome.key, positions)

        try:
            self.runner.run(specs, finish, on_search=self.recorder.on_search)
        finally:
            self.writer.flush()


def _worker_main(worker_id: int, config_dict: dict, games: list, directory: str, prefix: str, results) -> None:
    try:
        generator = _Generator(SelfPlayConfig(**config_dict), Path(directory), prefix)
        generator.play(games, lambda game, positions: results.put(("game", worker_id, positions)))
        results.put(("done", worker_id, (generator.writer.written_games, generator.writer.written_positions)))
    except BaseException:
        results.put(("error", worker_id, traceback.format_exc()))


def _identity(config: SelfPlayConfig) -> dict:
    from src.search.turn_search import SEARCH_IMPLEMENTATION

    # Deals depend only on the seed and game index, so a directory can be
    # extended to more games; topology settings only change speed.
    identity = {
        key: value for key, value in asdict(config).items()
        if key not in ("games", "workers", "numba_threads", "batch_size", "shard_positions")
    }
    identity["model_sha256"] = file_sha256(config.model_path)
    identity["search_implementation"] = SEARCH_IMPLEMENTATION
    identity["shard_format"] = SHARD_FORMAT
    return identity


def generate_selfplay(config: SelfPlayConfig, output_dir: str | Path, log_fn=print, model=None) -> dict:
    """Generate missing self-play games into *output_dir* and return a summary."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    identity = _identity(config)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("identity") != identity:
            raise ValueError(f"{manifest_path} was written for a different configuration")
    else:
        manifest_path.write_text(json.dumps({"identity": identity}, indent=2) + "\n", encoding="utf-8")

    # A process stopped while writing leaves a temporary shard; its games are
    # missing from finished shards and are replayed.
    for stale in directory.glob("shard_*.tmp.npz"):
        stale.unlink()
    done = completed_games(directory)
    unexpected = {game for game in done if not 0 <= game < config.games}
    if unexpected:
        raise ValueError(
            f"shards contain {len(unexpected)} games outside 0..{config.games - 1}; "
            "use at least the recorded number of games"
        )
    tasks = [index for index in range(config.games) if index not in done]
    log_fn(f"search self-play: games={config.games} budget={config.budget} remaining={len(tasks)} workers={config.workers}")
    started = time.perf_counter()
    prefix = f"{time.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"
    progress = {"games": 0, "positions": 0}

    def on_game(game: int, positions: int) -> None:
        progress["games"] += 1
        progress["positions"] += positions
        if progress["games"] % 256 == 0 or progress["games"] == len(tasks):
            elapsed = time.perf_counter() - started
            log_fn(
                f"  {progress['games']}/{len(tasks)} games, {progress['positions']} positions "
                f"({elapsed:.0f}s, {60 * progress['games'] / max(elapsed, 1e-9):.0f} games/min)"
            )

    if tasks and config.workers == 1:
        _Generator(config, directory, f"{prefix}_w00", model=model).play(tasks, on_game)
    elif tasks:
        context = mp.get_context("spawn")
        results = context.Queue()
        workers = min(config.workers, len(tasks))
        processes = [
            context.Process(
                target=_worker_main,
                args=(worker, asdict(config), tasks[worker::workers], str(directory), f"{prefix}_w{worker:02d}", results),
            )
            for worker in range(workers)
        ]
        for process in processes:
            process.start()
        finished = 0
        try:
            while finished < workers:
                try:
                    kind, worker, payload = results.get(timeout=30.0)
                except queue_module.Empty:
                    dead = [process for process in processes if process.exitcode not in (None, 0)]
                    if dead:
                        raise RuntimeError(f"self-play worker exited with code {dead[0].exitcode}")
                    continue
                if kind == "game":
                    on_game(-1, payload)
                elif kind == "done":
                    finished += 1
                else:
                    raise RuntimeError(f"self-play worker {worker} failed:\n{payload}")
        finally:
            for process in processes:
                process.join(timeout=30.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5.0)

    recorded = completed_games(directory)
    missing = [index for index in range(config.games) if index not in recorded]
    if missing:
        raise RuntimeError(f"{len(missing)} games are missing after generation, first {missing[:5]}")
    positions = 0
    shards = finished_shards(directory)
    for path in shards:
        with np.load(path) as shard:
            positions += len(shard["outcome"])
    summary = {
        "identity": identity,
        "games": config.games,
        "positions": positions,
        "shards": len(shards),
        "elapsed_s": time.perf_counter() - started,
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log_fn(f"  complete: {config.games} games, {positions} positions")
    return summary
