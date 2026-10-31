"""Unified CLI entrypoint for the Star Realms AI project.

Usage:
    python -m src train [options]
    python -m src simulate [options]
    python -m src benchmark [options]
    python -m src elo [options]
"""
import argparse
import sys


def _add_common_args(parser: argparse.ArgumentParser):
    """Add args shared across subcommands."""
    from src.config import DataConfig
    parser.add_argument("--cards-path", type=str, default=DataConfig().cards_path)


def _build_train_parser(sub: argparse._SubParsersAction):
    from src.config import (
        DataConfig,
        DeviceConfig,
        ModelConfig,
        PPOConfig,
        RunConfig,
        TRAINING_PRESETS,
    )
    _ppo = PPOConfig()
    _run = RunConfig()
    _dev = DeviceConfig()
    _mdl = ModelConfig()

    p = sub.add_parser("train", help="Run PPO training")
    _add_common_args(p)
    p.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=sorted(TRAINING_PRESETS),
        help=(
            "Apply a published training recipe from src/config.py as the "
            "defaults for this run; explicit flags still override it. "
            "Use scripts/train_reference.ps1 to run the complete arc in "
            "fresh-process chunks outside the caller's process tree."
        ),
    )
    p.add_argument(
        "--models-dir",
        type=str,
        default=DataConfig().models_dir,
        help="Checkpoint output directory",
    )
    # PPO hyperparameters (defaults sourced from PPOConfig dataclass)
    p.add_argument("--lr",          type=float, default=_ppo.lr)
    p.add_argument("--lr-end",     type=float, default=_ppo.lr_end,
                   help="Terminal learning rate for cosine annealing")
    p.add_argument("--gamma",       type=float, default=_ppo.gamma)
    p.add_argument("--lam",         type=float, default=_ppo.lam)
    p.add_argument("--clip-eps",    type=float, default=_ppo.clip_eps)
    p.add_argument("--epochs",      type=int,   default=_ppo.epochs)
    p.add_argument("--batch-size",  type=int,   default=_ppo.batch_size)
    p.add_argument("--entropy",     type=float, default=_ppo.entropy_coef,
                   help="Entropy bonus coefficient")
    # Model architecture
    p.add_argument("--actor-type",  type=str, default=_mdl.actor_type,
                   choices=["mlp", "attention"],
                   help="Actor head type: mlp (flat linear) or attention (query-key dot product)")
    p.add_argument("--pool-type",   type=str, default=_mdl.pool_type,
                   choices=["sum", "attention"],
                   help="Zone feature pooling: sum (presence-weighted) or attention (learned per-zone query)")
    p.add_argument("--token-features", action="store_true",
                   default=_mdl.token_features,
                   help="Experimental: enable per-card static feature tokens (cost, defense, faction, "
                        "ally faction, type) instead of pure embedding lookup. "
                        "Requires the card registry at model construction time. "
                        "Default remains disabled.")
    p.add_argument("--trunk-hidden-sizes", type=int, nargs="+", default=None,
                   metavar="N",
                   help=f"Shared trunk hidden layer sizes, e.g. '--trunk-hidden-sizes 512 512' "
                        f"(default: {_mdl.trunk_hidden_sizes})")
    p.add_argument("--actor-head-sizes", type=int, nargs="+", default=None,
                   metavar="N",
                   help=f"Actor head hidden layer sizes (default: {_mdl.actor_head_sizes})")
    p.add_argument("--critic-head-sizes", type=int, nargs="+", default=None,
                   metavar="N",
                   help=f"Critic head hidden layer sizes (default: {_mdl.critic_head_sizes})")
    p.add_argument("--card-emb-dim", type=int, default=_mdl.card_emb_dim,
                   help=f"Per-card embedding dimension (default: {_mdl.card_emb_dim})")
    # Run topology (defaults sourced from RunConfig dataclass)
    p.add_argument("--episodes",    type=int,   default=_run.episodes)
    p.add_argument("--updates",     type=int,   default=_run.updates)
    p.add_argument("--eval-every",  type=int,   default=_run.eval_every,
                   help="Evaluate every N updates (always on last)")
    p.add_argument("--eval-games",  type=int,   default=_run.eval_games)
    p.add_argument(
        "--final-eval-games",
        type=int,
        default=_run.final_eval_games,
        help="Games for the evaluation of this process's last update "
             "(default: --eval-games)",
    )
    p.add_argument(
        "--eval-num-workers",
        type=int,
        default=_run.eval_num_workers,
        help="Evaluation workers (default: inherit --num-workers)",
    )
    p.add_argument("--self-play",   action="store_true", default=_run.self_play,
                   help="Enable self-play snapshots in addition to the fixed opponent pool")
    p.add_argument("--opponents",   type=str, default=_run.opponents,
                   help=f"Fixed opponent mix (default: {_run.opponents}). "
                        "Examples: 'random,heuristic' or 'random:0.6,heuristic:0.4'")
    p.add_argument("--self-play-ratio", type=float, default=_run.self_play_ratio,
                   help="Final self-play ratio (or constant when schedule=constant)")
    p.add_argument("--self-play-ratio-start", type=float, default=_run.self_play_ratio_start,
                   help="Initial self-play ratio at the start of training (used with schedule)")
    p.add_argument("--self-play-schedule", type=str, default=_run.self_play_schedule,
                   choices=["constant", "linear", "cosine"],
                   help="Self-play ratio schedule: linear (default), constant, or cosine ramp")
    p.add_argument(
        "--self-play-horizon",
        type=int,
        default=_run.self_play_horizon,
        help=(
            "Absolute final update for self-play schedule progress. "
            "Used by fresh-process chunked resumes."
        ),
    )
    p.add_argument("--num-workers", type=int, default=_run.num_workers,
                   help=f"Simulation worker processes (1=single-process, >1=multi-process, default: {_run.num_workers})")
    p.add_argument("--engine", choices=["object", "vec"], default=_run.engine,
                   help="Rollout engine: object (default) or compiler-backed Numba vec")
    p.add_argument("--vec-threads", type=int, default=_run.vec_threads,
                   help=f"Numba threads for --engine vec (default: {_run.vec_threads})")
    p.add_argument(
        "--vec-max-active-snapshots",
        type=int,
        default=_run.vec_max_active_snapshots,
        help=(
            "Maximum snapshot models active in one vector rollout "
            f"(default: {_run.vec_max_active_snapshots}; 0=unlimited)"
        ),
    )
    p.add_argument(
        "--vec-graph-inference",
        action=argparse.BooleanOptionalAction,
        default=_run.vec_graph_inference,
        help=(
            "Evaluate vector-rollout policies by CUDA/HIP graph replay "
            f"(default: {_run.vec_graph_inference})"
        ),
    )
    p.add_argument("--num-concurrent", type=int, default=None,
                   help="Concurrent games per worker (default: episodes/workers)")
    p.add_argument("--pfsp", type=str, default=_run.pfsp_mode,
                   choices=["uniform", "hard", "variance"],
                   help="PFSP snapshot weighting: uniform (default), hard, or variance")
    p.add_argument(
        "--frozen-opponents",
        type=str,
        nargs="+",
        default=list(_run.frozen_opponents),
        metavar="CHECKPOINT",
        help=(
            "Frozen checkpoints to train against (vector engine only, not "
            "with --self-play). They are loaded into the opponent pool at the "
            "start of every process, fresh or resumed, and never learn"
        ),
    )
    p.add_argument(
        "--frozen-opponent-ratio",
        type=float,
        default=_run.frozen_opponent_ratio,
        help=(
            "Share of training games against the frozen opponents; the rest "
            f"use --opponents (default: {_run.frozen_opponent_ratio:g})"
        ),
    )
    p.add_argument(
        "--frozen-opponent-temperature",
        type=_temperature_arg,
        default=_run.frozen_opponent_temperature,
        help=(
            "Inference temperature of the frozen opponents: 1 samples their "
            "policy, 0 plays their highest-scoring legal action "
            f"(default: {_run.frozen_opponent_temperature:g})"
        ),
    )
    # Devices (defaults sourced from DeviceConfig dataclass)
    p.add_argument("--main-device",       type=str, default=_dev.main_device,
                   help="Device for training updates")
    p.add_argument("--simulation-device", type=str, default=_dev.simulation_device,
                   help="Device for episode simulation")
    # Model loading
    p.add_argument("--model-path",        type=str, default=None,
                   help="Path to a pretrained PPO model (weights only)")
    p.add_argument("--load-latest-model", action="store_true",
                   help="Auto-load the latest model from models/ (weights only)")
    p.add_argument("--resume",            type=str, default=None,
                   help="Resume training from a checkpoint: restores weights, "
                        "optimizer, LR scheduler, snapshot pool, and update counter. "
                        "--updates is interpreted as additional updates beyond the resumed step.")
    p.add_argument("--lr-horizon",        type=int, default=None,
                   help="Override the cosine LR scheduler horizon (final update). "
                        "Use this when chunking a multi-process resume so each chunk "
                        "re-pins T_max to the same final-target update (e.g. 200) "
                        "and the cosine LR curve flows smoothly across chunks "
                        "instead of decaying to the floor inside each one.")
    # Reproducibility (per docs/eval_protocol.md). When set, seeds Python
    # `random`, `numpy`, and `torch` deterministically — which in turn makes
    # the per-worker seed lists in batch_runner / mp_batch_runner reproducible
    # since they are drawn from `random.randint`.
    p.add_argument("--seed",              type=int, default=None,
                   help="Master RNG seed for reproducibility. Seeds Python "
                        "random, numpy, and torch (CPU+CUDA). When omitted, "
                        "training uses non-reproducible OS entropy.")
    return p


def _build_simulate_parser(sub: argparse._SubParsersAction):
    from src.config import DeviceConfig, SimConfig
    _dev = DeviceConfig()
    _sim = SimConfig()

    p = sub.add_parser("simulate", help="Run PPO vs opponent simulation")
    _add_common_args(p)
    p.add_argument("--model1", type=str, default=None,
                   help="PPO model for player 1 (default: latest)")
    p.add_argument("--model2", type=str, default=None,
                   help="PPO model for player 2")
    p.add_argument("--player2-random", action="store_true", default=True,
                   help="Use random agent for player 2")
    p.add_argument("--games", type=int, default=_sim.games)
    p.add_argument("--simulation-device", type=str, default=_dev.simulation_device,
                   help="Device for inference (cuda or cpu)")
    return p


def _build_benchmark_parser(sub: argparse._SubParsersAction):
    from src.config import DeviceConfig
    _dev = DeviceConfig()

    p = sub.add_parser("benchmark", help="Benchmark training throughput")
    _add_common_args(p)
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--mode", type=str, default="both",
                   choices=["sequential", "batched", "parallel", "both"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--simulation-device", type=str, default=_dev.simulation_device,
                   help="Device for inference (cuda or cpu)")
    return p


def _build_elo_parser(sub: argparse._SubParsersAction):
    from src.config import DeviceConfig, RunConfig
    _dev = DeviceConfig()
    _run = RunConfig()

    p = sub.add_parser("elo", help="Run Elo tournament between checkpoints and/or agents")
    _add_common_args(p)
    p.add_argument("--checkpoints", type=str, nargs="*", default=[],
                   help="Checkpoint glob patterns or paths "
                        "(e.g. 'models/ppo_agent_0415_*upd*0.pth')")
    p.add_argument("--agents", type=str, default=None,
                   help="Comma-separated built-in agent types to include "
                        "(e.g. 'random,heuristic,simple')")
    p.add_argument("--games-per-pair", type=int, default=1000,
                   help="Games to play per pairing (default: 1000)")
    p.add_argument("--simulation-device", type=str, default=_dev.simulation_device,
                   help="Device for inference (cuda or cpu)")
    p.add_argument("--num-concurrent", type=int, default=None,
                   help="Concurrent games per worker (default: games-per-pair)")
    p.add_argument("--num-workers", type=int, default=_run.num_workers,
                   help=f"Worker processes for PPO pairings (default: {_run.num_workers})")
    p.add_argument("--analyze", action="store_true", default=False,
                   help="Collect replays and generate comparative dashboard")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Directory for replay files (default: analysis/elo_<timestamp>/)")
    p.add_argument("--checkpoint-temperatures", type=_temperature_list_arg,
                   default=[1.0],
                   help="Comma-separated inference temperatures. Every "
                        "checkpoint participates once per temperature; "
                        "labels gain an '@T<temperature>' suffix when the "
                        "temperature is not 1 (default: 1)")
    p.add_argument("--output-json", type=str, default=None,
                   help="Write participants, ratings, and pairing results to this JSON file")
    return p


def _build_eval_parser(sub: argparse._SubParsersAction):
    from src.config import DeviceConfig, RunConfig
    _dev = DeviceConfig()
    _run = RunConfig()

    p = sub.add_parser("eval", help="Evaluate a trained model against fixed opponents")
    _add_common_args(p)
    p.add_argument("--model", type=str, default=None,
                   help="Model checkpoint path (default: latest)")
    p.add_argument("--load-latest", action="store_true",
                   help="Auto-load the latest model from models/")
    p.add_argument("--opponents", type=str, default="random,heuristic,simple",
                   help="Comma-separated opponent types (default: random,heuristic,simple)")
    p.add_argument("--games", type=int, default=_run.eval_games,
                   help=f"Total evaluation games (distributed across opponent types, default: {_run.eval_games})")
    p.add_argument("--simulation-device", type=str, default=_dev.simulation_device,
                   help="Device for inference (cuda or cpu)")
    p.add_argument("--num-concurrent", type=int, default=None,
                   help="Concurrent games per worker (default: games/workers)")
    p.add_argument("--num-workers", type=int, default=_run.num_workers,
                   help=f"Simulation worker processes (default: {_run.num_workers})")
    # Reproducibility (per docs/eval_protocol.md). Seeds Python random,
    # numpy, and torch before per-worker seed derivation so gauntlet runs
    # are reproducible up to the limits of GPU non-determinism.
    p.add_argument("--seed", type=int, default=None,
                   help="Master RNG seed for reproducible eval. Seeds "
                        "Python random, numpy, and torch (CPU+CUDA). When "
                        "omitted, eval uses non-reproducible OS entropy.")
    p.add_argument("--temperature", type=_temperature_arg, default=1.0,
                   help="Inference temperature for the evaluated model: 1 "
                        "samples the policy (protocol default), 0 selects "
                        "the highest-probability legal action")
    p.add_argument("--output-json", type=str, default=None,
                   help="Write per-opponent results and provenance to this JSON file")


def _temperature_arg(value: str) -> float:
    from src.ppo.action_selection import validate_temperature

    try:
        return validate_temperature(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _temperature_list_arg(value: str) -> list[float]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one temperature")
    temperatures = [_temperature_arg(item) for item in items]
    if len(set(temperatures)) != len(temperatures):
        raise argparse.ArgumentTypeError(f"duplicate temperatures in {value!r}")
    return temperatures


def _build_analyze_parser(sub: argparse._SubParsersAction):
    from src.config import DeviceConfig, RunConfig
    _dev = DeviceConfig()
    _run = RunConfig()

    p = sub.add_parser("analyze", help="Collect replays and analyze agent behavior")
    _add_common_args(p)
    p.add_argument("--model", type=str, default=None,
                   help="Model checkpoint path (default: latest)")
    p.add_argument("--games", type=int, default=200,
                   help="Number of games to collect (default: 200)")
    p.add_argument("--opponents", type=str, default="random,heuristic,simple",
                   help="Opponent types: random,heuristic,simple,self (default: random,heuristic,simple)")
    p.add_argument("--opponent-model", type=str, default=None,
                   help="Path to a model checkpoint to use as opponent (overrides --opponents)")
    p.add_argument("--output", type=str, default=None,
                   help="Output replay file path (default: analysis/replays_<timestamp>.json.gz)")
    p.add_argument("--replay", type=str, default=None,
                   help="Path to existing replay file to analyze (skips game collection)")
    p.add_argument("--simulation-device", type=str, default=_dev.simulation_device,
                   help="Device for inference (cuda or cpu)")
    p.add_argument("--num-concurrent", type=int, default=None,
                   help="Concurrent games per worker (default: games/workers)")
    p.add_argument("--dashboard", action="store_true", default=False,
                   help="Generate interactive HTML dashboard (in addition to PNGs)")
    return p


def _build_search_eval_parser(sub: argparse._SubParsersAction):
    from src.config import DeviceConfig
    from src.search.turn_search import TurnSearchConfig

    p = sub.add_parser(
        "search-eval",
        help="Measure turn-bounded search against the same policy without search",
    )
    p.add_argument("--model", type=str, required=True,
                   help="Checkpoint whose policy and critic guide the search and baseline")
    p.add_argument("--budgets", type=int, nargs="+", default=[0, 16, 64, 256],
                   help="Rollout budgets per decision; 0 plays greedily (fairness control)")
    p.add_argument("--games", type=int, default=1000,
                   help="Games per budget (even; exact seat alternation)")
    p.add_argument("--baseline", choices=["greedy", "sampled", "heuristic"], default="greedy",
                   help="Opponent: the checkpoint at temperature 0 or 1, or the Heuristic agent")
    p.add_argument("--max-candidates", type=int, default=TurnSearchConfig().max_candidates,
                   help="Highest-prior root actions compared by rollouts")
    p.add_argument("--seed", type=int, default=0,
                   help="Master seed for game deals and search worlds")
    p.add_argument("--workers", type=int, default=2,
                   help="Worker processes, each with its own policy copy and game batch")
    p.add_argument("--numba-threads", type=int, default=4,
                   help="Numba threads per worker")
    p.add_argument("--batch-size", type=int, default=1024,
                   help="Games played concurrently per worker; their policy and "
                        "search decisions are evaluated together")
    p.add_argument("--simulation-device", type=str, default=DeviceConfig().simulation_device,
                   help="Device for policy and critic inference")
    p.add_argument("--output-json", type=str, default=None,
                   help="Write the summary and per-game results; resumes from its .games.jsonl journal")
    return p


def _build_search_selfplay_parser(sub: argparse._SubParsersAction):
    from src.config import DeviceConfig
    from src.search.selfplay_data import SelfPlayConfig

    defaults = SelfPlayConfig(model_path="", games=1)
    p = sub.add_parser(
        "search-selfplay",
        help="Record search-versus-search games as training shards",
    )
    p.add_argument("--model", type=str, required=True,
                   help="Checkpoint whose policy and critic guide both searching seats")
    p.add_argument("--games", type=int, required=True, help="Games to record")
    p.add_argument("--budget", type=int, default=defaults.budget,
                   help="Rollouts per searched decision")
    p.add_argument("--max-candidates", type=int, default=defaults.max_candidates,
                   help="Highest-prior root actions compared by rollouts")
    p.add_argument("--seed", type=int, default=defaults.seed,
                   help="Master seed for game deals and search worlds")
    p.add_argument("--workers", type=int, default=defaults.workers,
                   help="Worker processes, each with its own policy copy and game batch")
    p.add_argument("--numba-threads", type=int, default=defaults.numba_threads,
                   help="Numba threads per worker")
    p.add_argument("--batch-size", type=int, default=defaults.batch_size,
                   help="Games played concurrently per worker")
    p.add_argument("--shard-positions", type=int, default=defaults.shard_positions,
                   help="Positions buffered before a worker writes a shard")
    p.add_argument("--simulation-device", type=str, default=DeviceConfig().simulation_device,
                   help="Device for policy and critic inference")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Directory for shards; an existing directory with the same configuration resumes")
    return p


def _build_distill_parser(sub: argparse._SubParsersAction):
    from src.search.distill import DistillConfig

    defaults = DistillConfig(model_path="", data_dir="", output_path="")
    p = sub.add_parser(
        "distill",
        help="Train a checkpoint on search self-play targets",
    )
    p.add_argument("--model", type=str, required=True,
                   help="Checkpoint to start from (the one that guided the search)")
    p.add_argument("--data", type=str, required=True, help="search-selfplay output directory")
    p.add_argument("--output", type=str, required=True, help="Checkpoint path to write")
    p.add_argument("--lr", type=float, default=defaults.lr)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--epochs", type=int, default=defaults.max_epochs,
                   help="Maximum training epochs")
    p.add_argument("--patience", type=int, default=defaults.patience,
                   help="Epochs without validation improvement before stopping")
    p.add_argument("--value-coef", type=float, default=defaults.value_coef)
    p.add_argument("--grad-clip", type=float, default=defaults.grad_clip)
    p.add_argument("--validation-modulus", type=int, default=defaults.validation_modulus,
                   help="Games with index %% modulus == 0 are held out for validation")
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--device", type=str, default=defaults.device)
    return p


def _build_league_parser(sub: argparse._SubParsersAction):
    from src.ai.held_out_agents import HELD_OUT_STYLES
    from src.ppo.league import LeagueConfig

    defaults = LeagueConfig(calibration=True)
    p = sub.add_parser(
        "league",
        help="Play checkpoints against the held-out scripted opponents",
        description=(
            "Held-out league of docs/decisions/0013-strength-yardstick.md: each "
            "checkpoint plays every held-out opponent at temperature 0 with exact "
            "seat alternation. Finished matchups are journaled next to the output, "
            "so re-running resumes."
        ),
    )
    p.add_argument("--models", nargs="*", default=[], metavar="CHECKPOINT",
                   help="Checkpoints to measure")
    p.add_argument("--opponents", type=str, default=",".join(defaults.opponents),
                   help=f"Comma-separated held-out opponents (default: all of {', '.join(HELD_OUT_STYLES)})")
    p.add_argument("--games", type=int, default=defaults.games,
                   help=f"Games per checkpoint and opponent (default: {defaults.games})")
    p.add_argument("--seed", type=int, default=defaults.seed,
                   help=f"Master deal seed (default: {defaults.seed})")
    p.add_argument("--batch-size", type=int, default=defaults.batch_size,
                   help=f"Games in flight per process (default: {defaults.batch_size})")
    p.add_argument("--calibration", action="store_true",
                   help=(f"Also play each held-out opponent against Heuristic "
                         f"({defaults.calibration_games} games, deal seed {defaults.calibration_seed})"))
    p.add_argument("--workers", type=int, default=defaults.workers,
                   help=f"Matchups played in parallel processes (default: {defaults.workers})")
    p.add_argument("--simulation-device", type=str, default=defaults.device,
                   help="Device for checkpoint inference")
    p.add_argument("--output-json", type=str, required=True,
                   help="League summary; its .journal.jsonl records finished matchups")
    return p


def _build_purchase_audit_parser(sub: argparse._SubParsersAction):
    from src.search.purchase_audit import AuditConfig

    defaults = AuditConfig(model_path="")
    p = sub.add_parser(
        "purchase-audit",
        help="Measure the win probability a checkpoint's purchases give away",
        description=(
            "Purchase audit of docs/decisions/0013-strength-yardstick.md: one "
            "uniformly sampled purchase decision per greedy self-play game, every "
            "candidate scored by complete greedy rollouts from the acting player's "
            "information set. Finished decisions are journaled next to the output, "
            "so re-running resumes."
        ),
    )
    p.add_argument("--model", type=str, required=True, help="Checkpoint to audit")
    p.add_argument("--decisions", type=int, default=defaults.decisions,
                   help=f"Decisions to audit, one per game (default: {defaults.decisions})")
    p.add_argument("--rollouts", type=int, default=defaults.rollouts,
                   help=f"Complete rollouts per candidate (default: {defaults.rollouts})")
    p.add_argument("--seed", type=int, default=defaults.seed,
                   help=f"Master seed for deals, targets, and rollouts (default: {defaults.seed})")
    p.add_argument("--rows-per-chunk", type=int, default=defaults.rows_per_chunk,
                   help=f"Rollout rows per vector population (default: {defaults.rows_per_chunk})")
    p.add_argument("--numba-threads", type=int, default=defaults.numba_threads)
    p.add_argument("--simulation-device", type=str, default=defaults.device,
                   help="Device for policy inference")
    p.add_argument("--output-json", type=str, required=True,
                   help="Audit records and summary; its .journal.jsonl records finished decisions")
    return p


def build_cli_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    """Build the unified CLI parser and return it with the train subparser."""
    parser = argparse.ArgumentParser(
        prog="space-deck-builder",
        description="Star Realms AI — training, simulation, and benchmarking",
    )
    sub = parser.add_subparsers(dest="command")
    train_parser = _build_train_parser(sub)
    _build_simulate_parser(sub)
    _build_benchmark_parser(sub)
    _build_elo_parser(sub)
    _build_eval_parser(sub)
    _build_analyze_parser(sub)
    _build_search_eval_parser(sub)
    _build_search_selfplay_parser(sub)
    _build_distill_parser(sub)
    _build_league_parser(sub)
    _build_purchase_audit_parser(sub)
    return parser, train_parser


def parse_cli_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments, applying a training preset before explicit flags.

    A preset changes the train subparser's defaults and the arguments are
    parsed again, so every flag given on the command line overrides the
    preset while omitted flags take the published recipe values.
    """
    parser, train_parser = build_cli_parser()
    args = parser.parse_args(argv)
    if args.command == "train" and getattr(args, "preset", None):
        from src.config import get_training_preset

        preset = get_training_preset(args.preset)
        train_parser.set_defaults(**preset.train_defaults)
        args = parser.parse_args(argv)
    return args


def main(argv=None):
    args = parse_cli_args(argv)

    if args.command is None:
        build_cli_parser()[0].print_help()
        sys.exit(1)

    if args.command == "train":
        _run_train(args)
    elif args.command == "simulate":
        _run_simulate(args)
    elif args.command == "benchmark":
        _run_benchmark(args)
    elif args.command == "elo":
        _run_elo(args)
    elif args.command == "eval":
        _run_eval(args)
    elif args.command == "analyze":
        _run_analyze(args)
    elif args.command == "search-eval":
        _run_search_eval(args)
    elif args.command == "search-selfplay":
        _run_search_selfplay(args)
    elif args.command == "distill":
        _run_distill(args)
    elif args.command == "league":
        _run_league(args)
    elif args.command == "purchase-audit":
        _run_purchase_audit(args)


def _run_purchase_audit(args):
    """Audit a checkpoint's purchase decisions with complete rollouts."""
    from src.search.purchase_audit import AuditConfig, run_audit

    try:
        config = AuditConfig(
            model_path=args.model,
            decisions=args.decisions,
            rollouts=args.rollouts,
            seed=args.seed,
            rows_per_chunk=args.rows_per_chunk,
            numba_threads=args.numba_threads,
            device=args.simulation_device,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    run_audit(config, args.output_json)


def _run_league(args):
    """Play checkpoints against the held-out opponents and write the league table."""
    from src.ppo.league import LeagueConfig, run_league

    try:
        config = LeagueConfig(
            models=tuple(args.models),
            opponents=tuple(name.strip() for name in args.opponents.split(",") if name.strip()),
            games=args.games,
            seed=args.seed,
            batch_size=args.batch_size,
            calibration=args.calibration,
            workers=args.workers,
            device=args.simulation_device,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    run_league(config, args.output_json)


def _run_search_selfplay(args):
    """Record search self-play shards."""
    from src.search.selfplay_data import SelfPlayConfig, generate_selfplay

    try:
        config = SelfPlayConfig(
            model_path=args.model,
            games=args.games,
            budget=args.budget,
            max_candidates=args.max_candidates,
            seed=args.seed,
            device=args.simulation_device,
            workers=args.workers,
            numba_threads=args.numba_threads,
            batch_size=args.batch_size,
            shard_positions=args.shard_positions,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    generate_selfplay(config, args.output_dir)


def _run_distill(args):
    """Distill search self-play targets into a checkpoint."""
    from src.search.distill import DistillConfig, distill

    try:
        config = DistillConfig(
            model_path=args.model,
            data_dir=args.data,
            output_path=args.output,
            lr=args.lr,
            batch_size=args.batch_size,
            max_epochs=args.epochs,
            patience=args.patience,
            value_coef=args.value_coef,
            grad_clip=args.grad_clip,
            validation_modulus=args.validation_modulus,
            seed=args.seed,
            device=args.device,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    distill(config)


def _run_search_eval(args):
    """Run the search uplift ladder and print its summary."""
    from src.search.uplift import UpliftConfig, run_uplift

    try:
        config = UpliftConfig(
            model_path=args.model,
            budgets=tuple(args.budgets),
            games=args.games,
            baseline=args.baseline,
            max_candidates=args.max_candidates,
            seed=args.seed,
            device=args.simulation_device,
            workers=args.workers,
            numba_threads=args.numba_threads,
            batch_size=args.batch_size,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    run_uplift(config, output_json=args.output_json)


def _run_train(args):
    """Construct configs from CLI args and delegate to the trainer."""
    from src.config import DataConfig, PPOConfig, RunConfig, DeviceConfig, ModelConfig
    from src.ppo.ppo_trainer import train

    if getattr(args, "preset", None):
        print(f"Training preset: {args.preset}", flush=True)
    data_cfg = DataConfig(
        cards_path=args.cards_path,
        models_dir=args.models_dir,
    )
    ppo_cfg = PPOConfig(
        lr=args.lr, gamma=args.gamma, lam=args.lam,
        clip_eps=args.clip_eps, epochs=args.epochs,
        batch_size=args.batch_size, entropy_coef=args.entropy,
        lr_end=args.lr_end,
    )
    model_cfg_kwargs = dict(
        actor_type=args.actor_type,
        pool_type=args.pool_type,
        token_features=args.token_features,
        card_emb_dim=args.card_emb_dim,
    )
    if args.trunk_hidden_sizes is not None:
        model_cfg_kwargs["trunk_hidden_sizes"] = list(args.trunk_hidden_sizes)
    if args.actor_head_sizes is not None:
        model_cfg_kwargs["actor_head_sizes"] = list(args.actor_head_sizes)
    if args.critic_head_sizes is not None:
        model_cfg_kwargs["critic_head_sizes"] = list(args.critic_head_sizes)
    model_cfg = ModelConfig(**model_cfg_kwargs)
    run_cfg = RunConfig(
        engine=args.engine,
        vec_threads=args.vec_threads,
        vec_max_active_snapshots=args.vec_max_active_snapshots,
        vec_graph_inference=args.vec_graph_inference,
        episodes=args.episodes, updates=args.updates,
        eval_every=args.eval_every, eval_games=args.eval_games,
        final_eval_games=args.final_eval_games,
        eval_num_workers=args.eval_num_workers,
        self_play=args.self_play,
        opponents=args.opponents,
        self_play_ratio=args.self_play_ratio,
        self_play_ratio_start=args.self_play_ratio_start,
        self_play_schedule=args.self_play_schedule,
        self_play_horizon=args.self_play_horizon,
        pfsp_mode=args.pfsp,
        frozen_opponents=list(args.frozen_opponents),
        frozen_opponent_ratio=args.frozen_opponent_ratio,
        frozen_opponent_temperature=args.frozen_opponent_temperature,
        num_workers=args.num_workers,
        num_concurrent=args.num_concurrent,
        resume=args.resume,
        lr_horizon=args.lr_horizon,
    )
    dev_cfg = DeviceConfig(
        main_device=args.main_device,
        simulation_device=args.simulation_device,
    )
    train(data_cfg, ppo_cfg, run_cfg, dev_cfg,
          model_path=args.model_path,
          load_latest=args.load_latest_model,
          model_config=model_cfg,
          seed=args.seed)


def _run_simulate(args):
    """Construct configs and delegate to the simulator."""
    from src.config import DataConfig, SimConfig
    from src.ppo.ppo_simulate import simulate

    data_cfg = DataConfig(cards_path=args.cards_path)
    sim_cfg = SimConfig(games=args.games, player2_random=args.player2_random)
    simulate(data_cfg, sim_cfg,
             model1_path=args.model1,
             model2_path=args.model2,
             device=args.simulation_device)


def _run_benchmark(args):
    """Delegate to the benchmark function."""
    from scripts.benchmark import benchmark
    benchmark(
        num_episodes=args.episodes,
        device=args.simulation_device,
        mode=args.mode,
        workers=args.workers,
    )


def _run_elo(args):
    """Resolve checkpoint paths, build config, and run the Elo tournament."""
    from src.config import DataConfig
    from src.ppo.elo_tournament import run_tournament, resolve_checkpoint_paths

    paths = resolve_checkpoint_paths(args.checkpoints)
    agent_types = [a.strip() for a in args.agents.split(",") if a.strip()] if args.agents else []

    # Validate agent types early for clear CLI error messages
    from src.ppo.elo_tournament import BUILTIN_AGENT_TYPES
    invalid = [a for a in agent_types if a not in BUILTIN_AGENT_TYPES]
    if invalid:
        print(f"Error: unknown agent type(s): {', '.join(invalid)}. "
              f"Valid types: {', '.join(sorted(BUILTIN_AGENT_TYPES))}")
        sys.exit(1)

    checkpoint_participants = len(paths) * len(args.checkpoint_temperatures)
    total_participants = checkpoint_participants + len(agent_types)
    if total_participants < 2:
        print(f"Error: need at least 2 participants, found {total_participants} "
              f"({len(paths)} checkpoints x {len(args.checkpoint_temperatures)} "
              f"temperatures, {len(agent_types)} agents)")
        sys.exit(1)

    data_cfg = DataConfig(cards_path=args.cards_path)
    num_concurrent = args.num_concurrent or args.games_per_pair
    run_tournament(
        checkpoint_paths=paths,
        data_cfg=data_cfg,
        games_per_pair=args.games_per_pair,
        device=args.simulation_device,
        num_concurrent=num_concurrent,
        agent_types=agent_types or None,
        num_workers=args.num_workers,
        collect_replays=args.analyze,
        replay_output_dir=getattr(args, 'output_dir', None),
        checkpoint_temperatures=args.checkpoint_temperatures,
        output_json=args.output_json,
    )


def _run_eval(args):
    """Load a model and evaluate it against fixed opponent types."""
    from src.config import DataConfig
    from src.ppo.ppo_eval import load_and_evaluate

    data_cfg = DataConfig(cards_path=args.cards_path)
    try:
        load_and_evaluate(
            model_path=args.model,
            load_latest=args.load_latest or args.model is None,
            data_cfg=data_cfg,
            device=args.simulation_device,
            opponents=args.opponents,
            eval_games=args.games,
            num_concurrent=args.num_concurrent,
            num_workers=args.num_workers,
            seed=args.seed,
            temperature=args.temperature,
            output_json=args.output_json,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


def _run_analyze(args):
    """Collect replays from games and/or analyze existing replay data."""
    from src.analysis.analyzer import analyze_replays

    # If a replay file is provided, skip collection and just analyze
    if args.replay:
        print(f"Analyzing existing replay file: {args.replay}")
        analyze_replays(args.replay)
        if args.dashboard:
            from src.analysis.dashboard import generate_dashboard
            dash_path = generate_dashboard(args.replay)
            print(f"Dashboard saved to: {dash_path}")
        return

    # Otherwise, collect replays by playing games
    import os
    import time
    import torch
    from datetime import datetime
    from src.config import DataConfig, DeviceConfig
    from src.ppo.ppo_simulate import get_latest_model
    from src.ppo.batch_runner import BatchRunner
    from src.ppo.ppo_actor_critic import PPOActorCritic
    from src.encoding.state_encoder import get_state_size
    from src.encoding.action_encoder import get_action_space_size
    from src.config import load_checkpoint
    from src.analysis.replay_collector import ReplayCollector
    from src.ppo.opponent_pool import OpponentPool
    from src.utils.logger import set_verbose, set_disabled

    set_verbose(False)
    set_disabled(True)

    data_cfg = DataConfig(cards_path=args.cards_path)
    device = DeviceConfig.resolve(args.simulation_device)

    cards = data_cfg.load_cards()
    registry = data_cfg.build_registry(cards)
    card_names = registry.card_names

    model_path = args.model or get_latest_model(data_cfg.models_dir)
    if not model_path:
        print("Error: no model found. Provide --model or train one first.")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    ckpt = load_checkpoint(model_path, map_location=device)
    saved_cfg = ckpt.get("config", {}).get("model")
    from src.config import ModelConfig
    model_config = ModelConfig.from_dict(saved_cfg) if saved_cfg else ModelConfig()

    state_dim = get_state_size(card_names)
    action_dim = get_action_space_size(card_names)

    model = PPOActorCritic(
        state_dim, action_dim, len(card_names), model_config=model_config,
        card_registry=registry,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Distribute games across opponent types, spreading the remainder
    opponent_types = [o.strip().split(":")[0] for o in args.opponents.split(",")]

    # --opponent-model overrides --opponents with a single model opponent
    if args.opponent_model:
        opponent_types = ["ppo-opponent"]

    num_types = len(opponent_types)
    base_games = args.games // num_types
    remainder = args.games % num_types
    games_schedule = [base_games + (1 if i < remainder else 0) for i in range(num_types)]

    collector = ReplayCollector(card_names, action_dim)
    total_wins = 0
    total_games = 0

    # Build opponent factories — model-based types use a shared model to
    # avoid per-game GPU allocations; fixed agent types go through OpponentPool.
    from src.ppo.elo_tournament import _make_opponent_factory

    # Pre-load opponent model if --opponent-model was given
    opp_model_sd = None
    opp_model_config = None
    if args.opponent_model:
        print(f"Loading opponent model: {args.opponent_model}")
        opp_ckpt = load_checkpoint(args.opponent_model, map_location=device)
        opp_model_sd = opp_ckpt["model_state_dict"]
        saved_opp_cfg = opp_ckpt.get("config", {}).get("model")
        opp_model_config = ModelConfig.from_dict(saved_opp_cfg) if saved_opp_cfg else model_config

    # Fixed agent types go through OpponentPool
    fixed_types = [t for t in opponent_types if t not in ("self", "ppo-opponent")]
    pool = OpponentPool(opponent_spec=",".join(fixed_types)) if fixed_types else None

    for opp_type, num_games in zip(opponent_types, games_schedule):
        if num_games == 0:
            continue
        print(f"Collecting {num_games} games vs {opp_type}...")

        if opp_type == "self":
            factory = _make_opponent_factory(
                ckpt["model_state_dict"], card_names, device,
                model_config=model_config, card_registry=registry,
            )
        elif opp_type == "ppo-opponent":
            factory = _make_opponent_factory(
                opp_model_sd, card_names, device,
                model_config=opp_model_config, card_registry=registry,
            )
        else:
            factory = pool.make_factory_for_type(opp_type, card_names, device=device)

        runner = BatchRunner(
            model=model,
            card_names=card_names,
            cards=cards,
            action_dim=action_dim,
            device=torch.device(device),
            opponent_factory=factory,
            num_concurrent=min(args.num_concurrent or args.games, num_games),
            registry=registry,
        )

        start = time.time()
        w, l, s = runner.run_analysis(num_games, collector, opp_type)
        elapsed = time.time() - start
        total_wins += w
        total_games += num_games
        print(f"  {w}/{num_games} wins ({w / num_games:.0%}) in {elapsed:.1f}s")

    print(f"\nTotal: {total_wins}/{total_games} wins "
          f"({total_wins / total_games:.0%}), "
          f"{len(collector.replays)} games collected, "
          f"{sum(len(r.decisions) for r in collector.replays):,} decisions")

    # Save replays
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"analysis/replays_{ts}.json.gz"
    os.makedirs(os.path.dirname(output_path) or "analysis", exist_ok=True)
    collector.save(output_path)
    print(f"Replays saved to: {output_path}")

    # Analyze
    analyze_replays(output_path, output_dir=os.path.dirname(output_path) or "analysis")

    # Generate interactive dashboard if requested
    if args.dashboard:
        from src.analysis.dashboard import generate_dashboard
        dash_path = generate_dashboard(
            output_path,
            model_info=f"Model: {model_path}",
        )
        print(f"Dashboard saved to: {dash_path}")


if __name__ == "__main__":
    main()
