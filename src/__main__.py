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
    from src.config import PPOConfig, RunConfig, DeviceConfig
    _ppo = PPOConfig()
    _run = RunConfig()
    _dev = DeviceConfig()

    p = sub.add_parser("train", help="Run PPO training")
    _add_common_args(p)
    # PPO hyperparameters (defaults sourced from PPOConfig dataclass)
    p.add_argument("--lr",          type=float, default=_ppo.lr)
    p.add_argument("--gamma",       type=float, default=_ppo.gamma)
    p.add_argument("--lam",         type=float, default=_ppo.lam)
    p.add_argument("--clip-eps",    type=float, default=_ppo.clip_eps)
    p.add_argument("--epochs",      type=int,   default=_ppo.epochs)
    p.add_argument("--batch-size",  type=int,   default=_ppo.batch_size)
    p.add_argument("--entropy",     type=float, default=_ppo.entropy_coef,
                   help="Entropy bonus coefficient")
    p.add_argument("--adv-norm",    type=str,   default=_ppo.adv_norm,
                   choices=["per_episode", "global"],
                   help="Advantage normalization mode")
    # Run topology (defaults sourced from RunConfig dataclass)
    p.add_argument("--episodes",    type=int,   default=_run.episodes)
    p.add_argument("--updates",     type=int,   default=_run.updates)
    p.add_argument("--eval-every",  type=int,   default=_run.eval_every,
                   help="Evaluate every N updates (always on last)")
    p.add_argument("--eval-games",  type=int,   default=_run.eval_games)
    p.add_argument("--self-play",   action="store_true")
    p.add_argument("--opponents",   type=str, default=_run.opponents,
                   help="Opponent mix: 'random,heuristic' or 'random:0.6,heuristic:0.4'")
    p.add_argument("--self-play-ratio", type=float, default=_run.self_play_ratio,
                   help="Fraction of games using PPO snapshots when self-play is active")
    p.add_argument("--num-workers", type=int, default=_run.num_workers,
                   help=f"Simulation worker processes (1=single-process, >1=multi-process, default: {_run.num_workers})")
    p.add_argument("--num-concurrent", type=int, default=_run.num_concurrent,
                   help=f"Total concurrent games across all workers (default: {_run.num_concurrent})")
    p.add_argument("--pfsp", type=str, default=_run.pfsp_mode,
                   choices=["uniform", "hard", "variance"],
                   help="PFSP snapshot weighting: uniform (default), hard, or variance")
    # Devices (defaults sourced from DeviceConfig dataclass)
    p.add_argument("--main-device",       type=str, default=_dev.main_device,
                   help="Device for training updates")
    p.add_argument("--simulation-device", type=str, default=_dev.simulation_device,
                   help="Device for episode simulation")
    # Model loading
    p.add_argument("--model-path",        type=str, default=None,
                   help="Path to a pretrained PPO model")
    p.add_argument("--load-latest-model", action="store_true",
                   help="Auto-load the latest model from models/")
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

    p = sub.add_parser("elo", help="Run Elo tournament between checkpoints")
    _add_common_args(p)
    p.add_argument("--checkpoints", type=str, nargs="+", required=True,
                   help="Checkpoint glob patterns or paths "
                        "(e.g. 'models/ppo_agent_0415_*upd*0.pth')")
    p.add_argument("--games-per-pair", type=int, default=50,
                   help="Games to play per pairing (default: 50)")
    p.add_argument("--simulation-device", type=str, default=_dev.simulation_device,
                   help="Device for inference (cuda or cpu)")
    p.add_argument("--num-concurrent", type=int, default=_run.num_concurrent,
                   help=f"Concurrent games in BatchRunner (default: {_run.num_concurrent})")
    return p


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="space-deck-builder",
        description="Star Realms AI — training, simulation, and benchmarking",
    )
    sub = parser.add_subparsers(dest="command")
    _build_train_parser(sub)
    _build_simulate_parser(sub)
    _build_benchmark_parser(sub)
    _build_elo_parser(sub)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "train":
        _run_train(args)
    elif args.command == "simulate":
        _run_simulate(args)
    elif args.command == "benchmark":
        _run_benchmark(args)
    elif args.command == "elo":
        _run_elo(args)


def _run_train(args):
    """Construct configs from CLI args and delegate to the trainer."""
    from src.config import DataConfig, PPOConfig, RunConfig, DeviceConfig
    from src.ppo.ppo_trainer import train

    data_cfg = DataConfig(cards_path=args.cards_path)
    ppo_cfg = PPOConfig(
        lr=args.lr, gamma=args.gamma, lam=args.lam,
        clip_eps=args.clip_eps, epochs=args.epochs,
        batch_size=args.batch_size, entropy_coef=args.entropy,
        adv_norm=args.adv_norm,
    )
    run_cfg = RunConfig(
        episodes=args.episodes, updates=args.updates,
        eval_every=args.eval_every, eval_games=args.eval_games,
        self_play=args.self_play,
        opponents=args.opponents,
        self_play_ratio=args.self_play_ratio,
        pfsp_mode=args.pfsp,
        num_workers=args.num_workers,
        num_concurrent=args.num_concurrent,
    )
    dev_cfg = DeviceConfig(
        main_device=args.main_device,
        simulation_device=args.simulation_device,
    )
    train(data_cfg, ppo_cfg, run_cfg, dev_cfg,
          model_path=args.model_path,
          load_latest=args.load_latest_model)


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
    if len(paths) < 2:
        print(f"Error: need at least 2 checkpoints, found {len(paths)}")
        sys.exit(1)

    data_cfg = DataConfig(cards_path=args.cards_path)
    run_tournament(
        checkpoint_paths=paths,
        data_cfg=data_cfg,
        games_per_pair=args.games_per_pair,
        device=args.simulation_device,
        num_concurrent=args.num_concurrent,
    )


if __name__ == "__main__":
    main()
