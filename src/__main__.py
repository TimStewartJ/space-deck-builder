"""Unified CLI entrypoint for the Star Realms AI project.

Usage:
    python -m src train [options]
    python -m src simulate [options]
    python -m src benchmark [options]
"""
import argparse
import sys


def _add_common_args(parser: argparse.ArgumentParser):
    """Add args shared across subcommands."""
    parser.add_argument("--cards-path", type=str, default="data/cards.csv")


def _build_train_parser(sub: argparse._SubParsersAction):
    p = sub.add_parser("train", help="Run PPO training")
    _add_common_args(p)
    # PPO hyperparameters
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--gamma",       type=float, default=0.995)
    p.add_argument("--lam",         type=float, default=0.99)
    p.add_argument("--clip-eps",    type=float, default=0.3)
    p.add_argument("--epochs",      type=int,   default=4)
    p.add_argument("--batch-size",  type=int,   default=1024)
    p.add_argument("--entropy",     type=float, default=0.025,
                   help="Entropy bonus coefficient")
    # Run topology
    p.add_argument("--episodes",    type=int,   default=1024)
    p.add_argument("--updates",     type=int,   default=4)
    p.add_argument("--eval-every",  type=int,   default=5,
                   help="Evaluate every N updates (always on last)")
    p.add_argument("--eval-games",  type=int,   default=100)
    p.add_argument("--self-play",   action="store_true")
    p.add_argument("--opponents",   type=str, default="random",
                   help="Opponent mix: 'random,heuristic' or 'random:0.6,heuristic:0.4'")
    p.add_argument("--self-play-ratio", type=float, default=0.5,
                   help="Fraction of games using PPO snapshots when self-play is active")
    # Devices
    p.add_argument("--main-device",       type=str, default="cuda",
                   help="Device for training updates")
    p.add_argument("--simulation-device", type=str, default="cpu",
                   help="Device for episode simulation")
    # Model loading
    p.add_argument("--model-path",        type=str, default=None,
                   help="Path to a pretrained PPO model")
    p.add_argument("--load-latest-model", action="store_true",
                   help="Auto-load the latest model from models/")
    return p


def _build_simulate_parser(sub: argparse._SubParsersAction):
    p = sub.add_parser("simulate", help="Run PPO vs opponent simulation")
    _add_common_args(p)
    p.add_argument("--model1", type=str, default=None,
                   help="PPO model for player 1 (default: latest)")
    p.add_argument("--model2", type=str, default=None,
                   help="PPO model for player 2")
    p.add_argument("--player2-random", action="store_true", default=True,
                   help="Use random agent for player 2")
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--simulation-device", type=str, default="cpu",
                   help="Device for inference (cuda or cpu)")
    return p


def _build_benchmark_parser(sub: argparse._SubParsersAction):
    p = sub.add_parser("benchmark", help="Benchmark training throughput")
    _add_common_args(p)
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--mode", type=str, default="both",
                   choices=["sequential", "batched", "parallel", "both"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--simulation-device", type=str, default="cpu",
                   help="Device for inference (cuda or cpu)")
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


def _run_train(args):
    """Construct configs from CLI args and delegate to the trainer."""
    from src.config import DataConfig, PPOConfig, RunConfig, DeviceConfig
    from src.ppo.ppo_trainer import train

    data_cfg = DataConfig(cards_path=args.cards_path)
    ppo_cfg = PPOConfig(
        lr=args.lr, gamma=args.gamma, lam=args.lam,
        clip_eps=args.clip_eps, epochs=args.epochs,
        batch_size=args.batch_size, entropy_coef=args.entropy,
    )
    run_cfg = RunConfig(
        episodes=args.episodes, updates=args.updates,
        eval_every=args.eval_every, eval_games=args.eval_games,
        self_play=args.self_play,
        opponents=args.opponents,
        self_play_ratio=args.self_play_ratio,
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


if __name__ == "__main__":
    main()
