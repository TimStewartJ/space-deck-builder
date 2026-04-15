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
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for ML inference (cuda or cpu)")


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
    return p


def _build_benchmark_parser(sub: argparse._SubParsersAction):
    p = sub.add_parser("benchmark", help="Benchmark training throughput")
    _add_common_args(p)
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--mode", type=str, default="both",
                   choices=["sequential", "batched", "parallel", "both"])
    p.add_argument("--workers", type=int, default=4)
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
    from src.ppo.ppo_trainer import main as trainer_main

    # The trainer's main() parses its own args, so we invoke it directly.
    # Re-dispatch via sys.argv so the existing trainer argparse works.
    import sys
    sys.argv = ["ppo_trainer"]
    for attr, flag in [
        ("episodes", "--episodes"), ("updates", "--updates"),
        ("cards_path", "--cards-path"), ("lr", "--lr"),
        ("gamma", "--gamma"), ("lam", "--lam"),
        ("clip_eps", "--clip-eps"), ("epochs", "--epochs"),
        ("batch_size", "--batch-size"), ("entropy", "--entropy"),
        ("device", "--device"), ("main_device", "--main-device"),
        ("simulation_device", "--simulation-device"),
        ("eval_every", "--eval-every"), ("eval_games", "--eval-games"),
    ]:
        val = getattr(args, attr, None)
        if val is not None:
            sys.argv.extend([flag, str(val)])
    if args.self_play:
        sys.argv.append("--self-play")
    if args.model_path:
        sys.argv.extend(["--model-path", args.model_path])
    if args.load_latest_model:
        sys.argv.append("--load-latest-model")
    trainer_main()


def _run_simulate(args):
    """Delegate to the simulator."""
    from src.ppo.ppo_simulate import main as simulate_main
    import sys
    sys.argv = ["ppo_simulate"]
    for attr, flag in [
        ("cards_path", "--cards-path"), ("device", "--device"),
        ("games", "--games"),
    ]:
        val = getattr(args, attr, None)
        if val is not None:
            sys.argv.extend([flag, str(val)])
    if args.model1:
        sys.argv.extend(["--model1", args.model1])
    if args.model2:
        sys.argv.extend(["--model2", args.model2])
    if args.player2_random:
        sys.argv.append("--player2-random")
    simulate_main()


def _run_benchmark(args):
    """Delegate to the benchmark script."""
    from scripts.benchmark import benchmark
    benchmark(
        num_episodes=args.episodes,
        device=args.device,
        mode=args.mode,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
