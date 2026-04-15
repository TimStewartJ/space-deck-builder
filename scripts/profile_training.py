"""Profile a multi-opponent training run to identify bottlenecks.

Usage:
    python scripts/profile_training.py [--device cpu|cuda]
"""
import argparse
import cProfile
import pstats
import io
import time

from src.config import DataConfig, PPOConfig, RunConfig, DeviceConfig
from src.ppo.ppo_trainer import train


def run_profiled_training(device: str = "cpu"):
    data_cfg = DataConfig(cards_path="data/cards.csv")
    ppo_cfg = PPOConfig(
        lr=3e-4, gamma=0.995, lam=0.99,
        clip_eps=0.3, epochs=4, batch_size=1024,
        entropy_coef=0.025,
    )
    run_cfg = RunConfig(
        episodes=128,
        updates=2,
        eval_every=1,
        eval_games=30,
        opponents="random,heuristic,simple",
        self_play=False,
    )
    dev_cfg = DeviceConfig(
        main_device=device,
        simulation_device=device,
    )

    print(f"=== Profiling multi-opponent training on {device} ===")
    print(f"  Episodes per update: {run_cfg.episodes}")
    print(f"  Updates: {run_cfg.updates}")
    print(f"  Opponents: {run_cfg.opponents}")
    print(f"  Eval games: {run_cfg.eval_games}")
    print()

    profiler = cProfile.Profile()
    wall_start = time.perf_counter()
    profiler.enable()

    train(data_cfg, ppo_cfg, run_cfg, dev_cfg)

    profiler.disable()
    wall_end = time.perf_counter()

    print(f"\n{'='*70}")
    print(f"Total wall time: {wall_end - wall_start:.2f}s")
    print(f"{'='*70}\n")

    # Top functions by cumulative time
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats("cumulative")

    print("=== TOP 40 BY CUMULATIVE TIME ===")
    stats.print_stats(40)
    print(stream.getvalue())

    # Top functions by total (self) time
    stream2 = io.StringIO()
    stats2 = pstats.Stats(profiler, stream=stream2)
    stats2.strip_dirs()
    stats2.sort_stats("tottime")

    print("\n=== TOP 40 BY SELF TIME ===")
    stats2.print_stats(40)
    print(stream2.getvalue())

    # Save raw profile for detailed analysis
    profiler.dump_stats("logs/training_profile.prof")
    print("Raw profile saved to logs/training_profile.prof")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for training (cpu or cuda)")
    args = parser.parse_args()
    run_profiled_training(args.device)
