"""Device benchmark matrix: tests all main_device × simulation_device combos."""
import time
import torch
import random
from src.config import DataConfig, PPOConfig, RunConfig, DeviceConfig
from src.ppo.ppo_trainer import train
from src.utils.logger import set_verbose

set_verbose(False)

CONFIGS = [
    ("cpu", "cpu", "CPU/CPU"),
    ("cuda", "cpu", "GPU/CPU"),
    ("cpu", "cuda", "CPU/GPU"),
    ("cuda", "cuda", "GPU/GPU"),
]

results = []
for main_dev, sim_dev, label in CONFIGS:
    print(f"\n===== {label}: main={main_dev}, sim={sim_dev} =====")

    # Fixed seeds for fair comparison
    random.seed(42)
    torch.manual_seed(42)

    data_cfg = DataConfig()
    ppo_cfg = PPOConfig()
    run_cfg = RunConfig(episodes=64, updates=4, eval_every=2, eval_games=100)
    dev_cfg = DeviceConfig(main_device=main_dev, simulation_device=sim_dev)

    start = time.perf_counter()
    train(data_cfg, ppo_cfg, run_cfg, dev_cfg)
    elapsed = time.perf_counter() - start

    print(f">>> [{label}] total={elapsed:.2f}s")
    results.append((label, main_dev, sim_dev, elapsed))

    # Force cleanup between runs
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("\n\n========== DEVICE BENCHMARK MATRIX ==========")
print(f"{'Config':<12} {'Main':>8} {'Sim':>8} {'Total':>10}")
print("-" * 42)
for label, main_dev, sim_dev, elapsed in results:
    print(f"{label:<12} {main_dev:>8} {sim_dev:>8} {elapsed:>9.2f}s")
