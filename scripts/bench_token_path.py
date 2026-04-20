"""Microbenchmark: compare PPOActorCritic forward throughput with and
without ``token_features`` on a realistic workload.

Run:
    .venv\\Scripts\\python.exe scripts\\bench_token_path.py [--device cuda] [--batch 64] [--iters 200]
"""
from __future__ import annotations

import argparse
import time

import torch

from src.config import DataConfig, ModelConfig
from src.encoding.action_encoder import get_action_space_size
from src.encoding.state_encoder import get_state_size
from src.ppo.ppo_actor_critic import PPOActorCritic


def _bench(model: PPOActorCritic, x: torch.Tensor, iters: int, warmup: int = 20) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--pool", default="sum", choices=["sum", "attention"])
    args = ap.parse_args()

    device = torch.device(args.device)
    data_cfg = DataConfig()
    cards = data_cfg.load_cards()
    registry = data_cfg.build_registry(cards)
    card_names = registry.card_names
    state_dim = get_state_size(card_names)
    action_dim = get_action_space_size(card_names)
    num_cards = len(card_names)

    x = torch.randn(args.batch, state_dim, device=device)

    legacy = PPOActorCritic(
        state_dim, action_dim, num_cards,
        model_config=ModelConfig(token_features=False, pool_type=args.pool),
    ).to(device)
    tokenized = PPOActorCritic(
        state_dim, action_dim, num_cards,
        model_config=ModelConfig(token_features=True, pool_type=args.pool),
        card_registry=registry,
    ).to(device)

    t_legacy = _bench(legacy, x, args.iters)
    t_tokens = _bench(tokenized, x, args.iters)

    fwd_legacy_us = t_legacy / args.iters * 1e6
    fwd_tokens_us = t_tokens / args.iters * 1e6
    print(f"Device={args.device} batch={args.batch} pool={args.pool} num_cards={num_cards} iters={args.iters}")
    print(f"  legacy   : {fwd_legacy_us:8.1f} us/forward  ({args.batch / (t_legacy / args.iters):.0f} samples/s)")
    print(f"  tokenized: {fwd_tokens_us:8.1f} us/forward  ({args.batch / (t_tokens / args.iters):.0f} samples/s)")
    print(f"  slowdown : {fwd_tokens_us / fwd_legacy_us:.2f}x")


if __name__ == "__main__":
    main()
