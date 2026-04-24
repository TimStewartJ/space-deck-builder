"""Realistic benchmark for the token-features path.

Measures wall time of a full training-shaped step (forward + backward + opt)
at the actual training batch size, with both code paths:

    legacy   :  token_features=False
    tokens   :  token_features=True (uses the cube-free fused sum pool)

The earlier version of this script measured forward-only at batch size 64
and reported a 1.06–1.12× slowdown for the tokenized path. That number
turned out to be wildly optimistic on the real workload — the original
``_build_tokens`` materialized a ``[B, Z, C, E]`` cube whose memory cost
dominates a real PPO update step (B=512, ~14k minibatches per update) and
backprop. A 200-update run that should have taken ~1 hour was on track for
~16 hours. The fast cube-free path (see ``_token_features_sum_pool``) was
written to fix this; this bench is what we now use to validate the cost.

Reports per-step wall time and end-to-end slowdown for the tokenized path
relative to legacy. Use ``--device cuda`` for ROCm/CUDA runs.

Run:
    .venv\\Scripts\\python.exe scripts\\bench_token_path.py [--device cuda] [--batch 512] [--iters 50]
"""
from __future__ import annotations

import argparse
import time

import torch

from src.config import DataConfig, ModelConfig
from src.encoding.action_encoder import get_action_space_size
from src.encoding.state_encoder import get_state_size
from src.ppo.ppo_actor_critic import PPOActorCritic


def _bench_step(
    model: PPOActorCritic,
    x: torch.Tensor,
    iters: int,
    warmup: int = 10,
    forward_only: bool = False,
) -> float:
    """Time ``iters`` training-shaped steps. Returns total seconds."""
    opt = torch.optim.SGD(model.parameters(), lr=1e-4) if not forward_only else None
    model.train(not forward_only)

    is_cuda = x.is_cuda

    def _one():
        if forward_only:
            with torch.no_grad():
                model(x)
            return
        opt.zero_grad(set_to_none=True)
        logits, value = model(x)
        loss = logits.pow(2).mean() + value.pow(2).mean()
        loss.backward()
        opt.step()

    for _ in range(warmup):
        _one()
    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _one()
    if is_cuda:
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--batch", type=int, default=512,
        help="Minibatch size. Default 512 matches the PPO update size.",
    )
    ap.add_argument(
        "--iters", type=int, default=50,
        help="Step count to time (after warmup).",
    )
    ap.add_argument(
        "--pool", default="sum", choices=["sum", "attention"],
        help="Pool type. Only 'sum' uses the fast token-features path; "
             "'attention' falls back to the cube path.",
    )
    ap.add_argument(
        "--forward-only", action="store_true",
        help="Skip backward + optimizer step. Use to isolate forward cost.",
    )
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

    t_legacy = _bench_step(legacy, x, args.iters, forward_only=args.forward_only)
    t_tokens = _bench_step(tokenized, x, args.iters, forward_only=args.forward_only)

    step_legacy_ms = t_legacy / args.iters * 1e3
    step_tokens_ms = t_tokens / args.iters * 1e3
    mode = "forward-only" if args.forward_only else "fwd+bwd+opt"
    print(
        f"Device={args.device} batch={args.batch} pool={args.pool} "
        f"num_cards={num_cards} iters={args.iters} mode={mode}"
    )
    print(f"  legacy  : {step_legacy_ms:8.2f} ms/step  ({args.batch / step_legacy_ms * 1e3:7.0f} samples/s)")
    print(f"  tokens  : {step_tokens_ms:8.2f} ms/step  ({args.batch / step_tokens_ms * 1e3:7.0f} samples/s)")
    print(f"  slowdown: {step_tokens_ms / step_legacy_ms:.2f}x")


if __name__ == "__main__":
    main()
