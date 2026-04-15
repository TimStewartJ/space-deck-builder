"""Benchmark script for PPO training throughput."""
import time
import random
import torch
from src.config import DataConfig, PPOConfig
from src.cards.loader import load_trade_deck_cards
from src.engine.game import Game
from src.ai.ppo_agent import PPOAgent
from src.ai.random_agent import RandomAgent
from src.encoding.action_encoder import get_action_space_size
from src.ppo.ppo_actor_critic import PPOActorCritic
from src.encoding.state_encoder import get_state_size
from src.ppo.batch_runner import BatchRunner
from src.utils.logger import set_disabled

def run_single_episode(agent, opponent, cards):
    game = Game(cards)
    game.add_player(agent.name, agent)
    game.add_player(opponent.name, opponent)
    game.start_game()
    steps = 0
    while not game.is_game_over:
        game.step()
        steps += 1
    winner = game.get_winner()
    return steps, winner == agent.name

def benchmark_sequential(num_episodes, card_names, cards, device):
    """Original sequential benchmark (one game at a time)."""
    agent = PPOAgent("PPO", card_names, device=device, main_device=device, simulation_device=device)
    opponent = RandomAgent("Rand")

    total_steps = 0
    wins = 0

    start = time.perf_counter()
    for ep in range(num_episodes):
        agent.clear_buffers()
        steps, won = run_single_episode(agent, opponent, cards)
        total_steps += steps
        if won:
            wins += 1
    elapsed = time.perf_counter() - start
    return total_steps, wins, elapsed

def benchmark_batched(num_episodes, card_names, cards, device):
    """Batched benchmark using BatchRunner."""
    state_dim = get_state_size(card_names)
    action_dim = get_action_space_size(card_names)
    model = PPOActorCritic(state_dim, action_dim, len(card_names)).to(device)

    runner = BatchRunner(
        model=model,
        card_names=card_names,
        cards=cards,
        action_dim=action_dim,
        device=device,
        num_concurrent=min(num_episodes, 64),
    )

    start = time.perf_counter()
    states, actions, old_lp, returns, advs, _masks = runner.run_episodes(num_episodes)
    elapsed = time.perf_counter() - start
    total_steps = states.shape[0]
    return total_steps, None, elapsed

def benchmark_parallel(num_episodes, card_names, cards, device, num_workers=4):
    """Multi-process batched benchmark."""
    from src.ppo.batch_runner import run_episodes_parallel
    state_dim = get_state_size(card_names)
    action_dim = get_action_space_size(card_names)
    model = PPOActorCritic(state_dim, action_dim, len(card_names)).to(device)

    start = time.perf_counter()
    states, actions, old_lp, returns, advs, _masks = run_episodes_parallel(
        model=model, card_names=card_names, cards=cards,
        action_dim=action_dim, num_episodes=num_episodes,
        num_workers=num_workers,
        games_per_worker=max(1, min(num_episodes // num_workers, 32)),
        device=device,
    )
    elapsed = time.perf_counter() - start
    total_steps = states.shape[0]
    return total_steps, None, elapsed

def benchmark(num_episodes=64, device="cpu", mode="both", workers=4):
    set_disabled(True)
    data_cfg = DataConfig()
    cards = data_cfg.load_cards()
    card_names = data_cfg.get_card_names(cards)

    if mode in ("sequential", "both"):
        total_steps, wins, elapsed = benchmark_sequential(num_episodes, card_names, cards, device)
        print(f"=== Sequential Benchmark ===")
        print(f"Episodes:      {num_episodes}")
        print(f"Total steps:   {total_steps}")
        print(f"Wall time:     {elapsed:.2f}s")
        print(f"Steps/sec:     {total_steps / elapsed:.1f}")
        print(f"Episodes/sec:  {num_episodes / elapsed:.2f}")
        print(f"Win rate:      {wins}/{num_episodes} ({wins/num_episodes:.1%})")
        print()

    if mode in ("batched", "both"):
        total_steps, _, elapsed = benchmark_batched(num_episodes, card_names, cards, device)
        print(f"=== Batched (1 process) ===")
        print(f"Episodes:      {num_episodes}")
        print(f"Total steps:   {total_steps}")
        print(f"Wall time:     {elapsed:.2f}s")
        print(f"Steps/sec:     {total_steps / elapsed:.1f}")
        print(f"Episodes/sec:  {num_episodes / elapsed:.2f}")
        print(f"Device:        {device}")
        print()

    if mode in ("parallel", "both"):
        total_steps, _, elapsed = benchmark_parallel(num_episodes, card_names, cards, torch.device("cpu"), workers)
        print(f"=== Parallel ({workers} workers) ===")
        print(f"Episodes:      {num_episodes}")
        print(f"Total steps:   {total_steps}")
        print(f"Wall time:     {elapsed:.2f}s")
        print(f"Steps/sec:     {total_steps / elapsed:.1f}")
        print(f"Episodes/sec:  {num_episodes / elapsed:.2f}")
        print(f"Device:        cpu (per-worker)")
