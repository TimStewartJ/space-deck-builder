"""Benchmark script for PPO training throughput."""
import time
import random
import torch
from src.cards.loader import load_trade_deck_cards
from src.engine.game import Game
from src.ai.ppo_agent import PPOAgent
from src.ai.random_agent import RandomAgent
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

def benchmark(num_episodes=64, device="cpu"):
    set_disabled(True)
    cards = load_trade_deck_cards("data/cards.csv", filter_sets=["Core Set"], log_cards=False)
    card_names = list(dict.fromkeys(c.name for c in cards)) + ["Scout", "Viper", "Explorer"]

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

    print(f"=== Benchmark Results ===")
    print(f"Episodes:      {num_episodes}")
    print(f"Total steps:   {total_steps}")
    print(f"Wall time:     {elapsed:.2f}s")
    print(f"Steps/sec:     {total_steps / elapsed:.1f}")
    print(f"Episodes/sec:  {num_episodes / elapsed:.2f}")
    print(f"Avg steps/ep:  {total_steps / num_episodes:.1f}")
    print(f"Win rate:      {wins}/{num_episodes} ({wins/num_episodes:.1%})")
    print(f"Device:        {device}")
    return {
        "episodes": num_episodes,
        "total_steps": total_steps,
        "elapsed": elapsed,
        "steps_per_sec": total_steps / elapsed,
        "episodes_per_sec": num_episodes / elapsed,
        "win_rate": wins / num_episodes,
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    benchmark(args.episodes, args.device)
