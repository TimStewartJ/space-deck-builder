import argparse
import glob
import os
import torch
from pathlib import Path
from src.cards.loader import load_trade_deck_cards
from src.engine.game import Game
from src.ai.ppo_agent import PPOAgent
from src.ai.random_agent import RandomAgent
from src.utils.logger import log, set_verbose

def get_latest_model(models_dir="models"):
    model_files = glob.glob(os.path.join(models_dir, "ppo_agent_*.pth"))
    if not model_files:
        return None
    model_files.sort(key=os.path.getmtime, reverse=True)
    return model_files[0]

def main():
    parser = argparse.ArgumentParser("PPO Game Simulator")
    parser.add_argument("--cards-path", type=str, default="data/cards.csv")
    parser.add_argument("--model1", type=str, default=None, help="Path to PPO model for player 1 (default: latest model)")
    parser.add_argument("--model2", type=str, default=None, help="Path to PPO model for player 2 (default: random agent)")
    parser.add_argument("--player2-random", action="store_true", default=True, help="Set player 2 as random agent (default: True)")
    parser.add_argument("--games", type=int, default=50, help="Number of games to simulate")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    set_verbose(False)
    cards = load_trade_deck_cards(args.cards_path, filter_sets=["Core Set"], log_cards=False)
    names = [c.name for c in cards]
    names = list(dict.fromkeys(names)) + ["Scout", "Viper", "Explorer"]

    # Player 1: PPO agent
    model1_path = args.model1 or get_latest_model()
    if not model1_path:
        raise RuntimeError("No PPO model found for player 1.")
    log(f"Loading PPO model for player 1 from {model1_path}")
    agent1 = PPOAgent("PPO_1", names, device=args.device, model_path=model1_path)

    # Player 2: PPO agent or random agent
    if args.player2_random or not args.model2:
        agent2 = RandomAgent("Rand")
        log("Player 2 set to RandomAgent.")
    else:
        log(f"Loading PPO model for player 2 from {args.model2}")
        agent2 = PPOAgent("PPO_2", names, device=args.device, model_path=args.model2)

    wins1, wins2 = 0, 0
    for i in range(args.games):
        game = Game(cards)
        game.add_player(agent1.name, agent1)
        game.add_player(agent2.name, agent2)
        game.start_game()
        done = False
        while not done:
            done = game.step()
        winner = game.get_winner()
        if winner == agent1.name:
            wins1 += 1
        elif winner == agent2.name:
            wins2 += 1
    print(f"\nResults after {args.games} games:")
    print(f"{agent1.name} wins: {wins1}")
    print(f"{agent2.name} wins: {wins2}")
    print(f"Win rate: {agent1.name}: {wins1/args.games:.2%}, {agent2.name}: {wins2/args.games:.2%}")

if __name__ == "__main__":
    main()
