import argparse
from datetime import datetime
from pathlib import Path
import torch
import pickle
from src.cards.loader import load_trade_deck_cards
from src.ai.neural_agent import NeuralAgent
from src.ai.random_agent import RandomAgent
from src.engine.game import Game
from src.utils.logger import log, set_verbose
from src.utils.decay_rate_calculator import calculate_exploration_decay_rate
from src.nn.parallel_worker import worker_run_episode

def main():
    parser = argparse.ArgumentParser(description="Simulate games using the Neural Agent.")
    parser.add_argument("--games", type=int, default=10, help="Number of games to simulate.")
    parser.add_argument("--model", type=str, required=True, help="Path to the neural agent model file.")
    parser.add_argument("--cards", type=str, default="data/cards.csv", help="Path to the cards csv file.")
    args = parser.parse_args()

    # Disable verbose logging during simulation
    set_verbose(False)

    # Load cards and extract card names
    cards = load_trade_deck_cards(args.cards, filter_sets=["Core Set"])
    card_names = [card.name for card in cards]
    # Remove duplicates and add starter cards
    card_names = list(dict.fromkeys(card_names))
    card_names += ["Scout", "Viper"]

    # Initialize the agents
    neural_agent = NeuralAgent("NeuralAgent",
                               learning_rate=0.001,
                               cards=card_names,
                               model_file_path=args.model,
                               min_exploration_rate=0.0,
                               initial_exploration_rate=0.0)
    random_agent = RandomAgent("RandomAgent")

    lambda_param = 1.0  # Discount factor for reward updates
    # Run episodes using the parallel worker
    experiences_list = worker_run_episode(
        episode_count=args.games,
        cards=cards,
        card_names=card_names,
        first_agent=neural_agent,
        second_agent=random_agent,
        first_agent_name=neural_agent.name,
        second_agent_name=random_agent.name,
        lambda_param=lambda_param
    )
    # Initialize statistics
    wins = {neural_agent.name: 0, random_agent.name: 0}
    game_durations = []
    # Process each episode's results
    for experiences, game_stats, winner in experiences_list:
        wins[winner] += 1
        # GameStats contains start and end times
        game_durations.append(game_stats.get_game_duration())
    # Persist experiences to disk
    output_filename = Path(f"experiences_{args.games}_{datetime.now():%Y%m%d_%H%M%S}.pkl")
    with open(output_filename, 'wb') as f:
        pickle.dump(experiences_list, f)
    log(f"Saved experiences to {output_filename}")

    total_time = sum(game_durations)
    average_time = total_time / args.games if args.games else 0
    log(f"Simulated {args.games} games.")
    log(f"Results: {wins}")
    log(f"Percentage of wins:")
    for agent, win_count in wins.items():
        percentage = (win_count / args.games) * 100 if args.games else 0
        log(f"{agent}: {percentage:.2f}% wins")
    log(f"Average game duration: {average_time:.4f} seconds.")

if __name__ == "__main__":
    main()