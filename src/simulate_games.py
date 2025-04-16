import argparse
from datetime import datetime
from pathlib import Path
import torch
from src.cards.loader import load_trade_deck_cards
from src.ai.neural_agent import NeuralAgent
from src.ai.random_agent import RandomAgent
from src.engine.game import Game
from src.utils.logger import log, set_verbose
from src.utils.decay_rate_calculator import calculate_exploration_decay_rate

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
    card_names += ["Viper", "Scout"]

    # Initialize the agents
    neural_agent = NeuralAgent("NeuralAgent",
                               learning_rate=0.001,
                               cards=card_names,
                               model_file_path=args.model,
                               min_exploration_rate=0.1,
                               initial_exploration_rate=1.0)
    random_agent = RandomAgent("RandomAgent")

    wins = {neural_agent.name: 0, random_agent.name: 0}
    game_durations = []

    for i in range(args.games):
        start_time = datetime.now()
        
        # Create a new game with the available cards.
        game = Game(cards)
        
        # Add players and assign agents
        player1 = game.add_player(neural_agent.name)
        player1.agent = neural_agent
        player2 = game.add_player(random_agent.name)
        player2.agent = random_agent
        
        # Play the game until it is over and get the winner's name.
        winner = game.play()
        
        # Record the result
        wins[winner] += 1
        duration = (datetime.now() - start_time).total_seconds()
        game_durations.append(duration)
        log(f"Game {i+1}: Winner - {winner} (Duration: {duration:.2f} seconds)")

    total_time = sum(game_durations)
    average_time = total_time / args.games if args.games else 0
    log(f"Simulated {args.games} games.")
    log(f"Results: {wins}")
    log(f"Average game duration: {average_time:.2f} seconds.")

if __name__ == "__main__":
    main()