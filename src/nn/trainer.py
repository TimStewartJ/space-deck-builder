import argparse
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
import torch
import numpy as np
from src.nn.state_encoder import encode_state
from src.engine.aggregate_stats import AggregateStats
from src.ai.random_agent import RandomAgent
from src.engine.game import Game
from src.cards.loader import load_trade_deck_cards
from src.ai.neural_agent import NeuralAgent
from src.utils.logger import log, set_verbose
# Import the worker function
from src.nn.parallel_worker import worker_run_episode

if TYPE_CHECKING:
    from src.engine.player import Player
    from src.engine.actions import Action

class Trainer:
    def __init__(self, episodes, episode_sample_size=100, lambda_param=0.999, 
                 exploration_decay_rate=0.999, cards_path='data/cards.csv', model_file_path=None):
        """Initialize the Trainer with the specified parameters"""
        self.episodes = episodes
        self.episode_sample_size = episode_sample_size
        self.lambda_param = lambda_param
        # Initialize cards to a list of card names, extract name from list of Card objects
        self.cards = load_trade_deck_cards(cards_path, filter_sets=["Core Set"])
        self.card_names = [card.name for card in self.cards]
        # Remove duplicates from card list
        self.card_names = list(dict.fromkeys(self.card_names))
        # Add starter cards to the list
        self.card_names += ["Viper", "Scout"]
        self.neural_agent = NeuralAgent("NeuralAgent", learning_rate=0.001, cards=self.card_names, exploration_decay_rate=exploration_decay_rate, model_file_path=model_file_path)
        self.opponent_agent = RandomAgent("RandomAgent")  # Choose an opponent type
          # Reward calculation is now handled within the worker_run_episode function
    
    def train(self):
        set_verbose(False)  # Disable verbose logging for training

        training_start_time = datetime.now()

        aggregate_stats = AggregateStats()
        player1Name = self.neural_agent.name # Use agent's name
        player2Name = self.opponent_agent.name # Use agent's name
        aggregate_stats.reset(player1_name=player1Name, player2_name=player2Name)

        for episode in range(self.episodes):
            log(f"Episode {episode+1}/{self.episodes}")
              # Run the episode using the worker function
            experiences, game_stats, winner = worker_run_episode(
                episode, 
                self.cards, 
                self.card_names, 
                self.neural_agent, 
                self.opponent_agent,
                player1Name,
                player2Name
            )

            # Store the collected experiences from the worker
            if len(experiences["states"]) > 0:
                for i in range(len(experiences["states"])):
                    self.neural_agent.remember(
                        experiences["states"][i],
                        experiences["actions"][i],
                        experiences["rewards"][i],
                        experiences["next_states"][i],
                        experiences["dones"][i]
                    )
                self.neural_agent.finish_remembering_episode()

            # Update aggregate statistics using results from the worker
            aggregate_stats.update(game_stats, winner)
            log(f"Game took {game_stats.get_game_duration():.4f} seconds.")
                
            # Train the network
            if (episode + 1) % self.episode_sample_size == 0 and episode > 0: # Train after sample_size episodes completed
                log(f"Training neural agent at episode {episode}...")
                # keep track of time taken to train
                start_time = datetime.now()
                self.neural_agent.train(lambda_param=self.lambda_param, episode_sample_size=self.episode_sample_size)
                end_time = datetime.now()
                log(f"Training took {(end_time - start_time).total_seconds():.4f} seconds.")

            # Save model periodically
            if (episode + 1) % (self.episodes // 10) == 0 and episode > 0: # Save after certain milestones
                log(f"Saving model at episode {episode}")
                log(aggregate_stats.get_summary())
                torch.save(self.neural_agent.model.state_dict(), f"models/neural_agent_{episode}.pth")

        training_time = (datetime.now() - training_start_time).total_seconds()
        log(f"Training completed in {training_time:.2f} seconds.")
        # Log the average time per episode
        log(f"Average time per episode: {training_time / self.episodes:.4f} seconds.")

        # Save final model
        log(aggregate_stats.get_summary())
        current_date_time = datetime.now().strftime("%m%d_%H%M")
        torch.save(self.neural_agent.model.state_dict(), f"models/neural_agent_final_{current_date_time}_{self.episodes}.pth")
        
        # Get the user's home directory
        home_dir = Path.home()

        # Construct the path to the Downloads folder
        downloads_dir = home_dir / "Downloads"
        # Define the path to the memory file
        # Add number of episodes to the filename
        memory_file_name = f"memory_{self.episodes}.pkl"
        memory_file = downloads_dir / memory_file_name

        memory_saving_time = datetime.now()
        log(f"Saving memory at {memory_saving_time} to {memory_file}")
        self.neural_agent.save_memory(memory_file=str(memory_file))
        log(f"Memory saved in {(datetime.now() - memory_saving_time).total_seconds():.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Neural Agent for Space Deck Builder.")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes to train for.")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of episodes to sample before training.")
    parser.add_argument("--lambda", type=float, default=0.999, dest="lambda_param", help="Lambda parameter for TD(λ) learning.")
    parser.add_argument("--decay", type=float, default=0.999, help="Exploration decay rate.")
    parser.add_argument("--model", type=str, default=None, help="Path to the model file.")
    args = parser.parse_args()

    trainer = Trainer(
        episodes=args.episodes,
        episode_sample_size=args.sample_size,
        lambda_param=args.lambda_param,
        exploration_decay_rate=args.decay,
        model_file_path=args.model
    )
    trainer.train()
