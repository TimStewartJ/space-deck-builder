import argparse
from datetime import datetime
from pathlib import Path
import pickle
import random
from typing import TYPE_CHECKING
import torch
import numpy as np
from src.nn.experience import Experience
from src.nn.save_probability import calculate_save_probability
from src.nn.state_encoder import encode_state
from src.engine.aggregate_stats import AggregateStats
from src.ai.random_agent import RandomAgent
from src.engine.game import Game
from src.cards.loader import load_trade_deck_cards
from src.ai.neural_agent import NeuralAgent
from src.utils.logger import log, set_verbose
from src.nn.parallel_worker import worker_run_episode
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from src.utils.decay_rate_calculator import calculate_exploration_decay_rate

if TYPE_CHECKING:
    from src.engine.player import Player
    from src.engine.actions import Action

class Trainer:
    def __init__(self, episodes, episode_batch_size=100, lambda_param=0.999, 
                 cards_path='data/cards.csv', model_file_path=None, 
                 min_exploration_rate=0.1, initial_exploration_rate=1.0):
        """Initialize the Trainer with the specified parameters"""
        self.episodes = episodes
        self.episode_batch_size = episode_batch_size
        self.lambda_param = lambda_param
        # Initialize cards to a list of card names, extract name from list of Card objects
        self.cards = load_trade_deck_cards(cards_path, filter_sets=["Core Set"])
        self.card_names = [card.name for card in self.cards]
        # Remove duplicates from card list
        self.card_names = list(dict.fromkeys(self.card_names))
        # Add starter cards to the list
        self.card_names += ["Scout", "Viper"]
        exploration_decay_rate = calculate_exploration_decay_rate(episodes // episode_batch_size, min_exploration_rate, 0.8)
        self.neural_agent = NeuralAgent("NeuralAgent", learning_rate=0.001, cards=self.card_names, 
                                        exploration_decay_rate=exploration_decay_rate, model_file_path=model_file_path, 
                                        min_exploration_rate=min_exploration_rate, initial_exploration_rate=initial_exploration_rate)
        self.opponent_agent = RandomAgent("RandomAgent")
    
    def train(self):
        set_verbose(False)  # Disable verbose logging for training

        training_start_time = datetime.now()

        aggregate_stats = AggregateStats()
        player1Name = self.neural_agent.name # Use agent's name
        player2Name = self.opponent_agent.name # Use agent's name
        aggregate_stats.reset(player1_name=player1Name, player2_name=player2Name)

        all_episodes: list[list[Experience]] = []
        all_experiences: list[Experience] = []
        episode_save_chance = calculate_save_probability(self.episodes, 10000)
        log(f"Episode save chance: {episode_save_chance*100:.2f}%")

        thread_count = 4

        for batch_start in range(0, self.episodes, self.episode_batch_size):
            batch_end = min(batch_start + self.episode_batch_size, self.episodes)
            log(f"Processing batch {batch_start + 1} to {batch_end}")

            batch_start_time = datetime.now()
            # Run episodes in parallel
            with ThreadPoolExecutor() as executor: 
                futures = [
                    executor.submit(
                        worker_run_episode,
                        self.episode_batch_size//thread_count,
                        self.cards,
                        self.card_names,
                        self.neural_agent,
                        self.opponent_agent,
                        player1Name,
                        player2Name,
                        self.lambda_param
                    ) for _ in range(thread_count)
                ]

                for future in futures:
                    experiences_list = future.result()

                    for experience in experiences_list:
                        experiences, game_stats, winner = experience

                        all_experiences.extend(experiences)

                        # Store the collected experiences from the worker
                        if random.random() < episode_save_chance:
                            all_episodes.append(experiences)

                        # Update aggregate statistics using results from the worker
                        aggregate_stats.update(game_stats, winner)
                        log(f"{winner} wins!")
            batch_duration = (datetime.now() - batch_start_time).total_seconds()
            log(f"Batch {batch_start + 1} took {batch_duration:.4f} seconds, average of {batch_duration / self.episode_batch_size:.4f} seconds per episode.")

            # Train the network after completing the batch
            log(f"Training neural agent for batch {batch_start + 1} to {batch_end}...")
            start_time = datetime.now()
            experiences_to_train = random.sample(all_experiences, min(len(all_experiences), self.episode_batch_size*100))
            self.neural_agent.train(lambda_param=self.lambda_param, experiences=experiences_to_train)
            end_time = datetime.now()
            log(f"Training took {(end_time - start_time).total_seconds():.4f} seconds.")

            # Save model periodically
            if (batch_end) % (self.episodes // 10) == 0 and batch_end > 0:
                log(f"Saving model at episode {batch_end}")
                log(aggregate_stats.get_summary())
                torch.save(self.neural_agent.model.state_dict(), f"models/neural_agent_{batch_end}.pth")

        training_time = (datetime.now() - training_start_time).total_seconds()
        log(f"Training completed in {training_time:.2f} seconds.")
        # Log the average time per episode
        log(f"Average time per episode: {training_time / len(all_episodes):.4f} seconds.")

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
        # Save all the episodes to a file
        with open(memory_file, "wb") as f:
            pickle.dump(
                {"episodes": all_episodes,
                 "batch_size": self.episode_batch_size,
                 "lambda_param": self.lambda_param,
                 "exploration_decay_rate": self.neural_agent.exploration_decay_rate,
                 }
                 , f)
        log(f"Memory saved in {(datetime.now() - memory_saving_time).total_seconds():.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Neural Agent for Space Deck Builder.")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes to train for.")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of episodes to batch before training.")
    parser.add_argument("--lambda", type=float, default=0.999, dest="lambda_param", help="Lambda parameter for TD(λ) learning.")
    parser.add_argument("--min-exploration", type=float, default=0.1, help="Minimum exploration rate.")
    parser.add_argument("--initial-exploration", type=float, default=1.0, help="Initial exploration rate.")
    parser.add_argument("--model", type=str, default=None, help="Path to the model file.")
    args = parser.parse_args()

    trainer = Trainer(
        episodes=args.episodes,
        episode_batch_size=args.batch_size,
        lambda_param=args.lambda_param,
        model_file_path=args.model,
        min_exploration_rate=args.min_exploration,
        initial_exploration_rate=args.initial_exploration,
    )
    trainer.train()
