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
from src.nn.memory_data import MemoryData
import copy

if TYPE_CHECKING:
    from src.engine.player import Player
    from src.engine.actions import Action

class Trainer:
    def __init__(self, episodes, episode_batch_size=100, lambda_param=0.999, 
                 cards_path='data/cards.csv', model_file_path=None, 
                 min_exploration_rate=0.1, initial_exploration_rate=1.0,
                 sample_size=None, memory_batches=None, self_play=False, self_play_update_batches=1):
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
        self.card_names += ["Scout", "Viper", "Explorer"]
        exploration_decay_rate = calculate_exploration_decay_rate(episodes // episode_batch_size, min_exploration_rate, 0.8, initial_rate=initial_exploration_rate)
        self.neural_agent = NeuralAgent("NeuralAgent", learning_rate=0.001, cards=self.card_names, 
                                        exploration_decay_rate=exploration_decay_rate, model_file_path=model_file_path, 
                                        min_exploration_rate=min_exploration_rate, initial_exploration_rate=initial_exploration_rate)
        self.opponent_agent = RandomAgent("RandomAgent")
        # Determine how many experiences to sample each training batch
        self.experiences_sample_size = sample_size if sample_size is not None else self.episode_batch_size * 100
        # Maximum number of batches worth of experiences to keep in memory
        self.memory_batches = memory_batches
        self.self_play = self_play  # store self-play flag
        self.self_play_update_batches = self_play_update_batches  # batches between opponent updates
    
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
        log(f"There will be {self.episodes/self.episode_batch_size} batches of {self.episode_batch_size} episodes.")

        thread_count = 4

        all_batch_winners = []
        epsilon_progression = []

        for batch_index, batch_start in enumerate(range(0, self.episodes, self.episode_batch_size)):
            batch_end = min(batch_start + self.episode_batch_size, self.episodes)
            log(f"Processing batch {batch_index} which includes episodes {batch_start + 1} to {batch_end}")

            training_start_index = 0 if self.memory_batches is None else self.memory_batches

            if self.self_play and (batch_index % self.self_play_update_batches == 0) and batch_index > training_start_index:
                # use previous network as opponent at configured interval
                log(f"Updating opponent agent for batch {batch_index} to the previous agent...")
                self.opponent_agent = copy.deepcopy(self.neural_agent)
                self.opponent_agent.exploration_rate = 0.0

            # Count experiences in this batch for memory trimming
            batch_experiences_count = 0

            batch_start_time = datetime.now()
            batch_winners = {
                player1Name: 0,
                player2Name: 0
            }
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
                        batch_experiences_count += len(experiences)

                        # Store the collected experiences from the worker
                        if random.random() < episode_save_chance:
                            all_episodes.append(experiences)

                        # Update aggregate statistics using results from the worker
                        aggregate_stats.update(game_stats, winner)
                        batch_winners[winner] += 1

            log(f"Batch winners: {batch_winners}")
            all_batch_winners.append(batch_winners)
            batch_duration = (datetime.now() - batch_start_time).total_seconds()
            log(
                f"Batch {batch_index} took {batch_duration:.4f} seconds and "
                f"{batch_experiences_count} experiences, average of "
                f"{batch_duration / self.episode_batch_size:.4f} seconds and "
                f"{batch_experiences_count / self.episode_batch_size:.4f} experiences per episode."
            )
            epsilon_progression.append(
                self.neural_agent.exploration_rate
            )

            # Trim experiences to only keep the last N batches worth
            if self.memory_batches is not None:
                max_exp = self.memory_batches * batch_experiences_count
                if len(all_experiences) > max_exp:
                    del all_experiences[:len(all_experiences) - max_exp]

            if batch_index > training_start_index:
                # Train the network after completing the batch
                log(f"Training neural agent for batch {batch_start + 1} to {batch_end}...")
                start_time = datetime.now()
                experiences_to_train = random.sample(all_experiences, min(len(all_experiences), self.experiences_sample_size))
                self.neural_agent.train(lambda_param=self.lambda_param, experiences=experiences_to_train)
                end_time = datetime.now()
                log(f"Training took {(end_time - start_time).total_seconds():.4f} seconds.")

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
            memory_data = MemoryData(
                episodes=all_episodes,
                batch_size=self.episode_batch_size,
                lambda_param=self.lambda_param,
                exploration_decay_rate=self.neural_agent.exploration_decay_rate,
                batch_winners=all_batch_winners,
                epsilon_progression=epsilon_progression
            )
            pickle.dump(memory_data, f)
        log(f"Memory saved in {(datetime.now() - memory_saving_time).total_seconds():.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Neural Agent for Space Deck Builder.")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes to train for.")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of episodes to batch before training.")
    parser.add_argument("--lambda", type=float, default=0.999, dest="lambda_param", help="Lambda parameter for TD(λ) learning.")
    parser.add_argument("--min-exploration", type=float, default=0.1, help="Minimum exploration rate.")
    parser.add_argument("--initial-exploration", type=float, default=1.0, help="Initial exploration rate.")
    parser.add_argument("--model", type=str, default=None, help="Path to the model file.")
    parser.add_argument("--sample-size", type=int, default=None, help="Number of experiences to sample for training per batch.")
    parser.add_argument("--memory-batches", type=int, default=None, help="Number of batches worth of experiences to keep in memory.")
    parser.add_argument("--self-play", action="store_true", dest="self_play", help="Train against previous versions of the neural agent.")
    parser.add_argument("--self-play-update-batches", type=int, default=1, dest="self_play_update_batches", help="Number of batches between opponent agent updates in self-play.")
    args = parser.parse_args()

    trainer = Trainer(
        episodes=args.episodes,
        episode_batch_size=args.batch_size,
        lambda_param=args.lambda_param,
        model_file_path=args.model,
        min_exploration_rate=args.min_exploration,
        initial_exploration_rate=args.initial_exploration,
        sample_size=args.sample_size,
        memory_batches=args.memory_batches,
        self_play=args.self_play,
        self_play_update_batches=args.self_play_update_batches
    )
    trainer.train()
