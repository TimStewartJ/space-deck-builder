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

if TYPE_CHECKING:
    from src.engine.player import Player
    from src.engine.actions import Action

class Trainer:
    def __init__(self, episodes, episode_sample_size=100, lambda_param=0.999, cards_path='data/cards.csv'):
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
        self.neural_agent = NeuralAgent("NeuralAgent", learning_rate=0.001, cards=self.card_names, exploration_decay_rate=0.999)
        self.opponent_agent = RandomAgent("RandomAgent")  # Choose an opponent type
        
    def calculate_reward(self, game: 'Game', player: 'Player', action_taken: 'Action | None', learner_name: str = "NeuralAgent") -> float:
        """Calculate reward for the current game state"""
        from src.engine.actions import ActionType

        if action_taken is None:
            return 0.0

        # Basic reward for winning/losing
        if game.is_game_over:
            return 1 if game.get_winner() == learner_name else -1

        if player.name != learner_name:
            return 0.0

        return 0.0
    
    def train(self):
        set_verbose(False)  # Disable verbose logging for training

        training_start_time = datetime.now()

        aggregate_stats = AggregateStats()
        player1Name = "NeuralAgent"
        player2Name = "Opponent"
        aggregate_stats.reset(player1_name=player1Name, player2_name=player2Name)

        for episode in range(self.episodes):
            log(f"Episode {episode+1}/{self.episodes}")
            
            # Setup game
            game = Game(self.cards)
            
            # Add players
            player1 = game.add_player(player1Name)
            player1.agent = self.neural_agent
            
            player2 = game.add_player(player2Name)
            player2.agent = self.opponent_agent

            game.start_game()

            # Main training loop
            current_episode_states = []
            current_episode_actions = []
            current_episode_rewards = []
            current_episode_next_states = []
            current_episode_dones = []

            while not game.is_game_over:
                # Store state before action
                current_player = game.current_player
                is_current_player_training = current_player.name == player1Name
                
                state = encode_state(game, is_current_player_training=is_current_player_training, cards=self.card_names)
                
                # Agent makes a decision and updates game state
                action = game.next_step()

                if is_current_player_training:
                    # Recalculate current player
                    is_current_player_training = current_player.name == player1Name

                    # Calculate reward and remember experience
                    reward = self.calculate_reward(game, current_player, action)
                    next_state = encode_state(game, is_current_player_training=is_current_player_training, cards=self.card_names)
                    done = game.is_game_over
                    
                    # Store experience
                    current_episode_states.append(state)
                    current_episode_actions.append(action)
                    current_episode_rewards.append(reward)
                    current_episode_next_states.append(next_state)
                    current_episode_dones.append(done)
                
            # When game is over, store the complete episode
            if len(current_episode_states) > 0:
                for i in range(len(current_episode_states)):
                    self.neural_agent.remember(
                        current_episode_states[i],
                        current_episode_actions[i],
                        current_episode_rewards[i],
                        current_episode_next_states[i],
                        current_episode_dones[i]
                    )
                self.neural_agent.finish_remembering_episode()
                
            # Train the network
            if episode % self.episode_sample_size == 0:
                log(f"Training neural agent at episode {episode}...")
                # keep track of time taken to train
                start_time = datetime.now()
                self.neural_agent.train(lambda_param=self.lambda_param, episode_sample_size=self.episode_sample_size)
                end_time = datetime.now()
                log(f"Training took {(end_time - start_time).total_seconds():.2f} seconds.")
            
            aggregate_stats.update(game.stats, game.get_winner())
            log(game.stats.get_summary())

            # Save model periodically
            if episode % (self.episodes/10) == 0:
                log(f"Saving model at episode {episode}")
                log(aggregate_stats.get_summary())
                torch.save(self.neural_agent.model.state_dict(), f"models/neural_agent_{episode}.pth")

        training_time = (datetime.now() - training_start_time).total_seconds()
        log(f"Training completed in {training_time:.2f} seconds.")
        # Log the average time per episode
        log(f"Average time per episode: {training_time / self.episodes:.2f} seconds.")

        # Save final model
        log(aggregate_stats.get_summary())
        torch.save(self.neural_agent.model.state_dict(), "models/neural_agent_final.pth")
        
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
    args = parser.parse_args()

    trainer = Trainer(episodes=args.episodes)
    trainer.train()
