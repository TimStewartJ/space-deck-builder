import argparse
from datetime import datetime
from typing import TYPE_CHECKING
import torch
import numpy as np
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
    def __init__(self, episodes=10000, batch_size=128, episode_sample_size=32):
        self.episodes = episodes
        self.episode_sample_size = episode_sample_size
        self.batch_size = batch_size
        self.cards = load_trade_deck_cards('data/cards.csv', filter_sets=["Core Set"])
        self.neural_agent = NeuralAgent("NeuralAgent", learning_rate=0.001, look_ahead_steps=10, cards=self.cards)
        self.opponent_agent = RandomAgent("RandomAgent")  # Choose an opponent type
        
    def calculate_reward(self, game: 'Game', player: 'Player', action_taken: 'Action', learner_name: str = "NeuralAgent") -> float:
        """Calculate reward for the current game state"""
        from src.engine.actions import ActionType
        # Basic reward for winning/losing
        if game.is_game_over:
            return 1000 if game.get_winner() == learner_name else -1000

        if action_taken.type == ActionType.END_TURN and player.name == learner_name:
            return -100  # Small penalty for ending turn without action

        reward = 0

        # Basic resource rewards
        reward += player.trade * 2  # Trading is important for deck building
        reward += player.combat * 1.5  # Combat allows attacking
        
        # Deck improvement rewards
        hand_value = sum(card.cost for card in player.hand)
        reward += hand_value * 0.5  # Reward for having valuable cards in hand
        
        # Base establishment rewards
        base_count = len(player.bases)
        reward += base_count * 10  # Bases provide lasting value
        
        # Health advantage reward
        opponent = game.get_opponent(player)
        health_advantage = player.health - opponent.health
        reward += health_advantage * 0.5

        # Reward for card synergies
        faction_counts = {"Blob": 0, "Trade Federation": 0, "Machine Cult": 0, "Star Empire": 0}
        all_cards = player.hand + player.bases + player.discard_pile + player.deck
        for card in all_cards:
            if card.faction in faction_counts:
                faction_counts[card.faction] += 1
        
        # Reward faction synergy potential
        dominant_faction = max(faction_counts, key=faction_counts.get)
        reward += faction_counts[dominant_faction] * 2

        # Reward for high average cost of all cards
        average_cost = np.mean([card.cost for card in all_cards]) if all_cards else 0
        reward += average_cost * 1.5  # Higher cost cards can be more powerful

        # Negate the reward if the current player is not the neural agent
        if player.name != learner_name:
            reward = -reward
        return reward
    
    def train(self):
        set_verbose(False)  # Disable verbose logging for training

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
                
                state = self.neural_agent.encode_state(game)
                current_episode_states.append(state)
                
                # Agent makes a decision and updates game state
                action = game.next_step()
                
                # Calculate reward and remember experience
                reward = self.calculate_reward(game, current_player, action)
                next_state = self.neural_agent.encode_state(game)
                done = game.is_game_over
                
                # Store experience
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
                
            # Train the network
            if episode % (self.episodes/100) == 0:
                log(f"Training neural agent at episode {episode}...")
                # keep track of time taken to train
                start_time = datetime.now()
                self.neural_agent.train(self.batch_size, lambda_param=0.7, episode_sample_size=self.episode_sample_size)
                end_time = datetime.now()
                log(f"Training took {(end_time - start_time).total_seconds():.2f} seconds.")
            
            aggregate_stats.update(game.stats, game.get_winner())
            log(game.stats.get_summary())

            # Save model periodically
            if episode % (self.episodes/10) == 0:
                log(f"Saving model at episode {episode}")
                log(aggregate_stats.get_summary())
                torch.save(self.neural_agent.model.state_dict(), f"models/neural_agent_{episode}.pth")
                
        # Save final model
        log(aggregate_stats.get_summary())
        torch.save(self.neural_agent.model.state_dict(), "models/neural_agent_final.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Neural Agent for Space Deck Builder.")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes to train for.")
    args = parser.parse_args()

    trainer = Trainer(episodes=args.episodes)
    trainer.train()
