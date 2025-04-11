import random
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

class Trainer:
    def __init__(self, episodes=1000, batch_size=64):
        self.episodes = episodes
        self.batch_size = batch_size
        self.neural_agent = NeuralAgent("NeuralAgent")
        self.opponent_agent = RandomAgent("RandomAgent")  # Choose an opponent type
        
    def calculate_reward(self, game: 'Game', player: 'Player'):
        """Calculate reward for the current game state"""
        # Basic reward for winning/losing
        if game.is_game_over:
            return 1000 if game.get_winner() == player.name else -1000

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
        for card in player.hand + player.bases + player.discard_pile + player.deck:
            if card.faction in faction_counts:
                faction_counts[card.faction] += 1
        
        # Reward faction synergy potential
        dominant_faction = max(faction_counts, key=faction_counts.get)
        reward += faction_counts[dominant_faction] * 2
    
        
        # Add more sophisticated reward signals
        return reward
    
    def train(self):
        set_verbose(False)  # Disable verbose logging for training

        cards = load_trade_deck_cards('data/cards.csv', filter_sets=["Core Set"])

        aggregate_stats = AggregateStats()
        player1Name = "NeuralAgent"
        player2Name = "Opponent"
        aggregate_stats.reset(player1_name=player1Name, player2_name=player2Name)

        for episode in range(self.episodes):
            log(f"Episode {episode+1}/{self.episodes}")
            
            # Setup game
            game = Game(cards)
            
            # Add players
            player1 = game.add_player(player1Name)
            player1.agent = self.neural_agent
            
            player2 = game.add_player(player2Name)
            player2.agent = self.opponent_agent

            game.start_game()
            
            # Main training loop
            while not game.is_game_over:
                # Store state before action
                current_player = game.current_player
                
                # Check if the current player is the neural agent
                if current_player.name == player1Name:
                    state = self.neural_agent.encode_state(game)
                
                # Agent makes a decision and updates game state
                action = game.next_step()
                
                # Calculate reward and remember experience
                # if the current player is the neural agent
                if current_player.name == player1Name:
                    reward = self.calculate_reward(game, current_player)
                    next_state = self.neural_agent.encode_state(game)
                    self.neural_agent.remember(state, action, reward, next_state, game.is_game_over)
                
            # Train the network
            if episode % (self.episodes/100) == 0:
                self.neural_agent.train(self.batch_size)
            
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
    trainer = Trainer(episodes=5000, batch_size=64)
    trainer.train()
