import pickle
import random
from typing import TYPE_CHECKING
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.nn.state_encoder import CARD_ENCODING_SIZE, STATE_SIZE, encode_state, get_state_size
from src.nn.action_encoder import decode_action, encode_action, get_action_space_size
from src.ai.agent import Agent
from src.engine.actions import get_available_actions, Action, ActionType
from src.utils.logger import log

if TYPE_CHECKING:
    from src.engine.game import Game
    from cards.card import Card
    from cards.effects import CardEffectType

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 768),
            nn.ReLU(),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )
        log(f"Neural Network initialized with input size: {input_size}, output size: {output_size}")
    
    def forward(self, x):
        return self.network(x)

class NeuralAgent(Agent):
    def __init__(self, name, cli_interface=None, learning_rate=0.001, cards: list[str] = [],
                 initial_exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay_rate=0.99): # Added exploration params
        super().__init__(name, cli_interface)
        # Determine device (GPU if available, else CPU)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        log(f"Using device: {self.device}")

        # Basic parameters
        self.initial_exploration_rate = initial_exploration_rate # Store initial rate
        self.min_exploration_rate = min_exploration_rate         # Store minimum rate
        self.exploration_decay_rate = exploration_decay_rate     # Store decay rate
        self.exploration_rate = self.initial_exploration_rate    # Start at initial rate
        self.CARD_ENCODING_SIZE = CARD_ENCODING_SIZE
        self.state_size = get_state_size(cards)  # Get state size based on cards
        self.cards = cards
        self.model = NeuralNetwork(self.state_size, get_action_space_size(self.cards))
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Change memory structure to track episodes
        self.memory = []
        self.current_episode = []

    def make_decision(self, game_state: 'Game'):
        available_actions = get_available_actions(game_state, game_state.current_player)
        state = encode_state(game_state, is_current_player_training=True, cards=self.cards, available_actions=available_actions)
        
        # Exploration-exploitation trade-off
        if np.random.random() < self.exploration_rate:
            # if there is an end turn action, remove it from available actions
            if len(available_actions) > 1:
                available_actions = [action for action in available_actions if action.type != ActionType.END_TURN]
            action = random.choice(available_actions)
            log(f"Random action chosen: {action}", v=True)
            return action
        
        with torch.no_grad():
            action_values: torch.Tensor = self.model(state)
        
        # Extract the action mask from the encoded state
        # The action part of the state is a one-hot encoding of available actions
        # STATE_SIZE represents all non-action parts of the state
        action_mask_start = STATE_SIZE
        action_mask_end = STATE_SIZE + get_action_space_size(self.cards)
        action_mask = state[action_mask_start:action_mask_end]
        
        # Apply the mask to action values (set unavailable actions to -inf)
        masked_action_values = action_values.clone()
        masked_action_values[action_mask == 0] = float('-inf')
        
        # Select best valid action index
        best_action_index = torch.argmax(masked_action_values).item()
        
        # Find the original Action object corresponding to the best index
        for action in available_actions:
            if encode_action(action, cards=self.cards) == best_action_index:
                log(f"Best action chosen: {action} (index {best_action_index})", v=True)
                return action
        
        # Fallback if decoding fails (should not happen if masking is correct)
        log(f"Warning: Could not decode action index {best_action_index}. Choosing randomly.")
        return random.choice(available_actions)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in current episode"""
        self.current_episode.append((state, action, reward, next_state, done))
    
    def finish_remembering_episode(self):
        self.memory.append(self.current_episode)
        self.current_episode = []

    def train(self, lambda_param, episode_sample_size):
        """Train model using experience replay"""
        # Need enough complete episodes
        if len(self.memory) < episode_sample_size:
            return
        
        # Take the most recent episodes
        sampled_episodes = self.memory[-episode_sample_size:]
        
        # Extract transitions from episodes
        all_states = []
        all_actions = []
        all_td_targets = []
        
        for episode in sampled_episodes:
            if len(episode) < 2:  # Skip very short episodes
                continue

            # Final reward for the last state in the episode
            last_state, last_action, last_reward, _, done = episode[-1]
                
            # Process each transition in the episode
            for t in range(len(episode)):
                state, action, reward, _, _ = episode[t]

                n_step_return = 0

                # Distance to the end of the episode, where the last step distance is 0
                distance_to_end = len(episode) - t - 1

                n_step_return += last_reward * (lambda_param ** distance_to_end)
                
                all_states.append(state)
                all_actions.append(encode_action(action, cards=self.cards))
                all_td_targets.append(n_step_return)
        
        # Convert to tensors
        indices = range(len(all_states))
        states = torch.stack([all_states[i] for i in indices])
        actions = torch.LongTensor([all_actions[i] for i in indices])
        td_targets = torch.FloatTensor([all_td_targets[i] for i in indices])
        
        # Compute Q values
        q_values: torch.Tensor = self.model(states)
        
        # Update model using the TD targets
        self.optimizer.zero_grad()
        # Select the Q-values corresponding to the actions taken
        predicted_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss: torch.Tensor = nn.MSELoss()(predicted_q_values, td_targets)
        loss.backward()
        self.optimizer.step()

        # Decay exploration rate after training step
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay_rate)
        log(f"Exploration rate decayed to: {self.exploration_rate:.4f}")        
        
        # Keep a random 1/10 sample of the episodes just trained on
        # Get the episodes that were just used for training
        trained_episodes = self.memory[-episode_sample_size:]
        # Keep all episodes before the ones we just trained on
        self.memory = self.memory[:-episode_sample_size]
        # Add a random 10% sample of the trained episodes back to memory
        sample_size = max(1, int(episode_sample_size * 0.1))
        if trained_episodes:
            sampled_episodes = random.sample(trained_episodes, sample_size)
            self.memory.extend(sampled_episodes)

    def save_memory(self, memory_file):
        """Saves the agent's memory (list of episodes) to a file using pickle."""
        log(f"Attempting to save memory to {memory_file}...")
        try:
            with open(memory_file, 'wb') as f:
                pickle.dump(self.memory, f)
            log(f"Successfully saved {len(self.memory)} episodes to {memory_file}.")
        except Exception as e:
            log(f"Error saving memory to {memory_file}: {e}")
