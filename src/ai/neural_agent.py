import random
from typing import TYPE_CHECKING
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.nn.state_encoder import CARD_ENCODING_SIZE, STATE_SIZE, encode_state
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
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )
        log(f"Neural Network initialized with input size: {input_size}, output size: {output_size}")
    
    def forward(self, x):
        return self.network(x)

class NeuralAgent(Agent):
    def __init__(self, name, cli_interface=None, learning_rate=0.001, look_ahead_steps=10, cards=None,
                 initial_exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay_rate=0.99): # Added exploration params
        super().__init__(name, cli_interface)
        # Basic parameters
        self.initial_exploration_rate = initial_exploration_rate # Store initial rate
        self.min_exploration_rate = min_exploration_rate         # Store minimum rate
        self.exploration_decay_rate = exploration_decay_rate     # Store decay rate
        self.exploration_rate = self.initial_exploration_rate    # Start at initial rate
        self.CARD_ENCODING_SIZE = CARD_ENCODING_SIZE
        self.state_size = STATE_SIZE
        self.cards = cards if cards is not None else []
        self.model = NeuralNetwork(self.state_size, get_action_space_size(self.cards))
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Change memory structure to track episodes
        self.memory = []
        self.current_episode = []
        self.gamma = 0.99  # Discount factor
        self.look_ahead_steps = look_ahead_steps  # New parameter for steps to look ahead
    
    def make_decision(self, game_state: 'Game'):
        available_actions = get_available_actions(game_state, game_state.current_player)
        state = encode_state(game_state, is_current_player_training=True)
        
        # Exploration-exploitation trade-off
        if np.random.random() < self.exploration_rate:
            # if there is an end turn action, remove it from available actions
            if len(available_actions) > 1:
                available_actions = [action for action in available_actions if action.type != ActionType.END_TURN]
            action = random.choice(available_actions)
            log(f"Random action chosen: {action}")
            return action
        
        with torch.no_grad():
            action_values: torch.Tensor = self.model(state)

        # Create a mask for available actions
        mask = torch.full_like(action_values, -float('inf'))
        available_action_indices = [encode_action(action, cards=self.cards) for action in available_actions]

        # Check for invalid indices before masking
        valid_indices = [idx for idx in available_action_indices if 0 <= idx < len(action_values)]
        if not valid_indices:
             # Fallback if no valid actions can be encoded (should not happen ideally)
             log("Warning: No valid actions found after encoding. Choosing randomly.")
             return random.choice(available_actions)

        mask[valid_indices] = action_values[valid_indices] # Use original values for valid actions
        
        # Select best valid action index
        best_action_index = torch.argmax(mask).item()

        # Decode the selected action index back to an Action object
        # Find the original Action object corresponding to the best index
        for action in available_actions:
            if encode_action(action, cards=self.cards) == best_action_index:
                log(f"Best action chosen: {action} (index {best_action_index})")
                return action
        
        # Fallback if decoding fails (should not happen if masking is correct)
        log(f"Warning: Could not decode action index {best_action_index}. Choosing randomly.")
        return random.choice(available_actions)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in current episode"""
        self.current_episode.append((state, action, reward, next_state, done))
    
        # When episode is done, store it in memory
        if done:
            self.memory.append(self.current_episode)
            self.current_episode = []  # Reset for next episode
    
    def train(self, batch_size=64, lambda_param=0.7, episode_sample_size=4):
        """Train model using experience replay with TD(λ) learning"""
        # Need enough complete episodes
        if len(self.memory) < episode_sample_size:
            return
        
        # Sample some episodes
        sampled_episodes = random.sample(self.memory, min(episode_sample_size, len(self.memory)))
        
        # Extract transitions from episodes
        all_states = []
        all_actions = []
        all_td_targets = []
        
        for episode in sampled_episodes:
            if len(episode) < 2:  # Skip very short episodes
                continue
                
            # Process each transition in the episode
            for t in range(len(episode)):
                state, action, reward, _, _ = episode[t]
                
                # Calculate n-step return with TD(λ)
                n_step_return = 0
                for n in range(min(len(episode) - t, self.look_ahead_steps)):  # Look ahead up to look_ahead_steps
                    # Get future reward n steps ahead
                    future_reward = episode[t + n][2]  # reward at t+n
                    
                    # Apply lambda and gamma discounting
                    discount = (self.gamma ** n) * (lambda_param ** n)
                    n_step_return += discount * future_reward
                
                # For states close to the end, add bootstrapped value
                if t + self.look_ahead_steps < len(episode):
                    final_state = episode[t + self.look_ahead_steps][3]  # next_state at t+look_ahead_steps
                    with torch.no_grad():
                        final_value = torch.max(self.model(final_state), dim=0)[0].item()
                        n_step_return += (self.gamma ** self.look_ahead_steps) * (lambda_param ** self.look_ahead_steps) * final_value
                
                all_states.append(state)
                all_actions.append(encode_action(action, cards=self.cards))
                all_td_targets.append(n_step_return)
        
        # Skip training if not enough samples
        if len(all_states) < batch_size:
            return
        
        # Convert to tensors
        indices = np.random.choice(len(all_states), batch_size, replace=False)
        states = torch.stack([all_states[i] for i in indices])
        actions = torch.LongTensor([all_actions[i] for i in indices])
        td_targets = torch.FloatTensor([all_td_targets[i] for i in indices])
        
        # Compute Q values
        q_values = self.model(states)
        
        # Update model using the TD targets
        self.optimizer.zero_grad()
        # Select the Q-values corresponding to the actions taken
        predicted_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.MSELoss()(predicted_q_values, td_targets)
        loss.backward()
        self.optimizer.step()

        # Decay exploration rate after training step
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay_rate)
        log(f"Exploration rate decayed to: {self.exploration_rate:.4f}")