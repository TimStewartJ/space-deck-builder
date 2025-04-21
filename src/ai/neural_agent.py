import pickle
import random
from typing import TYPE_CHECKING, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.nn.experience import Experience
from src.nn.state_encoder import encode_state, get_state_size
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
            nn.Linear(input_size, 1280),
            nn.ReLU(),
            nn.Linear(1280, 1280),
            nn.ReLU(),
            nn.Linear(1280, 1280),
            nn.ReLU(),
            nn.Linear(1280, 1280),
            nn.ReLU(),
            nn.Linear(1280, 1280),
            nn.ReLU(),
            nn.Linear(1280, output_size)
        )
        log(f"Neural Network initialized with input size: {input_size}, output size: {output_size}")
    
    def forward(self, x):
        return self.network(x)

class NeuralAgent(Agent):
    def __init__(self, name, cli_interface=None, learning_rate=0.001, cards: list[str] = [],
                 initial_exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay_rate=0.99,
                 model_file_path:str|None=None): # Added exploration params
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
        self.state_size = get_state_size(cards)  # Get state size based on cards
        self.action_size = get_action_space_size(cards)
        self.cards = cards
        self.model = NeuralNetwork(self.state_size, get_action_space_size(self.cards))
        if model_file_path:
            log(f"Loading model from {model_file_path}")
            self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def make_decision(self, game_state: 'Game'):
        available_actions = get_available_actions(game_state, game_state.current_player)
        state = encode_state(game_state, is_current_player_training=True, cards=self.cards, available_actions=available_actions)
        
        # Exploration-exploitation trade-off
        if np.random.random() < self.exploration_rate:
            action = random.choice(available_actions)
            log(f"Random action chosen: {action}", v=True)
            return action
        
        with torch.no_grad():
            action_values: torch.Tensor = self.model(state)
        
        # Extract the action mask from the encoded state
        action_mask_start = self.state_size - self.action_size
        action_mask_end = self.state_size
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

    def train(self, experiences: List[Experience], lambda_param: float):
        """
        Train model using provided episodes
        
        Args:
            episodes: List of episodes to train on, where each episode is a list of (state, action, reward, next_state, done) tuples
            lambda_param: Lambda parameter for n-step returns calculation
        """ 
        indices = range(len(experiences))
        states = torch.stack([experiences[i].state for i in indices])
        actions = torch.LongTensor([experiences[i].action for i in indices])
        td_targets = torch.FloatTensor([experiences[i].reward for i in indices])
        
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
