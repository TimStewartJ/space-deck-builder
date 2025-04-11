from typing import TYPE_CHECKING
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.ai.agent import Agent
from src.engine.actions import get_available_actions, Action, ActionType

if TYPE_CHECKING:
    from src.engine.game import Game
    from cards.card import Card
    from cards.effects import CardEffectType

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class NeuralAgent(Agent):
    def __init__(self, name, cli_interface=None, learning_rate=0.001):
        super().__init__(name, cli_interface)
        # Basic parameters
        self.exploration_rate = 0.2  # Adjust as needed
        self.CARD_ENCODING_SIZE = 19  # Adjust based on your card encoding
        self.state_size = 860
        self.action_size = 50
        self.model = NeuralNetwork(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.memory = []

    def encode_state(self, game_state: 'Game'):
        """Convert variable-length game state to fixed-length tensor"""
        MAX_HAND = 20
        MAX_TRADE_ROW = 5
        MAX_BASES = 10
        
        # Initialize state vector with zeros
        state = []
        
        # Encode player resources (fixed length)
        state.extend([
            game_state.current_player.trade / 100.0,  # Normalize values
            game_state.current_player.combat / 100.0,
            game_state.current_player.health / 100.0,
            len(game_state.current_player.deck) / 40.0,
            len(game_state.current_player.discard_pile) / 40.0
        ])
        
        # Encode player hand (variable -> fixed)
        hand_encoding = []
        for card in game_state.current_player.hand[:MAX_HAND]:
            hand_encoding.extend(self.encode_card(card))
        # Pad to fixed length
        padding_needed = MAX_HAND - len(game_state.current_player.hand)
        hand_encoding.extend([0] * (padding_needed * self.CARD_ENCODING_SIZE))
        state.extend(hand_encoding)
        
        # Encode trade row
        trade_row_encoding = []
        for card in game_state.trade_row[:MAX_TRADE_ROW]:
            trade_row_encoding.extend(self.encode_card(card))
        padding_needed = MAX_TRADE_ROW - len(game_state.trade_row)
        trade_row_encoding.extend([0] * (padding_needed * self.CARD_ENCODING_SIZE))
        state.extend(trade_row_encoding)

        # Encode player bases
        bases_encoding = []
        for base in game_state.current_player.bases[:MAX_BASES]:
            bases_encoding.extend(self.encode_card(base))
        padding_needed = MAX_BASES - len(game_state.current_player.bases)
        bases_encoding.extend([0] * (padding_needed * self.CARD_ENCODING_SIZE))
        state.extend(bases_encoding)

        # Encode opponent bases
        opponent = game_state.get_opponent(game_state.current_player)
        opponent_bases_encoding = []
        for base in opponent.bases[:MAX_BASES]:
            opponent_bases_encoding.extend(self.encode_card(base))
        padding_needed = MAX_BASES - len(opponent.bases)
        opponent_bases_encoding.extend([0] * (padding_needed * self.CARD_ENCODING_SIZE))
        state.extend(opponent_bases_encoding)
        
        return torch.FloatTensor(state)
    
    def encode_card(self, card: 'Card'):
        """Convert a card to a fixed-length embedding vector"""
        # imports
        from src.cards.effects import CardEffectType

        # Card type one-hot encoding (ship, base, outpost)
        card_type = [0, 0, 0]  # [ship, base, outpost]
        if card.card_type == "ship":
            card_type[0] = 1
        elif card.card_type == "base":
            card_type[1] = 1
            if card.is_outpost():
                card_type[2] = 1
        
        # Faction one-hot encoding
        faction = [0, 0, 0, 0, 0]  # [trade_federation, blob, machine_cult, star_empire, unaligned]
        if card.faction == "Trade Federation":
            faction[0] = 1
        elif card.faction == "Blob":
            faction[1] = 1
        elif card.faction == "Machine Cult":
            faction[2] = 1
        elif card.faction == "Star Empire":
            faction[3] = 1
        else:  # Unaligned
            faction[4] = 1
        
        # Numeric properties
        properties = [
            card.cost / 10.0,  # Normalize cost
            card.defense / 10.0 if card.defense else 0,
        ]
        
        # Effects encoding
        effects = []
        for effect in card.effects[:1]:  # Limit to first effect for simplicity
            effect_type = [0] * len(CardEffectType)  # One-hot for effect types
            effect_type[list(CardEffectType).index(effect.effect_type)] = 1
            effects.extend(effect_type + [effect.value / 10.0])
        
        # Combine all features and ensure fixed length
        card_encoding = card_type + faction + properties + effects
        return card_encoding

    def encode_action(self, action: Action) -> int:
        """Convert an Action object to a numerical index for neural network processing
        
        Maps different action types to different index ranges:
        - 0: END_TURN
        - 1: SKIP_DECISION
        - 2-21: PLAY_CARD (indices 0-19 for cards in hand)
        - 22-26: BUY_CARD (indices 0-4 for trade row)
        - 27-36: ATTACK_BASE (indices 0-9 for bases)
        - 37-38: ATTACK_PLAYER (indices 0-1 for players)
        - 39-49: APPLY_EFFECT, SCRAP_CARD, etc.
        
        Returns an integer representation of the action.
        """
        if action.type == ActionType.END_TURN:
            return 0
        
        if action.type == ActionType.SKIP_DECISION:
            return 1
        
        if action.type == ActionType.PLAY_CARD:
            # Find the card index in player's hand
            if hasattr(action, 'card_index'):
                # If index is already computed
                return 2 + action.card_index
            else:
                # Compute card index (max 20 cards in hand)
                for i, card in enumerate(getattr(action, 'player_hand', [])):
                    if card.name == action.card_id:
                        return 2 + min(i, 19)
                return 2  # Default to first card if not found
        
        if action.type == ActionType.BUY_CARD:
            # Find the card index in trade row
            if hasattr(action, 'card_index'):
                # If index is already computed
                return 22 + action.card_index
            else:
                # Compute card index (max 5 cards in trade row)
                for i, card in enumerate(getattr(action, 'trade_row', [])):
                    if card.name == action.card_id:
                        return 22 + min(i, 4)
                return 22  # Default to first card if not found
        
        if action.type == ActionType.ATTACK_BASE:
            # Encode base index (max 10 bases)
            if hasattr(action, 'base_index'):
                return 27 + min(action.base_index, 9)
            return 27  # Default to first base
        
        if action.type == ActionType.ATTACK_PLAYER:
            # Encode player index (max 2 players)
            if hasattr(action, 'player_index'):
                return 37 + min(action.player_index, 1)
            return 37  # Default to first player
        
        if action.type == ActionType.APPLY_EFFECT:
            # Encode effect actions (max 10 effects)
            if hasattr(action, 'effect_index'):
                return 39 + min(action.effect_index, 9)
            return 39  # Default to first effect
        
        if action.type == ActionType.SCRAP_CARD:
            return 49  # Single index for scrap actions
        
        # Default case
        return 0

    def decode_action(self, action_idx: int, available_actions: list[Action]) -> Action:
        """Convert a neural network action index back to a game Action object
        
        Parameters:
        - action_idx: The output index from the neural network
        - available_actions: List of valid actions for the current game state
        
        Returns the corresponding Action from available_actions that matches
        the action_idx, or falls back to a default action if invalid.
        """
        # Bound check
        if action_idx < 0 or not available_actions:
            return available_actions[0] if available_actions else None
        
        # Create mappings of available actions by type
        action_by_type = {ActionType.END_TURN: [], ActionType.SKIP_DECISION: [],
                        ActionType.PLAY_CARD: [], ActionType.BUY_CARD: [],
                        ActionType.ATTACK_BASE: [], ActionType.ATTACK_PLAYER: [],
                        ActionType.APPLY_EFFECT: [], ActionType.SCRAP_CARD: []}
        
        for action in available_actions:
            if action.type in action_by_type:
                action_by_type[action.type].append(action)
        
        # Handle END_TURN (index 0)
        if action_idx == 0 and action_by_type[ActionType.END_TURN]:
            return action_by_type[ActionType.END_TURN][0]
        
        # Handle SKIP_DECISION (index 1)
        if action_idx == 1 and action_by_type[ActionType.SKIP_DECISION]:
            return action_by_type[ActionType.SKIP_DECISION][0]
        
        # Handle PLAY_CARD (indices 2-21)
        if 2 <= action_idx <= 21 and action_by_type[ActionType.PLAY_CARD]:
            card_idx = action_idx - 2
            if card_idx < len(action_by_type[ActionType.PLAY_CARD]):
                return action_by_type[ActionType.PLAY_CARD][card_idx]
        
        # Handle BUY_CARD (indices 22-26)
        if 22 <= action_idx <= 26 and action_by_type[ActionType.BUY_CARD]:
            card_idx = action_idx - 22
            if card_idx < len(action_by_type[ActionType.BUY_CARD]):
                return action_by_type[ActionType.BUY_CARD][card_idx]
        
        # Handle ATTACK_BASE (indices 27-36)
        if 27 <= action_idx <= 36 and action_by_type[ActionType.ATTACK_BASE]:
            base_idx = action_idx - 27
            if base_idx < len(action_by_type[ActionType.ATTACK_BASE]):
                return action_by_type[ActionType.ATTACK_BASE][base_idx]
        
        # Handle ATTACK_PLAYER (indices 37-38)
        if 37 <= action_idx <= 38 and action_by_type[ActionType.ATTACK_PLAYER]:
            player_idx = action_idx - 37
            if player_idx < len(action_by_type[ActionType.ATTACK_PLAYER]):
                return action_by_type[ActionType.ATTACK_PLAYER][player_idx]
        
        # Handle APPLY_EFFECT (indices 39-48)
        if 39 <= action_idx <= 48 and action_by_type[ActionType.APPLY_EFFECT]:
            effect_idx = action_idx - 39
            if effect_idx < len(action_by_type[ActionType.APPLY_EFFECT]):
                return action_by_type[ActionType.APPLY_EFFECT][effect_idx]
        
        # Handle SCRAP_CARD (index 49)
        if action_idx == 49 and action_by_type[ActionType.SCRAP_CARD]:
            return action_by_type[ActionType.SCRAP_CARD][0]
        
        # Fallback: return first available action
        return available_actions[0]
        
    def make_decision(self, game_state: 'Game'):
        available_actions = get_available_actions(game_state, game_state.current_player)
        state = self.encode_state(game_state)
        
        # Exploration-exploitation trade-off
        if np.random.random() < self.exploration_rate:
            return np.random.choice(available_actions)
        
        with torch.no_grad():
            action_values = self.model(state)
        
        # Select best valid action
        return self.decode_action(action_values.argmax().item(), available_actions)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self, batch_size=64):
        """Train model using experience replay"""
        if len(self.memory) < batch_size:
            return
            
        # Sample batch from memory
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for i in batch:
            state, action, reward, next_state, done = self.memory[i]
            states.append(state)
            actions.append(self.encode_action(action))
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        # Convert to tensors
        states = torch.stack(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.stack(next_states)
        dones = torch.FloatTensor(dones)
        
        # Compute Q values
        q_values = self.model(states)
        next_q_values = self.model(next_states).detach()
        
        # Use Q-learning update rule
        target_q = rewards + 0.99 * (1 - dones) * torch.max(next_q_values, dim=1)[0]
        
        # Update model
        self.optimizer.zero_grad()
        loss = nn.MSELoss()(q_values.gather(1, actions.unsqueeze(1)).squeeze(1), target_q)
        loss.backward()
        self.optimizer.step()