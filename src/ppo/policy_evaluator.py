"""Stateless policy evaluator for batched inference."""
import torch
from typing import List
from src.ppo.ppo_actor_critic import PPOActorCritic
from src.engine.actions import ActionType, Action
from src.nn.state_encoder import encode_state, get_state_size
from src.nn.action_encoder import encode_action, get_action_space_size


class PolicyEvaluator:
    """Stateless policy evaluation — no buffers, no game coupling.
    
    Takes encoded states + action masks, returns actions/log_probs/values.
    Supports both single and batched evaluation.
    """

    def __init__(self, model: PPOActorCritic, card_names: List[str],
                 action_dim: int, device: torch.device):
        self.model = model
        self.card_names = card_names
        self.action_dim = action_dim
        self.device = device

    def evaluate_single(self, game_state, available_actions: List[Action]):
        """Evaluate a single game state. Returns (action, act_idx, log_prob, value, encoded_state)."""
        encoded_actions = [encode_action(a, cards=self.card_names) for a in available_actions]
        has_meaningful_actions = any(
            a.type in (ActionType.ATTACK_PLAYER, ActionType.PLAY_CARD)
            for a in available_actions
        )
        state = encode_state(
            game_state, is_current_player_training=True,
            cards=self.card_names, available_actions=available_actions
        ).to(self.device)

        logits, value = self.model(state)
        mask = torch.zeros(self.action_dim, device=self.device)
        mask[encoded_actions] = 1
        if has_meaningful_actions and 1 in encoded_actions:
            mask[1] = 0
        logits = logits.masked_fill(mask == 0, float('-inf'))

        dist = torch.distributions.Categorical(logits=logits)
        act_idx = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(act_idx, device=self.device))
        action = available_actions[encoded_actions.index(int(act_idx))]

        return action, act_idx, log_prob, value, state

    @torch.no_grad()
    def evaluate_batch(self, states: torch.Tensor, masks: torch.Tensor,
                       meaningful_action_flags: torch.Tensor):
        """Evaluate a batch of states. Returns (act_indices, log_probs, values).
        
        Args:
            states: (B, state_dim) encoded states
            masks: (B, action_dim) boolean masks of valid actions
            meaningful_action_flags: (B,) whether each game has play/attack actions
        """
        logits, values = self.model(states)

        # Suppress END_TURN (index 1) when meaningful actions exist
        end_turn_suppress = meaningful_action_flags & masks[:, 1].bool()
        masks_adj = masks.clone()
        masks_adj[end_turn_suppress, 1] = 0

        logits = logits.masked_fill(masks_adj == 0, float('-inf'))
        dist = torch.distributions.Categorical(logits=logits)
        act_indices = dist.sample()
        log_probs = dist.log_prob(act_indices)

        return act_indices, log_probs, values
