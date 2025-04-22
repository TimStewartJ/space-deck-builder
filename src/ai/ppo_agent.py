import torch
import torch.nn as nn
import torch.optim as optim
from typing import TYPE_CHECKING, List
import numpy as np
from src.ai.agent import Agent
from src.engine.actions import get_available_actions
from src.nn.state_encoder import encode_state, get_state_size
from src.nn.action_encoder import encode_action, decode_action, get_action_space_size
from src.utils.logger import log

if TYPE_CHECKING:
    from src.engine.game import Game

class PPOActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        # actor head
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim)
        )
        # critic head
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self, x: torch.Tensor):
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value

class PPOAgent(Agent):
    def __init__(
        self,
        name: str,
        cards: List[str],
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        epochs: int = 4,
        batch_size: int = 64,
    ):
        super().__init__(name)
        self.device = torch.device("cpu")
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.cards = cards

        self.state_dim = get_state_size(cards)
        self.action_dim = get_action_space_size(cards)
        log(f"State size: {self.state_dim}, Action size: {self.action_dim}")
        self.model = PPOActorCritic(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # rollout buffers
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def make_decision(self, game_state: 'Game'):
        available = get_available_actions(game_state, game_state.current_player)
        state = encode_state(
            game_state, is_current_player_training=True,
            cards=self.cards, available_actions=available
        ).to(self.device)

        logits, value = self.model(state)
        # mask out unavailable
        mask = state[-self.action_dim:]
        logits = logits.masked_fill(mask == 0, float('-inf'))

        dist = torch.distributions.Categorical(torch.softmax(logits, -1))
        act_idx = dist.sample().item()
        logp = dist.log_prob(torch.tensor(act_idx, device=self.device))

        # decode back to Action
        action = next(a for a in available if encode_action(a, cards=self.cards) == act_idx)

        # store
        self.states.append(state.detach())
        self.actions.append(act_idx)
        self.log_probs.append(logp.detach())
        self.values.append(value.detach())
        self.dones.append(False)

        return action

    def store_reward(self, reward: float, done: bool):
        self.rewards.append(reward)
        self.dones[-1] = done

    def finish_batch(self, next_value: torch.Tensor):
        # compute GAE & returns
        returns, advs = [], []
        gae = 0.0
        vals = self.values + [next_value]
        for step in reversed(range(len(self.rewards))):
            delta = (
                self.rewards[step]
                + self.gamma * vals[step + 1] * (1 - self.dones[step])
                - vals[step]
            )
            gae = delta + self.gamma * self.lam * (1 - self.dones[step]) * gae
            advs.insert(0, gae)
            returns.insert(0, gae + vals[step])

        # to tensors
        states   = torch.stack(self.states)
        actions  = torch.tensor(self.actions, dtype=torch.int64, device=self.device)
        old_lp   = torch.stack(self.log_probs)
        returns  = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advs     = torch.tensor(advs, dtype=torch.float32, device=self.device)

        # clear buffers
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.values, self.dones      = [], [], []

        return states, actions, old_lp, returns, advs

    def update(self, states, actions, old_lp, returns, advs):
        actor_loss = 0.0
        critic_loss = 0.0

        for _ in range(self.epochs):
            idxs = torch.randperm(len(states))
            for start in range(0, len(states), self.batch_size):
                b = idxs[start:start+self.batch_size]
                s = states[b]
                a = actions[b]
                olp = old_lp[b]
                R = returns[b]
                A = advs[b]

                logits, vals = self.model(s)
                dist = torch.distributions.Categorical(torch.softmax(logits, -1))
                nl = dist.log_prob(a)
                ratio = (nl - olp).exp()

                s1 = ratio * A
                s2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * A
                actor_loss = -torch.min(s1, s2).mean()
                critic_loss = nn.MSELoss()(vals, R)
                loss = actor_loss + 0.5*critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        log(f"PPO update done. Actor loss {actor_loss:.3f}  Critic loss {critic_loss:.3f}")

    def clear_buffers(self):
        """Clear all rollout buffers."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []