import torch
import torch.nn as nn
import torch.optim as optim
from typing import TYPE_CHECKING, List, Optional
import time
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
            nn.Linear(state_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1024),
            nn.Tanh(),
            nn.Linear(1024, action_dim)
        )
        # critic head
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value

class PPOAgent(Agent):
    def __init__(
        self,
        name: str,
        card_names: List[str],
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        epochs: int = 4,
        batch_size: int = 64,
        entropy_coef: float = 0.01,
        device: str = "cuda",
        main_device: str = "cuda",
        simulation_device: str = "cpu",
        model_path: Optional[str] = None,
        log_debug: bool = False,
    ):
        super().__init__(name)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.main_device = torch.device(main_device if torch.cuda.is_available() else "cpu")
        self.simulation_device = torch.device(simulation_device if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.cards = card_names

        self.state_dim = get_state_size(card_names)
        self.action_dim = get_action_space_size(card_names)

        if log_debug:
            log(f"State size: {self.state_dim}, Action size: {self.action_dim}")
            log(f"Using device: {self.device}")

        self.model = PPOActorCritic(self.state_dim, self.action_dim).to(self.device)
        if model_path:
            log(f"Loading PPO model from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # rollout buffers
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

        # decision timing
        self.total_decision_time = 0.0
        self.num_decisions = 0

    def make_decision(self, game_state: 'Game'):
        self.device = self.simulation_device
        self.model.to(self.simulation_device)

        start_time = time.perf_counter()
        available = get_available_actions(game_state, game_state.current_player)
        state = encode_state(
            game_state, is_current_player_training=True,
            cards=self.cards, available_actions=available
        ).to(self.device)

        logits: torch.Tensor
        value: torch.Tensor
        logits, value = self.model(state)
        # mask out unavailable
        mask = state[-self.action_dim:]
        logits = logits.masked_fill(mask == 0, float('-inf'))

        dist = torch.distributions.Categorical(torch.softmax(logits, -1))
        act_idx = dist.sample().item()
        logp: torch.Tensor = dist.log_prob(torch.tensor(act_idx, device=self.device))

        # decode back to Action
        action = next(a for a in available if encode_action(a, cards=self.cards) == act_idx)

        # store
        self.states.append(state.detach())
        self.actions.append(act_idx)
        self.log_probs.append(logp.detach())
        self.values.append(value.detach())
        self.dones.append(False)

        # timing
        elapsed = time.perf_counter() - start_time
        self.total_decision_time += elapsed
        self.num_decisions += 1

        return action

    def store_reward(self, reward: float, done: bool):
        self.rewards.append(reward)
        self.dones[-1] = done

    def make_last_reward_negative(self):
        """Make the last reward negative to indicate end of episode."""
        if self.rewards:
            self.rewards[-1] = -1.0
            self.dones[-1] = True

    def finish_batch(self):
        self.device = self.simulation_device
        self.model.to(self.device)
        # compute GAE & returns
        returns, advs = [], []
        gae = 0.0
        vals = self.values + [torch.tensor([0.0], device=self.device)]  # add terminal value
        # Compute Generalized Advantage Estimation (GAE) and returns
        for step in reversed(range(len(self.rewards))):
            # Temporal difference error (delta)
            delta = (
                self.rewards[step]                           # immediate reward at this step
                + self.gamma * vals[step + 1] * (1 - self.dones[step])  # discounted next value if not done
                - vals[step]                                 # subtract current value estimate
            )
            # GAE calculation
            gae = delta + self.gamma * self.lam * (1 - self.dones[step]) * gae
            advs.insert(0, gae)  # Insert advantage at the beginning
            returns.insert(0, gae + vals[step])  # Return = advantage + value

        # to tensors
        states   = torch.stack(self.states).to(self.device)
        actions  = torch.tensor(self.actions, dtype=torch.int64, device=self.device)
        old_lp   = torch.stack(self.log_probs).to(self.device)
        returns  = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advs     = torch.tensor(advs, dtype=torch.float32, device=self.device)

        # clear buffers
        self.clear_buffers()

        return states, actions, old_lp, returns, advs

    def update(self, states, actions, old_lp, returns, advs):
        self.device = self.main_device
        self.model.to(self.main_device)
        actor_loss = 0.0
        critic_loss = 0.0

        for _ in range(self.epochs):
            idxs = torch.randperm(len(states))  # Shuffle indices for mini-batch SGD
            for start in range(0, len(states), self.batch_size):
                b = idxs[start:start+self.batch_size]  # Indices for current batch
                s = states[b]        # Batch of states
                a = actions[b]       # Batch of actions
                olp = old_lp[b]      # Batch of old log probabilities
                R = returns[b]       # Batch of returns (discounted rewards)
                A = advs[b]          # Batch of advantages

                logits, vals = self.model(s)  # Forward pass: get action logits and value estimates
                dist = torch.distributions.Categorical(torch.softmax(logits, -1))  # Action distribution
                nl = dist.log_prob(a)  # New log probabilities for taken actions
                ratio = (nl - olp).exp()  # Probability ratio for PPO objective

                s1 = ratio * A  # Unclipped surrogate objective
                s2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * A  # Clipped surrogate objective
                actor_loss = -torch.min(s1, s2).mean()  # PPO actor loss (policy update)
                critic_loss = nn.MSELoss()(vals, R)     # Critic loss (value function update)
                entropy = dist.entropy().mean()         # entropy bonus term
                loss = actor_loss + 0.5*critic_loss - self.entropy_coef * entropy  # include entropy bonus

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

    def get_average_decision_time(self, reset: bool = True):
        if self.num_decisions == 0:
            avg = 0.0
        else:
            avg = self.total_decision_time / self.num_decisions
        if reset:
            self.total_decision_time = 0.0
            self.num_decisions = 0
        return avg