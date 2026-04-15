"""Per-episode rollout buffer for PPO training data."""
import torch
from typing import Optional


class RolloutBuffer:
    """Stores rollout data for a single episode/game."""

    def __init__(self):
        self.states: list[torch.Tensor] = []
        self.actions: list[int] = []
        self.log_probs: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []

    def add(self, state: torch.Tensor, action: int, log_prob: torch.Tensor,
            value: torch.Tensor, reward: float = 0.0, done: bool = False):
        self.states.append(state.detach())
        self.actions.append(action)
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())
        self.rewards.append(reward)
        self.dones.append(done)

    def fill_last_reward(self, reward: float):
        """Set the terminal reward and mark the episode as done."""
        self.rewards[-1] = reward
        self.dones[-1] = True

    def __len__(self):
        return len(self.states)

    def finish(self, gamma: float, lam: float, device: torch.device):
        """Compute GAE returns and advantages. Returns (states, actions, old_lp, returns, advantages)."""
        returns: list[torch.Tensor] = []
        advs: list[torch.Tensor] = []
        gae = 0.0
        vals = self.values + [torch.tensor(0.0, device=device)]

        for step in reversed(range(len(self.rewards))):
            delta = (
                self.rewards[step]
                + gamma * vals[step + 1] * (1 - self.dones[step])
                - vals[step]
            )
            gae = delta + gamma * lam * (1 - self.dones[step]) * gae
            advs.insert(0, gae)
            returns.insert(0, gae + vals[step])

        states = torch.stack(self.states).to(device)
        actions = torch.tensor(self.actions, dtype=torch.int64, device=device)
        old_lp = torch.stack(self.log_probs).to(device)
        returns_t = torch.stack(returns).to(device)
        advs_t = torch.stack(advs).to(device)
        advs_t = (advs_t - advs_t.mean()) / (advs_t.std(unbiased=False) + 1e-8)

        return states, actions, old_lp, returns_t, advs_t


def merge_rollouts(rollout_results: list[tuple]) -> tuple:
    """Merge multiple (states, actions, old_lp, returns, advs) tuples."""
    S, A, OL, R, Adv = zip(*rollout_results)
    return (
        torch.cat(S),
        torch.cat(A),
        torch.cat(OL),
        torch.cat(R),
        torch.cat(Adv),
    )
