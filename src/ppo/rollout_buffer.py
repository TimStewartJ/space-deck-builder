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
        self.masks: list[torch.Tensor] = []

    def add(self, state: torch.Tensor, action: int, log_prob: torch.Tensor,
            value: torch.Tensor, reward: float = 0.0, done: bool = False,
            mask: Optional[torch.Tensor] = None):
        self.states.append(state.detach())
        self.actions.append(action)
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())
        self.rewards.append(reward)
        self.dones.append(done)
        if mask is not None:
            self.masks.append(mask.detach().bool())

    def fill_last_reward(self, reward: float):
        """Set the terminal reward and mark the episode as done."""
        self.rewards[-1] = reward
        self.dones[-1] = True

    def __len__(self):
        return len(self.states)

    def finish(self, gamma: float, lam: float, device: torch.device,
              normalize: bool = True):
        """Compute GAE returns and advantages.

        Args:
            normalize: If True, normalize advantages to mean=0/std=1 for this
                episode. Set to False when using global normalization after
                merging multiple rollouts.

        Returns (states, actions, old_lp, returns, advantages, masks).
        """
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
        if normalize:
            advs_t = (advs_t - advs_t.mean()) / (advs_t.std(unbiased=False) + 1e-8)

        # Include action masks if they were stored
        masks_t = torch.stack(self.masks).to(device) if self.masks else None
        return states, actions, old_lp, returns_t, advs_t, masks_t


def merge_rollouts(rollout_results: list[tuple],
                   normalize: bool = False) -> tuple:
    """Merge multiple (states, actions, old_lp, returns, advs, masks) tuples.

    Args:
        normalize: If True, normalize the concatenated advantages globally
            to mean=0/std=1 after merging.
    """
    S, A, OL, R, Adv, M = zip(*rollout_results)
    has_masks = all(m is not None for m in M)
    advs = torch.cat(Adv)
    if normalize:
        advs = (advs - advs.mean()) / (advs.std(unbiased=False) + 1e-8)
    return (
        torch.cat(S),
        torch.cat(A),
        torch.cat(OL),
        torch.cat(R),
        advs,
        torch.cat(M) if has_masks else None,
    )
