"""Per-episode rollout buffer for PPO training data."""
import numpy as np
import torch
from typing import Optional, Union


class RolloutBuffer:
    """Stores rollout data for a single episode/game.

    Storage contract: raw numpy arrays and Python primitives are kept
    per-step (no per-step torch wrapping). At :meth:`finish` everything
    is stacked and moved to the target device in a handful of bulk
    conversions. ``add`` accepts either numpy arrays / Python scalars
    (preferred, used by the IPC worker path) or torch tensors (used by
    the in-process batch runner) and normalizes to the raw storage form.
    """

    def __init__(self):
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.masks: list[np.ndarray] = []

    @staticmethod
    def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    @staticmethod
    def _to_float(x: Union[float, torch.Tensor]) -> float:
        if isinstance(x, torch.Tensor):
            return float(x.detach().item())
        return float(x)

    def add(self, state, action: int, log_prob,
            value, reward: float = 0.0, done: bool = False,
            mask=None):
        self.states.append(self._to_numpy(state))
        self.actions.append(action)
        self.log_probs.append(self._to_float(log_prob))
        self.values.append(self._to_float(value))
        self.rewards.append(reward)
        self.dones.append(done)
        if mask is not None:
            m = self._to_numpy(mask)
            self.masks.append(m.astype(np.bool_, copy=False))

    def fill_last_reward(self, reward: float):
        """Set the terminal reward and mark the episode as done."""
        self.rewards[-1] = reward
        self.dones[-1] = True

    def __len__(self):
        return len(self.states)

    def finish(self, gamma: float, lam: float, device: torch.device,
              normalize: bool = True):
        """Compute GAE returns and advantages and stack rollout tensors.

        States, log_probs, values and masks are stored as raw numpy /
        Python primitives and stacked once here with bulk torch
        conversions — this avoids per-step ``torch.tensor(...)``
        wrapping overhead in the hot worker loop. GAE is computed in
        pure Python over the reward/value lists (cheap vs. the encoding
        and inference paths), then the final rollout tensors are moved
        onto ``device``.

        Args:
            normalize: If True, normalize advantages to mean=0/std=1 for
                this episode. Set to False when using global
                normalization after merging multiple rollouts.

        Returns: (states, actions, old_lp, returns, advantages, masks)
        """
        n = len(self.rewards)
        values_list = self.values

        advs_list: list[float] = [0.0] * n
        returns_list: list[float] = [0.0] * n
        gae = 0.0
        next_value = 0.0
        for step in range(n - 1, -1, -1):
            not_done = 0.0 if self.dones[step] else 1.0
            delta = self.rewards[step] + gamma * next_value * not_done - values_list[step]
            gae = delta + gamma * lam * not_done * gae
            advs_list[step] = gae
            returns_list[step] = gae + values_list[step]
            next_value = values_list[step]

        states = torch.from_numpy(np.stack(self.states)).to(device)
        actions = torch.tensor(self.actions, dtype=torch.int64, device=device)
        old_lp = torch.tensor(self.log_probs, dtype=torch.float32, device=device)
        returns_t = torch.tensor(returns_list, dtype=torch.float32, device=device)
        advs_t = torch.tensor(advs_list, dtype=torch.float32, device=device)
        if normalize:
            advs_t = (advs_t - advs_t.mean()) / (advs_t.std(unbiased=False) + 1e-8)

        if self.masks:
            masks_t = torch.from_numpy(np.stack(self.masks)).to(device)
        else:
            masks_t = None
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
