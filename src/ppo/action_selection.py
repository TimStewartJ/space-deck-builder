"""Action selection from masked policy logits at an inference temperature."""
from __future__ import annotations

import math

import torch


def validate_temperature(temperature: float) -> float:
    """Return *temperature* as a float, rejecting negative or non-finite values."""
    try:
        value = float(temperature)
    except (TypeError, ValueError):
        raise ValueError(
            f"temperature must be a finite non-negative number, got {temperature!r}"
        ) from None
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"temperature must be a finite non-negative number, got {temperature!r}"
        )
    return value


def select_actions(
    masked_logits: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose one action per row from logits whose illegal entries are ``-inf``.

    Temperature 1 samples the policy distribution, other positive temperatures
    sample ``softmax(logits / temperature)``, and temperature 0 selects the
    highest-probability legal action (the lowest index on ties). Returns
    ``(actions, log_probs)`` where log-probabilities are under the distribution
    actually used for selection; a deterministic choice has log-probability 0.
    """
    value = validate_temperature(temperature)
    if value == 0.0:
        actions = masked_logits.argmax(dim=-1)
        return actions, torch.zeros(
            actions.shape,
            dtype=masked_logits.dtype,
            device=masked_logits.device,
        )
    logits = masked_logits if value == 1.0 else masked_logits / value
    distribution = torch.distributions.Categorical(
        logits=logits, validate_args=False
    )
    actions = distribution.sample()
    return actions, distribution.log_prob(actions)
