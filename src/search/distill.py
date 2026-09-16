"""Distill search self-play targets into a policy checkpoint.

The student starts from the checkpoint that guided the search. Each step
minimizes the cross-entropy between the search's completed-Q improved policy
and the student's policy over legal actions, plus a weighted squared error
between the critic and the game outcome. Games with ``index %
validation_modulus == 0`` are held out; training stops when the validation
loss has not improved for ``patience`` epochs, and the best validation
checkpoint is saved.
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from src.search.selfplay_data import file_sha256, load_selfplay_data


@dataclass(frozen=True)
class DistillConfig:
    model_path: str
    data_dir: str
    output_path: str
    lr: float = 1e-4
    batch_size: int = 8192
    max_epochs: int = 8
    patience: int = 2
    value_coef: float = 0.5
    grad_clip: float = 0.5
    validation_modulus: int = 20
    seed: int = 0
    device: str = "cuda"

    def __post_init__(self):
        if self.lr <= 0 or self.value_coef < 0 or self.grad_clip <= 0:
            raise ValueError("lr and grad_clip must be positive and value_coef non-negative")
        for name in ("batch_size", "max_epochs", "patience", "validation_modulus"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")


def policy_value_losses(logits, values, masks, policy, outcome):
    """Per-row policy cross-entropy and squared value error."""
    log_probs = torch.log_softmax(logits.masked_fill(~masks, float("-inf")), dim=1)
    cross_entropy = -(policy * log_probs.masked_fill(~masks, 0.0)).sum(dim=1)
    return cross_entropy, (values.reshape(-1) - outcome) ** 2


def _batches(indices: np.ndarray, batch_size: int):
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


class _Tensors:
    def __init__(self, data, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self.observations = torch.from_numpy(data.observations)
        self.masks = torch.from_numpy(data.masks)
        self.policy = torch.from_numpy(data.policy)
        self.outcome = torch.from_numpy(data.outcome)
        self.action = torch.from_numpy(data.action.astype(np.int64))
        self.greedy = torch.from_numpy(data.greedy.astype(np.int64))
        self.forced = torch.from_numpy(data.forced)

    def batch(self, rows: np.ndarray):
        index = torch.from_numpy(rows)
        return (
            self.observations[index].to(self.device, dtype=self.dtype, non_blocking=True),
            self.masks[index].to(self.device, non_blocking=True),
            self.policy[index].to(self.device, dtype=self.dtype, non_blocking=True),
            self.outcome[index].to(self.device, dtype=self.dtype, non_blocking=True),
            self.action[index].to(self.device, non_blocking=True),
            self.greedy[index].to(self.device, non_blocking=True),
            self.forced[index].to(self.device, non_blocking=True),
        )


@torch.no_grad()
def evaluate(model, initial, tensors: _Tensors, rows: np.ndarray, config: DistillConfig) -> dict:
    """Validation metrics for *model* on *rows*."""
    model.eval()
    totals = {
        "policy_ce": 0.0, "target_entropy": 0.0, "value_mse": 0.0, "kl_initial": 0.0,
        "search_agreement": 0.0, "initial_greedy_agreement": 0.0,
    }
    searched = 0
    outcome_sum = outcome_square = error_sum = error_square = 0.0
    for batch in _batches(rows, config.batch_size):
        observations, masks, policy, outcome, action, greedy, forced = tensors.batch(batch)
        logits, values = model(observations)
        cross_entropy, squared = policy_value_losses(logits, values, masks, policy, outcome)
        entropy = -(policy * torch.log(policy.clamp_min(1e-30))).sum(dim=1)
        initial_logits, _ = initial(observations)
        log_probs = torch.log_softmax(logits.masked_fill(~masks, float("-inf")), dim=1)
        initial_log_probs = torch.log_softmax(initial_logits.masked_fill(~masks, float("-inf")), dim=1)
        kl = (initial_log_probs.exp() * (initial_log_probs - log_probs)).masked_fill(~masks, 0.0).sum(dim=1)
        predicted = logits.masked_fill(~masks, float("-inf")).argmax(dim=1)
        choice = ~forced
        totals["policy_ce"] += float(cross_entropy.sum())
        totals["target_entropy"] += float(entropy.sum())
        totals["value_mse"] += float(squared.sum())
        totals["kl_initial"] += float(kl.sum())
        totals["search_agreement"] += float((predicted == action)[choice].sum())
        totals["initial_greedy_agreement"] += float((predicted == greedy)[choice].sum())
        searched += int(choice.sum())
        error = outcome - values.reshape(-1)
        outcome_sum += float(outcome.sum())
        outcome_square += float((outcome * outcome).sum())
        error_sum += float(error.sum())
        error_square += float((error * error).sum())
    count = len(rows)
    metrics = {
        "positions": count,
        "policy_ce": totals["policy_ce"] / count,
        "target_entropy": totals["target_entropy"] / count,
        "value_mse": totals["value_mse"] / count,
        "kl_initial": totals["kl_initial"] / count,
        "search_agreement": totals["search_agreement"] / max(1, searched),
        "initial_greedy_agreement": totals["initial_greedy_agreement"] / max(1, searched),
    }
    metrics["policy_kl"] = metrics["policy_ce"] - metrics["target_entropy"]
    outcome_variance = outcome_square / count - (outcome_sum / count) ** 2
    error_variance = error_square / count - (error_sum / count) ** 2
    metrics["value_explained_variance"] = (
        1.0 - error_variance / outcome_variance if outcome_variance > 0 else float("nan")
    )
    metrics["loss"] = metrics["policy_ce"] + config.value_coef * metrics["value_mse"]
    return metrics


def distill(config: DistillConfig, log_fn=print) -> dict:
    """Train a student checkpoint from self-play shards and return its history."""
    from src.config import DataConfig, ModelConfig, load_checkpoint, save_checkpoint
    from src.encoding.action_encoder import get_action_space_size
    from src.encoding.state_encoder import get_state_size
    from src.ppo.ppo_actor_critic import PPOActorCritic
    from src.utils.logger import set_disabled

    set_disabled(True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu")

    data_cfg = DataConfig()
    cards = data_cfg.load_cards()
    registry = data_cfg.build_registry(cards)
    names = registry.card_names
    checkpoint = load_checkpoint(config.model_path, map_location="cpu")
    saved = checkpoint.get("config", {}).get("model")
    model_config = ModelConfig.from_dict(saved) if saved else ModelConfig()
    model = PPOActorCritic(
        get_state_size(names), get_action_space_size(names), len(names),
        model_config=model_config, card_registry=registry,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    initial = copy.deepcopy(model).eval()
    for parameter in initial.parameters():
        parameter.requires_grad_(False)
    dtype = next(model.parameters()).dtype

    started = time.perf_counter()
    data = load_selfplay_data(config.data_dir)
    if data.observations.shape[1] != model.state_dim or data.masks.shape[1] != model.action_dim:
        raise ValueError("self-play data dimensions do not match the model")
    validation = data.game % config.validation_modulus == 0
    train_rows = np.flatnonzero(~validation)
    validation_rows = np.flatnonzero(validation)
    if len(train_rows) == 0 or len(validation_rows) == 0:
        raise ValueError("training and validation splits must both be non-empty")
    tensors = _Tensors(data, device, dtype)
    log_fn(
        f"distill: {data.positions} positions from {len(np.unique(data.game))} games "
        f"({len(train_rows)} train, {len(validation_rows)} validation), loaded in "
        f"{time.perf_counter() - started:.0f}s"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history = [{"epoch": 0, "validation": evaluate(model, initial, tensors, validation_rows, config)}]
    best = history[0]["validation"]["loss"]
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    log_fn(f"  epoch 0: " + ", ".join(f"{key}={value:.4f}" for key, value in history[0]["validation"].items() if key != "positions"))
    generator = np.random.default_rng(config.seed)
    stale = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        order = generator.permutation(train_rows)
        sums = {"policy_ce": 0.0, "value_mse": 0.0, "grad_norm": 0.0}
        steps = 0
        epoch_started = time.perf_counter()
        for batch in _batches(order, config.batch_size):
            observations, masks, policy, outcome, _, _, _ = tensors.batch(batch)
            logits, values = model(observations)
            cross_entropy, squared = policy_value_losses(logits, values, masks, policy, outcome)
            loss = cross_entropy.mean() + config.value_coef * squared.mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite distillation loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            sums["policy_ce"] += float(cross_entropy.detach().mean())
            sums["value_mse"] += float(squared.detach().mean())
            sums["grad_norm"] += float(norm)
            steps += 1
        train = {key: value / steps for key, value in sums.items()}
        train["steps"] = steps
        train["elapsed_s"] = time.perf_counter() - epoch_started
        metrics = evaluate(model, initial, tensors, validation_rows, config)
        history.append({"epoch": epoch, "train": train, "validation": metrics})
        log_fn(
            f"  epoch {epoch}: train ce={train['policy_ce']:.4f} mse={train['value_mse']:.4f} | validation "
            + ", ".join(f"{key}={value:.4f}" for key, value in metrics.items() if key != "positions")
        )
        if metrics["loss"] < best:
            best = metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    model.load_state_dict(best_state)
    summary = {
        "config": asdict(config),
        "source_model_sha256": file_sha256(config.model_path),
        "data_positions": data.positions,
        "train_positions": len(train_rows),
        "validation_positions": len(validation_rows),
        "best_epoch": best_epoch,
        "history": history,
        "elapsed_s": time.perf_counter() - started,
    }
    os.makedirs(os.path.dirname(os.path.abspath(config.output_path)), exist_ok=True)
    save_checkpoint(
        config.output_path,
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        model_config=model_config,
        extra={"distill": summary},
    )
    with open(f"{config.output_path}.history.json", "w", encoding="utf-8") as fh:
        json.dump(_json_safe(summary), fh, indent=2)
        fh.write("\n")
    log_fn(f"  saved epoch {best_epoch} checkpoint to {config.output_path}")
    return summary


def _json_safe(value):
    """Replace non-finite floats with None so the history is valid JSON."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
