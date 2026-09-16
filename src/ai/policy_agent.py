"""Scalar agent that acts from a PPO policy at a fixed inference temperature."""
from __future__ import annotations

import numpy as np
import torch

from src.ai.agent import Agent
from src.encoding.action_context import build_action_context
from src.encoding.action_encoder import END_TURN_INDEX, get_action_space_size
from src.encoding.state_encoder import encode_state, get_state_size
from src.ppo.action_selection import select_actions, validate_temperature


class PolicyAgent(Agent):
    """Choose each action from a PPO policy with production END_TURN rules.

    END_TURN is masked while a card can be played or the opponent can be
    attacked directly, matching training and evaluation. Temperature 0 selects
    the highest-probability legal action.
    """

    def __init__(
        self,
        name: str,
        model: torch.nn.Module,
        card_names: list[str],
        card_index_map: dict[str, int],
        *,
        device: torch.device | str = "cpu",
        temperature: float = 0.0,
    ):
        super().__init__(name)
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        parameter = next(self.model.parameters(), None)
        self.dtype = parameter.dtype if parameter is not None else torch.float32
        self.card_names = card_names
        self.card_index_map = card_index_map
        self.temperature = validate_temperature(temperature)
        self.action_dim = get_action_space_size(card_names)
        self._state_buf = np.zeros(get_state_size(card_names), dtype=np.float32)

    @torch.no_grad()
    def make_decision(self, game_state):
        player = game_state.current_player
        context = build_action_context(
            game_state, player, self.card_index_map, self.action_dim
        )
        observation = encode_state(
            game_state,
            is_current_player_training=True,
            cards=self.card_names,
            card_index_map=self.card_index_map,
            state_buf=self._state_buf,
            can_buy=context.can_buy,
            has_actions=context.has_meaningful,
            return_numpy=True,
        )
        mask = context.mask.copy()
        if context.has_meaningful:
            mask[END_TURN_INDEX] = False
        states = torch.from_numpy(observation.copy()).to(
            self.device, dtype=self.dtype
        ).unsqueeze(0)
        logits, _ = self.model(states)
        logits = logits.reshape(1, -1).masked_fill(
            ~torch.from_numpy(mask).to(self.device).unsqueeze(0),
            float("-inf"),
        )
        action, _ = select_actions(logits, self.temperature)
        return context.resolvers[int(action.item())]
