import torch
import torch.nn as nn

from src.config import ModelConfig
from src.encoding.state_utils import unpack_state
from src.utils.logger import log


def _build_mlp(input_dim: int, hidden_sizes: list[int], output_dim: int) -> nn.Sequential:
    """Build a multi-layer perceptron from a list of hidden layer sizes."""
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class PPOActorCritic(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 num_cards: int,
                 card_emb_dim: int | None = None,
                 model_config: ModelConfig | None = None):
        super().__init__()
        cfg = model_config or ModelConfig()
        # Legacy kwarg takes precedence for backward compat
        emb_dim = card_emb_dim if card_emb_dim is not None else cfg.card_emb_dim

        self.num_cards = num_cards
        self.action_dim = action_dim

        # Card zone locations for embedding lookup
        self.locations = [
            'trade_row',
            'train_hand','train_disc','train_deck','train_bases',
            'opp_unseen', 'opp_disc', 'opp_bases'
        ]
        self.loc_emb = nn.Embedding(self.num_cards, emb_dim)

        # numeric: 4 flags + 5 training resources + 6 opponent resources
        self.numeric_dim = 4 + 5 + 6
        combined_dim = self.numeric_dim + len(self.locations) * emb_dim * num_cards

        self.actor = _build_mlp(combined_dim, cfg.actor_hidden_sizes, action_dim)
        self.critic = _build_mlp(combined_dim, cfg.critic_hidden_sizes, 1)

    def forward(self, x: torch.Tensor):
        # x: [B, state_dim] or [state_dim] if single
        pieces, single = unpack_state(x, self.num_cards, self.action_dim)
        # pieces: dict of tensors, each [B, num_cards] or [num_cards] for card locations,
        #         and [B, n] or [n] for numeric features

        feats = []
        for i, loc in enumerate(self.locations):
            # pieces[loc]: [B, num_cards]
            idx = torch.arange(self.num_cards, device=x.device) + i * self.num_cards  # [num_cards]
            # loc_embs: torch.Tensor = self.loc_emb(idx)  # [num_cards, card_emb_dim]
            # For each card in this location, multiply presence by embedding
            # [B, num_cards, card_emb_dim]
            loc_feat = pieces[loc].unsqueeze(-1) * self.loc_emb.weight.unsqueeze(0)
            # Flatten per location: [B, num_cards * card_emb_dim]
            loc_feat = loc_feat.reshape(x.shape[0], -1) if len(x.shape) > 1 else loc_feat.reshape(1, -1)
            feats.append(loc_feat)

        # Concatenate all locations: [B, num_locations * num_cards * card_emb_dim]
        card_feat = torch.cat(feats, dim=1)

        # log(f"card feat: {card_feat.squeeze(0).mean()}")

        # pieces['is_train']: [B, 1]
        # pieces['is_first']: [B, 1]
        # pieces['has_actions']: [B, 1]
        # pieces['train_res']: [B, 5]
        # pieces['opp_res']: [B, 5]
        # numeric: [B, 3 + 5 + 5] = [B, 13]
        numeric = torch.cat([
            pieces['is_train'], pieces['is_first'], pieces['can_buy'], pieces['has_actions'],
            pieces['train_res'], pieces['opp_res']
        ], dim=1)

        # h: [B, numeric_dim + len(self.locations)*card_emb_dim]
        h = torch.cat([numeric, card_feat], dim=1)
        # logits: [B, action_dim]
        logits = self.actor(h)
        # value: [B, 1] -> [B]
        value  = self.critic(h).squeeze(-1)
        
        # log(f"forward: {x.shape} -> {logits.shape} {logits}, {value.shape} {value}")
        # log(f"loc emb: {self.loc_emb.weight.shape} {self.loc_emb.weight}")

        if single:
            # logits: [action_dim], value: []
            return logits.squeeze(0), value.squeeze(0)
        return logits, value
