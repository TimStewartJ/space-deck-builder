import torch
import torch.nn as nn

from src.config import ModelConfig
from src.encoding.state_utils import unpack_state


# Zone names used by unpack_state, in a fixed order for embedding indexing.
ZONE_NAMES = [
    'trade_row',
    'train_hand', 'train_disc', 'train_deck', 'train_bases',
    'opp_unseen', 'opp_disc', 'opp_bases',
]
NUM_ZONES = len(ZONE_NAMES)

# 4 flags + 5 training resources + 6 opponent resources
NUMERIC_DIM = 4 + 5 + 6


def _build_trunk(input_dim: int, hidden_sizes: list[int]) -> nn.Sequential:
    """Build a feature trunk: Linear → ReLU pairs (no bare output layer)."""
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    return nn.Sequential(*layers)


def _build_head(input_dim: int, hidden_sizes: list[int], output_dim: int) -> nn.Sequential:
    """Build a prediction head: hidden layers with ReLU, then a bare output layer."""
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class PPOActorCritic(nn.Module):
    """Actor-critic network with per-zone card embeddings, sum pooling,
    and a shared feature trunk.

    Card representation:
        Each card has a learned embedding (card_emb). Each of the 8 zones
        has a learned embedding (zone_emb). For a given zone, each card's
        feature is ``card_emb[card] + zone_emb[zone]``, weighted by the
        card's presence value and summed across all cards in the zone.
        This produces a fixed-size ``[emb_dim]`` vector per zone.

    Architecture:
        [zone_pooled (num_zones × emb_dim) | numeric (15)]
        → shared trunk (MLP)
        → actor head → logits
        → critic head → value
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 num_cards: int,
                 card_emb_dim: int | None = None,
                 model_config: ModelConfig | None = None):
        super().__init__()
        cfg = model_config or ModelConfig()
        emb_dim = card_emb_dim if card_emb_dim is not None else cfg.card_emb_dim

        self.num_cards = num_cards
        self.action_dim = action_dim

        # Card identity embedding and zone-position embedding
        self.card_emb = nn.Embedding(num_cards, emb_dim)
        self.zone_emb = nn.Embedding(NUM_ZONES, emb_dim)

        combined_dim = NUM_ZONES * emb_dim + NUMERIC_DIM

        # Shared feature trunk (ends with ReLU)
        self.trunk = _build_trunk(combined_dim, cfg.trunk_hidden_sizes)
        trunk_out_dim = cfg.trunk_hidden_sizes[-1] if cfg.trunk_hidden_sizes else combined_dim

        # Separate policy and value heads
        self.actor_head = _build_head(trunk_out_dim, cfg.actor_head_sizes, action_dim)
        self.critic_head = _build_head(trunk_out_dim, cfg.critic_head_sizes, 1)

    def forward(self, x: torch.Tensor):
        # x: [B, state_dim] or [state_dim]
        pieces, single = unpack_state(x, self.num_cards, self.action_dim)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        # -- Per-zone sum pooling with zone-aware embeddings --
        # card_emb.weight: [num_cards, emb_dim]
        card_w = self.card_emb.weight  # [C, E]

        zone_feats = []
        for zone_idx, zone_name in enumerate(ZONE_NAMES):
            presence = pieces[zone_name]                         # [B, C]
            # card + zone embedding for this zone
            combined = card_w + self.zone_emb.weight[zone_idx]   # [C, E]  (broadcast)
            # weight by presence and sum-pool: [B, C, 1] * [1, C, E] → sum → [B, E]
            pooled = (presence.unsqueeze(-1) * combined.unsqueeze(0)).sum(dim=1)
            zone_feats.append(pooled)

        # Concatenate zone summaries: [B, num_zones * emb_dim]
        card_feat = torch.cat(zone_feats, dim=1)

        # Numeric features: [B, NUMERIC_DIM]
        numeric = torch.cat([
            pieces['is_train'], pieces['is_first'], pieces['can_buy'], pieces['has_actions'],
            pieces['train_res'], pieces['opp_res'],
        ], dim=1)

        # Shared trunk
        h = self.trunk(torch.cat([card_feat, numeric], dim=1))

        # Heads
        logits = self.actor_head(h)
        value = self.critic_head(h).squeeze(-1)

        if single:
            return logits.squeeze(0), value.squeeze(0)
        return logits, value
