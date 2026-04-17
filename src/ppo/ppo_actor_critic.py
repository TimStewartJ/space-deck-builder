import math

import torch
import torch.nn as nn

from src.config import ModelConfig
from src.encoding.state_utils import unpack_state
from src.encoding.action_encoder import get_action_space_layout


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


class AttentionActorHead(nn.Module):
    """Actor head using query-key dot-product scoring for card-indexed actions.

    For each card-indexed action type (play, buy, attack_base, etc.), a
    state-conditioned query is projected from the trunk output and scored
    against dedicated action card embeddings via scaled dot product.
    Global actions (end_turn, skip, attack_player) use a small linear layer.

    All card-indexed query projections are fused into a single Linear layer
    and scored with one batched matmul + scatter for GPU efficiency.
    """

    def __init__(self, trunk_dim: int, emb_dim: int, num_cards: int,
                 action_dim: int):
        super().__init__()
        layout = get_action_space_layout(num_cards)

        self.action_dim = action_dim
        self.num_cards = num_cards
        self.scale = 1.0 / math.sqrt(emb_dim)

        # Dedicated action card embeddings — separate from state card_emb
        # to prevent policy gradients from distorting state representations.
        self.action_card_emb = nn.Embedding(num_cards, emb_dim)

        # Global actions: linear projection for non-card-indexed actions
        global_indices = layout['global_indices']
        self._global_idx_list = sorted(global_indices.values())
        self.global_head = nn.Linear(trunk_dim, len(self._global_idx_list))

        # Fused query projection: one Linear produces all type queries at once.
        # Output is reshaped to [B, num_types, emb_dim] for batched scoring.
        card_groups = layout['card_groups']  # [(name, offset, count), ...]
        self._num_card_types = len(card_groups)
        self.query_proj = nn.Linear(trunk_dim, self._num_card_types * emb_dim)

        # Pre-compute scatter index: maps [num_types * num_cards] → action_dim
        scatter_idx = []
        for name, offset, count in card_groups:
            scatter_idx.extend(range(offset, offset + count))
        self.register_buffer(
            '_global_indices',
            torch.tensor(self._global_idx_list, dtype=torch.long),
        )
        self.register_buffer(
            '_scatter_idx',
            torch.tensor(scatter_idx, dtype=torch.long),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Produce flat logits [B, action_dim] from trunk output h [B, trunk_dim]."""
        B = h.shape[0]
        logits = h.new_zeros(B, self.action_dim)

        # Global action logits
        global_logits = self.global_head(h)  # [B, num_global]
        logits.scatter_(1, self._global_indices.unsqueeze(0).expand(B, -1),
                        global_logits)

        # Fused card-indexed action logits:
        # 1) Single projection → all queries: [B, num_types * E] → [B, T, E]
        # 2) Single einsum scores all types against card embeddings: [B, T, C]
        # 3) Single scatter writes scores into the flat logits vector
        all_queries = self.query_proj(h).view(B, self._num_card_types, -1)
        all_scores = torch.einsum(
            'bte,ce->btc', all_queries, self.action_card_emb.weight,
        ) * self.scale                                      # [B, T, C]
        all_scores = all_scores.reshape(B, -1)              # [B, T*C]
        logits.scatter_(1, self._scatter_idx.unsqueeze(0).expand(B, -1),
                        all_scores)

        return logits


class AttentionZonePooling(nn.Module):
    """Per-zone attention pooling over cards.

    Replaces presence-weighted sum pooling. Each of the Z zones has a learned
    query vector. For every (batch, zone), cards are softmax-weighted by their
    dot product with that zone's query, masked to positions where ``presence
    > 0``, and then combined via a presence-weighted sum of the token values.

    Output shape matches ``sum`` pooling: ``[B, Z * E]``, so the downstream
    trunk and heads are unchanged.

    Empty zones (no cards present) produce zero vectors; the softmax would
    otherwise be NaN over all-``-inf`` rows.
    """

    def __init__(self, num_zones: int, emb_dim: int):
        super().__init__()
        self.num_zones = num_zones
        self.emb_dim = emb_dim
        self.scale = 1.0 / math.sqrt(emb_dim)
        # One learned query per zone. Initialized small so early-training
        # behavior approximates uniform attention.
        self.zone_query = nn.Parameter(torch.randn(num_zones, emb_dim) * 0.01)

    def forward(
        self,
        card_w: torch.Tensor,       # [C, E]
        zone_w: torch.Tensor,       # [Z, E]
        presence: torch.Tensor,     # [B, Z, C]
    ) -> torch.Tensor:
        B, Z, C = presence.shape
        E = card_w.shape[1]

        # Per-zone tokens: [Z, C, E] = card_emb[c] + zone_emb[z]
        tokens = card_w.unsqueeze(0) + zone_w.unsqueeze(1)

        # Scores[b, z, c] = tokens[z, c] · zone_query[z] (broadcast over B)
        # einsum keeps it in one kernel.
        scores = torch.einsum('zce,ze->zc', tokens, self.zone_query) * self.scale
        scores = scores.unsqueeze(0).expand(B, Z, C)  # [B, Z, C]

        mask = presence > 0
        any_present = mask.any(dim=-1, keepdim=True)  # [B, Z, 1]

        # Mask absent cards to -inf; all-absent rows are patched below.
        scores = scores.masked_fill(~mask, float('-inf'))
        # Replace all-masked rows with zeros pre-softmax to avoid NaN; output
        # gets zeroed by any_present gate anyway.
        safe_scores = torch.where(any_present, scores, torch.zeros_like(scores))
        attn = torch.softmax(safe_scores, dim=-1)

        # Weight by presence so stacked copies contribute more, matching the
        # semantics of the original sum pooling.
        weights = attn * presence  # [B, Z, C]

        # Pooled[b, z] = Σ_c weights[b, z, c] * tokens[z, c]
        pooled = torch.einsum('bzc,zce->bze', weights, tokens)
        pooled = pooled * any_present.to(pooled.dtype)  # zero out empty zones

        return pooled.reshape(B, Z * E)


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
        self.state_dim = state_dim
        self.actor_type = cfg.actor_type
        self.pool_type = cfg.pool_type

        # Card identity embedding and zone-position embedding
        self.card_emb = nn.Embedding(num_cards, emb_dim)
        self.zone_emb = nn.Embedding(NUM_ZONES, emb_dim)

        # Optional attention pooling module. When pool_type="sum" we keep the
        # original presence-weighted sum path for backward compatibility.
        if self.pool_type == "attention":
            self.zone_pool = AttentionZonePooling(NUM_ZONES, emb_dim)
        else:
            self.zone_pool = None

        combined_dim = NUM_ZONES * emb_dim + NUMERIC_DIM

        # Shared feature trunk (ends with ReLU)
        self.trunk = _build_trunk(combined_dim, cfg.trunk_hidden_sizes)
        trunk_out_dim = cfg.trunk_hidden_sizes[-1] if cfg.trunk_hidden_sizes else combined_dim

        # Policy head: MLP (flat linear) or attention (query-key dot product)
        if self.actor_type == "attention":
            self.actor_head = AttentionActorHead(
                trunk_out_dim, emb_dim, num_cards, action_dim,
            )
        else:
            self.actor_head = _build_head(trunk_out_dim, cfg.actor_head_sizes, action_dim)

        # Value head (unchanged regardless of actor type)
        self.critic_head = _build_head(trunk_out_dim, cfg.critic_head_sizes, 1)

    def forward(self, x: torch.Tensor):
        # x: [B, state_dim] or [state_dim]
        pieces, single = unpack_state(x, self.num_cards, self.action_dim)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        # -- Zone pooling: attention-based or presence-weighted sum --
        card_w = self.card_emb.weight  # [C, E]

        if self.zone_pool is not None:
            # Stack per-zone presence into [B, Z, C] and pool via attention.
            presence_stack = torch.stack(
                [pieces[name] for name in ZONE_NAMES], dim=1,
            )
            card_feat = self.zone_pool(card_w, self.zone_emb.weight, presence_stack)
        else:
            zone_feats = []
            for zone_idx, zone_name in enumerate(ZONE_NAMES):
                presence = pieces[zone_name]                         # [B, C]
                # card + zone embedding for this zone
                combined = card_w + self.zone_emb.weight[zone_idx]   # [C, E] (broadcast)
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
