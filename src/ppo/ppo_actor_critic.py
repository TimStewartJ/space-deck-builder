import math

import torch
import torch.nn as nn

from src.config import ModelConfig
from src.encoding.state_utils import unpack_state, unpack_state_tokens
from src.encoding.action_encoder import get_action_space_layout
from src.cards.factions import Faction
from src.cards.card import CardType
from src.cards.registry import CardRegistry


# Zone names used by unpack_state, in a fixed order for embedding indexing.
ZONE_NAMES = [
    'trade_row',
    'train_hand', 'train_disc', 'train_deck', 'train_bases',
    'opp_unseen', 'opp_disc', 'opp_bases',
]
NUM_ZONES = len(ZONE_NAMES)

# 4 flags + 5 training resources + 6 opponent resources
NUMERIC_DIM = 4 + 5 + 6

# Single-bit factions in canonical order, used to build one-hot encodings
# of card.faction and card.ally_factions for the static feature table.
_FACTION_BITS = (
    Faction.BLOB,
    Faction.MACHINE_CULT,
    Faction.STAR_EMPIRE,
    Faction.TRADE_FEDERATION,
)
NUM_FACTIONS = len(_FACTION_BITS)

# Static per-card feature vector layout (built from CardRegistry):
#   cost / 10                     (1)
#   defense / 10                  (1)
#   is_ship                       (1)
#   is_base                       (1)
#   is_outpost                    (1)
#   faction one-hot               (NUM_FACTIONS)
#   ally_faction one-hot          (NUM_FACTIONS)
STATIC_FEATURE_DIM = 5 + 2 * NUM_FACTIONS


class CardFeatureTable(nn.Module):
    """Static per-card feature lookup built from a ``CardRegistry``.

    Materializes a ``[num_cards, STATIC_FEATURE_DIM]`` float buffer at
    construction time encoding cost, defense, type flags, and faction /
    ally-faction one-hots. The registry is consumed once and not retained,
    so the module is self-contained and can move freely across devices.

    Cards present in the canonical name list but missing from the registry
    (e.g. starter cards without a ``CardDef``) get a zero feature row.
    Effects-derived signals (e.g. ``has_scrap_effect``) are intentionally
    deferred — see plan.md for the next phase.
    """

    def __init__(self, registry: CardRegistry):
        super().__init__()
        num_cards = registry.num_cards
        feats = torch.zeros(num_cards, STATIC_FEATURE_DIM, dtype=torch.float32)

        for name, idx in registry.card_index_map.items():
            cd = registry.get_by_name(name)
            if cd is None:
                continue  # starter without a CardDef → zero row
            row = feats[idx]
            row[0] = (cd.cost or 0) / 10.0
            row[1] = (cd.defense or 0) / 10.0
            row[2] = 1.0 if cd.card_type == CardType.SHIP else 0.0
            row[3] = 1.0 if cd.card_type in (CardType.BASE, CardType.OUTPOST) else 0.0
            row[4] = 1.0 if cd.card_type == CardType.OUTPOST else 0.0
            for j, bit in enumerate(_FACTION_BITS):
                if cd.faction & bit:
                    row[5 + j] = 1.0
            for j, bit in enumerate(_FACTION_BITS):
                if cd.ally_factions & bit:
                    row[5 + NUM_FACTIONS + j] = 1.0

        # Buffer (not parameter) — values are static per game configuration.
        self.register_buffer('features', feats)
        self.num_cards = num_cards
        self.feature_dim = STATIC_FEATURE_DIM


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
    """Per-zone attention pooling over precomputed card tokens.

    Each of the Z zones has a learned query vector. For every (batch, zone),
    tokens are softmax-weighted by their dot product with that zone's query,
    masked to slots where ``presence > 0``, and then combined via a
    presence-weighted sum. Output shape ``[B, Z * E]`` matches the legacy
    sum-pool, so the downstream trunk and heads are unchanged.

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
        tokens: torch.Tensor,       # [B, Z, C, E]
        presence: torch.Tensor,     # [B, Z, C]
    ) -> torch.Tensor:
        B, Z, C, E = tokens.shape

        # Scores[b, z, c] = tokens[b, z, c] · zone_query[z]
        scores = torch.einsum('bzce,ze->bzc', tokens, self.zone_query) * self.scale

        mask = presence > 0
        any_present = mask.any(dim=-1, keepdim=True)  # [B, Z, 1]

        # Mask absent cards to -inf; all-absent rows are patched below.
        scores = scores.masked_fill(~mask, float('-inf'))
        # Replace all-masked rows with zeros pre-softmax to avoid NaN; output
        # gets zeroed by the any_present gate anyway.
        safe_scores = torch.where(any_present, scores, torch.zeros_like(scores))
        attn = torch.softmax(safe_scores, dim=-1)

        # Weight by presence so stacked copies contribute more, matching the
        # semantics of the original sum pooling.
        weights = attn * presence  # [B, Z, C]

        # Pooled[b, z] = Σ_c weights[b, z, c] * tokens[b, z, c]
        pooled = torch.einsum('bzc,bzce->bze', weights, tokens)
        pooled = pooled * any_present.to(pooled.dtype)  # zero out empty zones

        return pooled.reshape(B, Z * E)


def _sum_pool_tokens(
    tokens: torch.Tensor,       # [B, Z, C, E]
    presence: torch.Tensor,     # [B, Z, C]
) -> torch.Tensor:
    """Presence-weighted sum pooling over precomputed tokens.

    Equivalent to the original inline pooling once tokens are constructed:
    ``pooled[b, z] = Σ_c presence[b, z, c] * tokens[b, z, c]``. Shares the
    ``(tokens, presence) → [B, Z * E]`` signature with ``AttentionZonePooling``
    so the trunk is agnostic to the pool choice.
    """
    B, Z, C, E = tokens.shape
    pooled = torch.einsum('bzc,bzce->bze', presence, tokens)
    return pooled.reshape(B, Z * E)


class PPOActorCritic(nn.Module):
    """Actor-critic network with per-zone card tokens, pooled into a fixed
    feature vector and fed through a shared trunk + actor/critic heads.

    Token construction (per slot = one (zone, card_def) pair):
        Default (``token_features=False``): token = card_emb[c] + zone_emb[z],
        broadcast across the batch. Pooled per zone using either
        presence-weighted sum or per-zone attention.

        With static features (``token_features=True``): token =
        token_proj([card_emb[c] | zone_emb[z] | static_features[c] | presence]).
        Static features carry cost, defense, type flags, and faction /
        ally-faction one-hots from the CardRegistry, so the embedding table no
        longer has to memorize this metadata. Requires a ``CardRegistry``.

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
                 model_config: ModelConfig | None = None,
                 card_registry: CardRegistry | None = None):
        super().__init__()
        cfg = model_config or ModelConfig()
        emb_dim = card_emb_dim if card_emb_dim is not None else cfg.card_emb_dim

        self.num_cards = num_cards
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.actor_type = cfg.actor_type
        self.pool_type = cfg.pool_type
        self.token_features = cfg.token_features

        # Card identity embedding and zone-position embedding
        self.card_emb = nn.Embedding(num_cards, emb_dim)
        self.zone_emb = nn.Embedding(NUM_ZONES, emb_dim)

        # Static-feature path: precomputed per-card metadata table + a fused
        # projection from [card_emb | zone_emb | static_features | presence]
        # down to emb_dim. Same pooler signature applies in both modes.
        if self.token_features:
            if card_registry is None:
                raise ValueError(
                    "token_features=True requires a CardRegistry; pass "
                    "card_registry= to PPOActorCritic."
                )
            if card_registry.num_cards != num_cards:
                raise ValueError(
                    f"CardRegistry has {card_registry.num_cards} cards but "
                    f"model was constructed with num_cards={num_cards}."
                )
            self.card_features = CardFeatureTable(card_registry)
            # Token input dim: card_emb (E) + zone_emb (E) + static (S) + presence (1)
            token_in_dim = 2 * emb_dim + STATIC_FEATURE_DIM + 1
            self.token_proj = nn.Linear(token_in_dim, emb_dim)
        else:
            self.card_features = None
            self.token_proj = None

        # Optional attention pooling module. When pool_type="sum" we use the
        # presence-weighted sum-pool helper for backward compatibility.
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

    def _build_tokens(
        self,
        presence: torch.Tensor,     # [B, Z, C]
    ) -> torch.Tensor:
        """Materialize per-slot token features ``[B, Z, C, emb_dim]``.

        Two paths share the same downstream pooler signature:
          * Legacy (``token_features=False``): tokens are
            ``card_emb[c] + zone_emb[z]`` broadcast over the batch, matching
            the original sum-pool semantics bit-for-bit.
          * Static-feature: concat of card_emb, zone_emb, static features and
            the per-slot presence value, projected to ``emb_dim``.
        """
        B, Z, C = presence.shape
        E = self.card_emb.embedding_dim

        card_w = self.card_emb.weight                      # [C, E]
        zone_w = self.zone_emb.weight                      # [Z, E]

        if self.token_features:
            card_part = card_w.view(1, 1, C, E).expand(B, Z, C, E)
            zone_part = zone_w.view(1, Z, 1, E).expand(B, Z, C, E)
            static = self.card_features.features            # [C, S]
            static_part = static.view(1, 1, C, -1).expand(B, Z, C, -1)
            presence_part = presence.unsqueeze(-1)          # [B, Z, C, 1]
            raw = torch.cat([card_part, zone_part, static_part, presence_part], dim=-1)
            return self.token_proj(raw)                     # [B, Z, C, E]

        # Legacy path: card_emb[c] + zone_emb[z], broadcast across batch.
        tokens_zc = card_w.unsqueeze(0) + zone_w.unsqueeze(1)  # [Z, C, E]
        return tokens_zc.unsqueeze(0).expand(B, Z, C, E)

    def forward(self, x: torch.Tensor):
        # x: [B, state_dim] or [state_dim]
        presence, numeric, single = unpack_state_tokens(
            x, self.num_cards, self.action_dim,
        )

        # Build per-slot tokens [B, Z, C, E] under both code paths.
        tokens = self._build_tokens(presence)

        # Pool: attention or presence-weighted sum.
        if self.zone_pool is not None:
            card_feat = self.zone_pool(tokens, presence)
        else:
            card_feat = _sum_pool_tokens(tokens, presence)

        # Shared trunk
        h = self.trunk(torch.cat([card_feat, numeric], dim=1))

        # Heads
        logits = self.actor_head(h)
        value = self.critic_head(h).squeeze(-1)

        if single:
            return logits.squeeze(0), value.squeeze(0)
        return logits, value
