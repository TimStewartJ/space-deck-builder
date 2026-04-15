import torch
import torch.nn as nn

from src.encoding.state_utils import unpack_state
from src.utils.logger import log

class PPOActorCritic(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 num_cards: int,
                 card_emb_dim: int = 4):
        super().__init__()
        self.num_cards = num_cards
        self.action_dim = action_dim

        # small location embedding
        self.locations = [
            'trade_row',
            'train_hand','train_disc','train_deck','train_bases',
            'opp_hand',  'opp_disc',  'opp_deck',  'opp_bases'
        ]
        # per-card location embeddings: num_locations * num_cards vectors
        # self.loc_emb = nn.Embedding(len(self.locations) * self.num_cards, card_emb_dim)
        self.loc_emb = nn.Embedding(self.num_cards, card_emb_dim)

        # numeric dims unchanged
        self.numeric_dim = 4 + 5 + 5
        combined_dim = self.numeric_dim + len(self.locations)*card_emb_dim*num_cards

        self.actor = nn.Sequential(
            nn.Linear(combined_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(combined_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

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
