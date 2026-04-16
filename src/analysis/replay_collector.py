"""Replay collection for post-hoc policy analysis.

Records compact per-decision snapshots during BatchRunner games.
All card references are stored as integer indices; human-readable
names are resolved at analysis time to minimize collection overhead.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from src.engine.game import Game
    from src.engine.player import Player


@dataclass
class DecisionRecord:
    """One PPO decision point — compact, index-based."""
    turn: int
    player_health: int
    opp_health: int
    trade: int
    combat: int
    hand_card_ids: list[int]
    bases_in_play_ids: list[int]
    trade_row_ids: list[int]
    deck_size: int
    discard_size: int
    action_idx: int
    action_type: str
    action_card_id: int | None
    top_k: list[tuple[int, float]]     # (action_idx, probability) top 5
    policy_entropy: float
    value_estimate: float
    # Buy availability: which card indices had legal BUY actions this step
    buyable_card_ids: list[int]


@dataclass
class GameReplay:
    """Complete replay for one game."""
    game_id: int
    winner: str
    total_turns: int
    opponent_type: str
    decisions: list[DecisionRecord] = field(default_factory=list)


class ReplayCollector:
    """Collects replay data from BatchRunner games.

    Maintains per-game-slot buffers. Call ``record_decision`` after each
    PPO action selection, and ``finish_game`` when a game ends.
    """

    def __init__(self, card_names: list[str], action_dim: int):
        self.card_names = card_names
        self.action_dim = action_dim
        self._active: dict[int, list[DecisionRecord]] = {}
        self.replays: list[GameReplay] = []
        self._next_game_id = 0

        # Pre-compute BUY_CARD action index range
        # Matches action_encoder.py: offset 3 + num_cards for PLAY, then BUY
        n = len(card_names)
        self._buy_offset = 3 + n
        self._buy_end = self._buy_offset + n

    def start_game(self, slot: int) -> None:
        """Begin tracking a new game in the given concurrent slot."""
        self._active[slot] = []

    def record_decision(
        self,
        slot: int,
        game: 'Game',
        player: 'Player',
        logits: torch.Tensor,
        value: float,
        mask: torch.Tensor | np.ndarray,
        action_idx: int,
        action_type: str,
        action_card_id: int | None,
    ) -> None:
        """Record a single decision. Called from BatchRunner Step 5."""
        if slot not in self._active:
            return

        # Convert mask to numpy if needed
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask

        # Compute probabilities from masked logits
        if isinstance(logits, torch.Tensor):
            logits_masked = logits.clone()
            logits_masked[mask_np == 0] = float('-inf')
            probs = torch.softmax(logits_masked, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            probs_np = probs.cpu().numpy()
        else:
            probs_np = np.zeros(self.action_dim)
            entropy = 0.0

        # Top-k actions by probability
        top_indices = np.argsort(probs_np)[-5:][::-1]
        top_k = [(int(idx), float(probs_np[idx])) for idx in top_indices if probs_np[idx] > 0]

        # Buyable card ids from mask
        buy_mask = mask_np[self._buy_offset:self._buy_end]
        buyable_ids = [int(i) for i in np.where(buy_mask > 0)[0]]

        # Opponent
        opponent = game.get_opponent(player)

        record = DecisionRecord(
            turn=game.stats.total_turns,
            player_health=player.health,
            opp_health=opponent.health if opponent else 0,
            trade=player.trade,
            combat=player.combat,
            hand_card_ids=[c.index for c in player.hand if c.index is not None],
            bases_in_play_ids=[b.index for b in player.bases if b.index is not None],
            trade_row_ids=[c.index for c in game.trade_row if c.index is not None],
            deck_size=len(player.deck),
            discard_size=len(player.discard_pile),
            action_idx=action_idx,
            action_type=action_type,
            action_card_id=action_card_id,
            top_k=top_k,
            policy_entropy=entropy,
            value_estimate=value,
            buyable_card_ids=buyable_ids,
        )
        self._active[slot].append(record)

    def finish_game(
        self,
        slot: int,
        winner: str,
        total_turns: int,
        opponent_type: str,
    ) -> None:
        """Finalize the game in this slot and add it to completed replays."""
        decisions = self._active.pop(slot, [])
        if not decisions:
            return
        replay = GameReplay(
            game_id=self._next_game_id,
            winner=winner,
            total_turns=total_turns,
            opponent_type=opponent_type,
            decisions=decisions,
        )
        self.replays.append(replay)
        self._next_game_id += 1

    def save(self, path: str) -> None:
        """Save replays to gzipped JSONL (one game per line)."""
        meta = {"card_names": self.card_names, "action_dim": self.action_dim}
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            f.write(json.dumps(meta) + '\n')
            for replay in self.replays:
                f.write(json.dumps(asdict(replay)) + '\n')

    @staticmethod
    def load(path: str) -> tuple[dict, list[GameReplay]]:
        """Load replays from gzipped JSONL.

        Returns (metadata_dict, list_of_GameReplay).
        """
        replays: list[GameReplay] = []
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            meta = json.loads(f.readline())
            for line in f:
                data = json.loads(line)
                decisions = [DecisionRecord(**d) for d in data.pop('decisions')]
                replays.append(GameReplay(**data, decisions=decisions))
        return meta, replays
