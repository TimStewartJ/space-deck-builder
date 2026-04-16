"""Tests for the replay collection and analysis system."""
import gzip
import json
import os
import tempfile

import pytest
import torch
import numpy as np

from src.analysis.replay_collector import (
    ReplayCollector, DecisionRecord, GameReplay,
)
from src.analysis.analyzer import analyze_replays, _game_phase, AnalysisResult
from src.config import DataConfig
from src.encoding.action_encoder import get_action_space_size
from src.encoding.state_encoder import get_state_size


# ---------- Fixtures ----------

@pytest.fixture
def card_setup():
    """Load real card names and compute dimensions."""
    dc = DataConfig()
    cards = dc.load_cards()
    card_names = dc.get_card_names(cards)
    action_dim = get_action_space_size(card_names)
    return card_names, action_dim


@pytest.fixture
def collector(card_setup):
    card_names, action_dim = card_setup
    return ReplayCollector(card_names, action_dim)


# ---------- ReplayCollector ----------

class TestReplayCollector:
    def test_start_and_finish_game(self, collector):
        """Starting and finishing a game produces one replay."""
        collector.start_game(0)
        collector.finish_game(0, winner="PPO", total_turns=20, opponent_type="random")
        # Empty game (no decisions) should not produce a replay
        assert len(collector.replays) == 0

    def test_record_decision_creates_record(self, collector, card_setup):
        """Recording a decision in an active game populates it."""
        card_names, action_dim = card_setup

        collector.start_game(0)

        # Create minimal mock game and player
        from unittest.mock import MagicMock
        game = MagicMock()
        game.stats.total_turns = 5
        game.trade_row = []
        player = MagicMock()
        player.health = 50
        player.trade = 3
        player.combat = 0
        player.hand = []
        player.bases = []
        player.deck = [1, 2, 3]
        player.discard_pile = []
        opponent = MagicMock()
        opponent.health = 45
        game.get_opponent.return_value = opponent

        logits = torch.randn(action_dim)
        mask = np.ones(action_dim, dtype=np.float32)
        value = 0.5

        collector.record_decision(
            slot=0, game=game, player=player,
            logits=logits, value=value, mask=mask,
            action_idx=1, action_type="END_TURN", action_card_id=None,
        )

        collector.finish_game(0, winner="PPO", total_turns=10, opponent_type="random")
        assert len(collector.replays) == 1
        assert len(collector.replays[0].decisions) == 1

        record = collector.replays[0].decisions[0]
        assert record.turn == 5
        assert record.player_health == 50
        assert record.opp_health == 45
        assert record.action_type == "END_TURN"
        assert record.policy_entropy > 0
        assert record.value_estimate == 0.5
        assert len(record.top_k) <= 5

    def test_multiple_concurrent_games(self, collector, card_setup):
        """Decisions from concurrent games stay separated."""
        card_names, action_dim = card_setup

        collector.start_game(0)
        collector.start_game(1)

        from unittest.mock import MagicMock
        for slot in [0, 1]:
            game = MagicMock()
            game.stats.total_turns = slot + 1
            game.trade_row = []
            player = MagicMock()
            player.health = 50
            player.trade = 0
            player.combat = 0
            player.hand = []
            player.bases = []
            player.deck = []
            player.discard_pile = []
            opponent = MagicMock()
            opponent.health = 50
            game.get_opponent.return_value = opponent

            # Record different number of decisions per slot
            for _ in range(slot + 1):
                collector.record_decision(
                    slot=slot, game=game, player=player,
                    logits=torch.randn(action_dim),
                    value=0.0, mask=np.ones(action_dim, dtype=np.float32),
                    action_idx=1, action_type="END_TURN", action_card_id=None,
                )

        collector.finish_game(0, "PPO", 10, "random")
        collector.finish_game(1, "Opp", 15, "heuristic")

        assert len(collector.replays) == 2
        assert len(collector.replays[0].decisions) == 1
        assert len(collector.replays[1].decisions) == 2
        assert collector.replays[0].winner == "PPO"
        assert collector.replays[1].winner == "Opp"

    def test_save_and_load_roundtrip(self, collector, card_setup):
        """Replays survive a save/load cycle via gzipped JSONL."""
        card_names, action_dim = card_setup

        collector.start_game(0)
        from unittest.mock import MagicMock
        game = MagicMock()
        game.stats.total_turns = 7
        game.trade_row = []
        player = MagicMock()
        player.health = 42
        player.trade = 5
        player.combat = 3
        player.hand = []
        player.bases = []
        player.deck = []
        player.discard_pile = []
        opponent = MagicMock()
        opponent.health = 30
        game.get_opponent.return_value = opponent

        collector.record_decision(
            slot=0, game=game, player=player,
            logits=torch.randn(action_dim),
            value=-0.25, mask=np.ones(action_dim, dtype=np.float32),
            action_idx=3, action_type="PLAY_CARD", action_card_id=0,
        )
        collector.finish_game(0, "PPO", 20, "simple")

        with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as f:
            path = f.name
        try:
            collector.save(path)

            meta, replays = ReplayCollector.load(path)
            assert meta["card_names"] == card_names
            assert len(replays) == 1
            assert replays[0].winner == "PPO"
            assert replays[0].total_turns == 20
            assert replays[0].opponent_type == "simple"
            d = replays[0].decisions[0]
            assert d.turn == 7
            assert d.player_health == 42
            assert d.action_type == "PLAY_CARD"
            assert d.value_estimate == -0.25
        finally:
            os.unlink(path)

    def test_buyable_card_ids_from_mask(self, collector, card_setup):
        """Buyable card IDs are correctly extracted from the action mask."""
        card_names, action_dim = card_setup
        num_cards = len(card_names)

        collector.start_game(0)
        from unittest.mock import MagicMock
        game = MagicMock()
        game.stats.total_turns = 1
        game.trade_row = []
        player = MagicMock()
        player.health = 50
        player.trade = 5
        player.combat = 0
        player.hand = []
        player.bases = []
        player.deck = []
        player.discard_pile = []
        opponent = MagicMock()
        opponent.health = 50
        game.get_opponent.return_value = opponent

        # Set specific BUY_CARD bits in the mask
        mask = np.zeros(action_dim, dtype=np.float32)
        mask[1] = 1  # END_TURN
        buy_offset = 3 + num_cards  # matches action_encoder layout
        mask[buy_offset + 0] = 1  # card 0 buyable
        mask[buy_offset + 2] = 1  # card 2 buyable

        collector.record_decision(
            slot=0, game=game, player=player,
            logits=torch.randn(action_dim),
            value=0.0, mask=mask,
            action_idx=1, action_type="END_TURN", action_card_id=None,
        )
        collector.finish_game(0, "PPO", 5, "random")

        record = collector.replays[0].decisions[0]
        assert 0 in record.buyable_card_ids
        assert 2 in record.buyable_card_ids
        assert 1 not in record.buyable_card_ids


# ---------- Game Phase ----------

class TestGamePhase:
    def test_early(self):
        assert _game_phase(1) == "early"
        assert _game_phase(5) == "early"

    def test_mid(self):
        assert _game_phase(6) == "mid"
        assert _game_phase(15) == "mid"

    def test_late(self):
        assert _game_phase(16) == "late"
        assert _game_phase(100) == "late"


# ---------- Analyzer ----------

class TestAnalyzer:
    def _make_replay_file(self, card_names, action_dim, num_games=5):
        """Create a minimal replay file for testing the analyzer."""
        collector = ReplayCollector(card_names, action_dim)
        from unittest.mock import MagicMock

        for g in range(num_games):
            collector.start_game(0)
            game = MagicMock()
            game.trade_row = []
            player = MagicMock()
            player.health = 50 - g * 2
            player.trade = g + 1
            player.combat = g
            player.hand = []
            player.bases = []
            player.deck = list(range(10))
            player.discard_pile = []
            opponent = MagicMock()
            opponent.health = 50
            game.get_opponent.return_value = opponent

            # Simulate a few turns across phases
            for turn in [1, 3, 8, 12, 20]:
                game.stats.total_turns = turn
                mask = np.zeros(action_dim, dtype=np.float32)
                mask[1] = 1  # END_TURN
                num_cards = len(card_names)
                buy_offset = 3 + num_cards
                mask[buy_offset + 0] = 1  # card 0 affordable

                action_type = "BUY_CARD" if turn < 10 else "ATTACK_PLAYER"
                card_id = 0 if turn < 10 else None

                collector.record_decision(
                    slot=0, game=game, player=player,
                    logits=torch.randn(action_dim),
                    value=0.1 * turn, mask=mask,
                    action_idx=1, action_type=action_type, action_card_id=card_id,
                )

            winner = "PPO" if g < 4 else "Opp"
            collector.finish_game(0, winner, total_turns=25, opponent_type="random")

        path = tempfile.mktemp(suffix=".json.gz")
        collector.save(path)
        return path

    def test_analyze_produces_result(self, card_setup):
        """analyze_replays returns a complete AnalysisResult."""
        card_names, action_dim = card_setup
        path = self._make_replay_file(card_names, action_dim)
        try:
            result = analyze_replays(path, output_dir=tempfile.mkdtemp())
            assert isinstance(result, AnalysisResult)
            assert result.num_games == 5
            assert result.num_decisions == 25
            assert result.win_rate == 0.8
            assert result.avg_game_length == 25.0
        finally:
            os.unlink(path)

    def test_analyze_buy_table(self, card_setup):
        """Buy table correctly tracks affordable vs bought."""
        card_names, action_dim = card_setup
        path = self._make_replay_file(card_names, action_dim)
        try:
            result = analyze_replays(path, output_dir=tempfile.mkdtemp())
            # Card 0 should appear in buy table (it was affordable and bought)
            card_0_name = card_names[0]
            assert card_0_name in result.buy_table
            entry = result.buy_table[card_0_name]
            assert entry["overall"]["bought"] > 0
            assert entry["overall"]["affordable"] > 0
            assert entry["overall"]["rate"] > 0
        finally:
            os.unlink(path)

    def test_analyze_entropy_phases(self, card_setup):
        """Entropy is tracked per phase."""
        card_names, action_dim = card_setup
        path = self._make_replay_file(card_names, action_dim)
        try:
            result = analyze_replays(path, output_dir=tempfile.mkdtemp())
            for phase in ["early", "mid", "late"]:
                assert phase in result.entropy_by_phase
                assert "mean" in result.entropy_by_phase[phase]
        finally:
            os.unlink(path)

    def test_analyze_value_accuracy(self, card_setup):
        """Value estimates are separated for wins vs losses."""
        card_names, action_dim = card_setup
        path = self._make_replay_file(card_names, action_dim)
        try:
            result = analyze_replays(path, output_dir=tempfile.mkdtemp())
            assert "wins_mean_value" in result.value_accuracy
            assert "losses_mean_value" in result.value_accuracy
            # Wins should have higher avg value than losses
            assert result.value_accuracy["wins_mean_value"] >= result.value_accuracy["losses_mean_value"]
        finally:
            os.unlink(path)

    def test_analyze_empty_file(self, card_setup):
        """Analyzer handles empty replay files gracefully."""
        card_names, action_dim = card_setup
        path = tempfile.mktemp(suffix=".json.gz")
        meta = {"card_names": card_names, "action_dim": action_dim}
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(json.dumps(meta) + "\n")
        try:
            result = analyze_replays(path, output_dir=tempfile.mkdtemp())
            assert result.num_games == 0
            assert result.num_decisions == 0
        finally:
            os.unlink(path)
