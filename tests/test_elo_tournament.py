"""Tests for Elo tournament enhancements: participant types, pairing
dispatch, built-in agent support, and multi-worker integration."""
import pytest
from unittest.mock import patch, MagicMock

from src.ppo.elo_tournament import (
    CheckpointParticipant,
    BuiltinParticipant,
    EloResult,
    BUILTIN_AGENT_TYPES,
    build_participants,
    expected_score,
    compute_mle_ratings,
    _extract_label,
    _validate_checkpoints,
    _play_builtin_games,
)
from src.config import ModelConfig


# ---------------------------------------------------------------------------
# Participant types
# ---------------------------------------------------------------------------

class TestParticipantTypes:
    """CheckpointParticipant and BuiltinParticipant dataclass behavior."""

    def test_checkpoint_participant_fields(self):
        cp = CheckpointParticipant(
            label="upd50", path="models/ckpt.pth",
            state_dict={"k": "v"}, model_config=ModelConfig(),
        )
        assert cp.label == "upd50"
        assert cp.path == "models/ckpt.pth"
        assert cp.state_dict == {"k": "v"}
        assert isinstance(cp.model_config, ModelConfig)

    def test_builtin_participant_fields(self):
        bp = BuiltinParticipant(label="random", agent_type="random")
        assert bp.label == "random"
        assert bp.agent_type == "random"
        assert bp.path == ""

    def test_builtin_agent_types_registry(self):
        assert "random" in BUILTIN_AGENT_TYPES
        assert "heuristic" in BUILTIN_AGENT_TYPES
        assert "simple" in BUILTIN_AGENT_TYPES


# ---------------------------------------------------------------------------
# build_participants
# ---------------------------------------------------------------------------

class TestBuildParticipants:
    """Participant construction and validation."""

    def _mock_checkpoint(self, model_config: ModelConfig | None = None):
        """Create a mock checkpoint dict matching load_checkpoint output."""
        cfg = model_config or ModelConfig()
        return {
            "schema_version": 2,
            "model_state_dict": {"dummy": "state"},
            "config": {"model": cfg.to_dict()},
        }

    @patch("src.ppo.elo_tournament.load_checkpoint")
    def test_checkpoint_only(self, mock_load):
        mock_load.return_value = self._mock_checkpoint()
        participants = build_participants(["a.pth", "b.pth"])
        assert len(participants) == 2
        assert all(isinstance(p, CheckpointParticipant) for p in participants)

    @patch("src.ppo.elo_tournament.load_checkpoint")
    def test_agents_only(self, mock_load):
        participants = build_participants([], agent_types=["random", "heuristic"])
        assert len(participants) == 2
        assert all(isinstance(p, BuiltinParticipant) for p in participants)
        mock_load.assert_not_called()

    @patch("src.ppo.elo_tournament.load_checkpoint")
    def test_mixed_participants(self, mock_load):
        mock_load.return_value = self._mock_checkpoint()
        participants = build_participants(["a.pth"], agent_types=["simple"])
        assert len(participants) == 2
        assert isinstance(participants[0], CheckpointParticipant)
        assert isinstance(participants[1], BuiltinParticipant)

    def test_invalid_agent_type_rejected(self):
        with pytest.raises(ValueError, match="Unknown agent type"):
            build_participants([], agent_types=["invalid_agent"])

    @patch("src.ppo.elo_tournament.load_checkpoint")
    def test_mixed_configs_allowed(self, mock_load):
        """Cross-architecture tournaments are the whole point of ELO."""
        cfg_a = ModelConfig(card_emb_dim=32)
        cfg_b = ModelConfig(card_emb_dim=64)
        mock_load.side_effect = [
            self._mock_checkpoint(cfg_a),
            self._mock_checkpoint(cfg_b),
        ]
        participants = build_participants(["a.pth", "b.pth"])
        assert len(participants) == 2
        assert participants[0].model_config.card_emb_dim == 32
        assert participants[1].model_config.card_emb_dim == 64


# ---------------------------------------------------------------------------
# _validate_checkpoints
# ---------------------------------------------------------------------------

class TestValidateCheckpoints:
    """Checkpoint compatibility validation."""

    def test_single_checkpoint_passes(self):
        cp = CheckpointParticipant("a", "a.pth", {}, ModelConfig())
        _validate_checkpoints([cp])

    def test_matching_configs_pass(self):
        cfg = ModelConfig()
        cps = [
            CheckpointParticipant("a", "a.pth", {}, cfg),
            CheckpointParticipant("b", "b.pth", {}, cfg),
        ]
        _validate_checkpoints(cps)

    def test_mismatched_configs_allowed(self):
        """Different architectures are allowed — ELO measures relative strength."""
        cps = [
            CheckpointParticipant("a", "a.pth", {}, ModelConfig(card_emb_dim=32)),
            CheckpointParticipant("b", "b.pth", {}, ModelConfig(card_emb_dim=64)),
        ]
        _validate_checkpoints(cps)  # no exception


# ---------------------------------------------------------------------------
# Elo math (unchanged, but verify still works)
# ---------------------------------------------------------------------------

class TestEloMath:
    """Maximum likelihood Elo rating computation."""

    def test_expected_score_equal_ratings(self):
        assert expected_score(1000, 1000) == pytest.approx(0.5)

    def test_expected_score_higher_rated(self):
        score = expected_score(1200, 1000)
        assert score > 0.5

    def test_mle_equal_record_equal_ratings(self):
        """Players with identical records get similar ratings."""
        # A beats B 50%, B beats A 50%
        wins = {(0, 1): 500}
        games = {(0, 1): 1000}
        ratings = compute_mle_ratings(2, wins, games)
        assert ratings[0] == pytest.approx(ratings[1], abs=1.0)

    def test_mle_dominant_player_ranked_higher(self):
        """Player who wins most games gets highest rating."""
        # A beats B 80%, A beats C 90%, B beats C 70%
        wins = {(0, 1): 800, (0, 2): 900, (1, 2): 700}
        games = {(0, 1): 1000, (0, 2): 1000, (1, 2): 1000}
        ratings = compute_mle_ratings(3, wins, games)
        assert ratings[0] > ratings[1] > ratings[2]

    def test_mle_lopsided_result_produces_large_gap(self):
        """88% win rate should produce a meaningful rating difference."""
        wins = {(0, 1): 880}
        games = {(0, 1): 1000}
        ratings = compute_mle_ratings(2, wins, games)
        assert ratings[0] > ratings[1]
        assert ratings[0] - ratings[1] > 100  # meaningful gap

    def test_mle_anchor_stays_fixed(self):
        """The anchor player (index 0) stays at INITIAL_ELO."""
        wins = {(0, 1): 700, (0, 2): 800, (1, 2): 600}
        games = {(0, 1): 1000, (0, 2): 1000, (1, 2): 1000}
        ratings = compute_mle_ratings(3, wins, games, anchor=0)
        assert ratings[0] == pytest.approx(1000.0)

    def test_mle_perfect_record(self):
        """Player with 100% win rate gets highest rating (no crash)."""
        wins = {(0, 1): 1000, (0, 2): 1000, (1, 2): 600}
        games = {(0, 1): 1000, (0, 2): 1000, (1, 2): 1000}
        ratings = compute_mle_ratings(3, wins, games)
        assert ratings[0] > ratings[1] > ratings[2]


# ---------------------------------------------------------------------------
# _play_builtin_games
# ---------------------------------------------------------------------------

class TestPlayBuiltinGames:
    """Built-in agent vs built-in agent direct game simulation."""

    @pytest.fixture
    def game_data(self):
        """Load cards and build card_index_map for game simulation."""
        from src.config import DataConfig
        data_cfg = DataConfig()
        cards = data_cfg.load_cards()
        registry = data_cfg.build_registry(cards)
        return cards, registry.card_index_map

    def test_random_vs_random_completes(self, game_data):
        cards, card_index_map = game_data
        wins_a, wins_b = _play_builtin_games(
            "random", "random", cards, card_index_map, num_games=5,
        )
        assert wins_a + wins_b == 5
        assert wins_a >= 0
        assert wins_b >= 0

    def test_heuristic_vs_simple_completes(self, game_data):
        cards, card_index_map = game_data
        wins_a, wins_b = _play_builtin_games(
            "heuristic", "simple", cards, card_index_map, num_games=5,
        )
        assert wins_a + wins_b == 5

    def test_all_builtin_types_supported(self, game_data):
        cards, card_index_map = game_data
        for agent_type in BUILTIN_AGENT_TYPES:
            wins_a, wins_b = _play_builtin_games(
                agent_type, "random", cards, card_index_map, num_games=2,
            )
            assert wins_a + wins_b == 2


# ---------------------------------------------------------------------------
# _extract_label
# ---------------------------------------------------------------------------

class TestExtractLabel:
    """Label extraction from checkpoint filenames."""

    def test_extracts_upd_with_timestamp(self):
        # Full ppo_trainer pattern: <date>_<time>_upd<N>_wins<W>.
        assert _extract_label("models/ppo_agent_0415_0348_upd200_wins95.pth") == "0415_0348_upd200"

    def test_extracts_upd_number_only_when_no_timestamp(self):
        # Only one numeric prefix → not enough to form a timestamp; fall back to upd tag.
        assert _extract_label("models/ppo_agent_0415_upd50_wins65.pth") == "upd50"

    def test_fallback_to_stem(self):
        assert _extract_label("models/my_model.pth") == "my_model"


# ---------------------------------------------------------------------------
# EloResult
# ---------------------------------------------------------------------------

class TestEloResult:
    """EloResult dataclass behavior."""

    def test_win_rate(self):
        r = EloResult("test", "", 1000, games_played=10, wins=7, losses=3)
        assert r.win_rate == pytest.approx(0.7)

    def test_win_rate_zero_games(self):
        r = EloResult("test", "", 1000, games_played=0, wins=0, losses=0)
        assert r.win_rate == 0.0
