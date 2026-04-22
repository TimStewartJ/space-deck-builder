"""Tests for OpponentPool PFSP (Prioritized Fictitious Self-Play)."""
import pytest
from unittest.mock import patch
from src.ppo.opponent_pool import OpponentPool, PFSP_MODES


class TestPFSPModes:
    """Validate PFSP mode parameter handling."""

    def test_valid_modes_accepted(self):
        for mode in PFSP_MODES:
            pool = OpponentPool(pfsp_mode=mode)
            assert pool.pfsp_mode == mode

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="Unknown PFSP mode"):
            OpponentPool(pfsp_mode="invalid")


class TestPFSPWeights:
    """Validate PFSP weight computation for hard and variance modes."""

    def _make_pool(self, mode: str) -> OpponentPool:
        pool = OpponentPool(pfsp_mode=mode)
        pool.add_snapshot({}, "snap_A")
        pool.add_snapshot({}, "snap_B")
        pool.add_snapshot({}, "snap_C")
        return pool

    def test_uniform_returns_equal_weights(self):
        pool = self._make_pool("uniform")
        pool._snapshot_ema["snap_A"] = 0.9
        pool._snapshot_ema["snap_B"] = 0.5
        pool._snapshot_ema["snap_C"] = 0.1
        weights = pool._pfsp_weights(["snap_A", "snap_B", "snap_C"])
        assert weights == [1.0, 1.0, 1.0]

    def test_hard_favors_low_win_rate(self):
        pool = self._make_pool("hard")
        pool._snapshot_ema["snap_A"] = 0.9   # easy opponent
        pool._snapshot_ema["snap_B"] = 0.5
        pool._snapshot_ema["snap_C"] = 0.1   # hard opponent
        # Set high game counts for full confidence
        for name in ["snap_A", "snap_B", "snap_C"]:
            pool._snapshot_games[name] = 100
        weights = pool._pfsp_weights(["snap_A", "snap_B", "snap_C"])
        # snap_C (low win rate = hard opponent) should get highest weight
        assert weights[2] > weights[1] > weights[0]

    def test_hard_weight_values(self):
        pool = self._make_pool("hard")
        pool._snapshot_ema["snap_A"] = 0.8
        pool._snapshot_ema["snap_B"] = 0.2
        # Set high game counts so confidence ramp is fully engaged
        pool._snapshot_games["snap_A"] = 100
        pool._snapshot_games["snap_B"] = 100
        weights = pool._pfsp_weights(["snap_A", "snap_B"])
        assert pytest.approx(weights[0], abs=1e-5) == 0.2   # 1 - 0.8
        assert pytest.approx(weights[1], abs=1e-5) == 0.8   # 1 - 0.2

    def test_variance_favors_50_percent(self):
        pool = self._make_pool("variance")
        pool._snapshot_ema["snap_A"] = 0.95  # easy
        pool._snapshot_ema["snap_B"] = 0.50  # balanced
        pool._snapshot_ema["snap_C"] = 0.05  # too hard
        # Set high game counts so confidence ramp is fully engaged
        for name in ["snap_A", "snap_B", "snap_C"]:
            pool._snapshot_games[name] = 100
        weights = pool._pfsp_weights(["snap_A", "snap_B", "snap_C"])
        # snap_B (near 50%) should get highest weight
        assert weights[1] > weights[0]
        assert weights[1] > weights[2]

    def test_variance_weight_values(self):
        pool = self._make_pool("variance")
        pool._snapshot_ema["snap_A"] = 0.5
        pool._snapshot_ema["snap_B"] = 0.9
        # Set high game counts for full confidence
        pool._snapshot_games["snap_A"] = 100
        pool._snapshot_games["snap_B"] = 100
        weights = pool._pfsp_weights(["snap_A", "snap_B"])
        assert pytest.approx(weights[0], abs=1e-5) == 0.25   # 0.5 * 0.5
        assert pytest.approx(weights[1], abs=1e-5) == 0.09   # 0.9 * 0.1

    def test_weights_have_minimum_floor(self):
        """Even a 100% win rate opponent should get a non-zero weight."""
        pool = self._make_pool("hard")
        pool._snapshot_ema["snap_A"] = 1.0  # agent always wins → priority 0
        pool._snapshot_games["snap_A"] = 100
        weights = pool._pfsp_weights(["snap_A"])
        assert weights[0] > 0

    def test_new_snapshot_gets_uniform_weight(self):
        """Untested snapshots should get uniform weight (confidence=0)."""
        pool = self._make_pool("variance")
        pool._snapshot_ema["snap_A"] = 0.5  # prior, but 0 games
        pool._snapshot_ema["snap_B"] = 0.9
        pool._snapshot_games["snap_A"] = 0    # no games → uniform
        pool._snapshot_games["snap_B"] = 100  # fully established
        weights = pool._pfsp_weights(["snap_A", "snap_B"])
        # snap_A at confidence=0 should get 1.0 (uniform), not 0.25 (variance)
        assert pytest.approx(weights[0], abs=1e-5) == 1.0
        assert pytest.approx(weights[1], abs=1e-5) == 0.09  # 0.9 * 0.1

    def test_empty_snapshots_returns_empty(self):
        pool = OpponentPool(pfsp_mode="hard")
        assert pool._pfsp_weights([]) == []


class TestEMATracking:
    """Validate EMA win rate updates."""

    def test_new_snapshot_starts_at_half(self):
        pool = OpponentPool(pfsp_mode="hard")
        pool.add_snapshot({}, "snap_1")
        assert pool._snapshot_ema["snap_1"] == 0.5

    def test_ema_update_moves_toward_batch_rate(self):
        pool = OpponentPool(pfsp_mode="hard", pfsp_ema_alpha=0.3)
        pool.add_snapshot({}, "snap_1")
        # Batch: 8 wins / 10 games = 80% win rate
        # effective_alpha = 0.3 * min(10/10, 1.0) = 0.3  (full-size batch)
        pool.update_results({"snap_1": (8, 10)})
        # EMA = 0.3 * 0.8 + 0.7 * 0.5 = 0.59
        assert pytest.approx(pool._snapshot_ema["snap_1"], abs=1e-5) == 0.59

    def test_ema_small_sample_has_less_impact(self):
        """A 1-game batch should move the EMA much less than a 10-game batch."""
        pool_small = OpponentPool(pfsp_mode="hard", pfsp_ema_alpha=0.3)
        pool_small.add_snapshot({}, "snap_1")
        pool_small.update_results({"snap_1": (1, 1)})  # 1 game, 100% WR

        pool_large = OpponentPool(pfsp_mode="hard", pfsp_ema_alpha=0.3)
        pool_large.add_snapshot({}, "snap_1")
        pool_large.update_results({"snap_1": (10, 10)})  # 10 games, 100% WR

        # Small sample should barely move from 0.5 prior
        small_delta = abs(pool_small._snapshot_ema["snap_1"] - 0.5)
        large_delta = abs(pool_large._snapshot_ema["snap_1"] - 0.5)
        assert large_delta > small_delta * 3

    def test_ema_converges_with_repeated_updates(self):
        pool = OpponentPool(pfsp_mode="hard", pfsp_ema_alpha=0.3)
        pool.add_snapshot({}, "snap_1")
        for _ in range(20):
            pool.update_results({"snap_1": (9, 10)})
        # Should converge near 0.9
        assert pool._snapshot_ema["snap_1"] > 0.85

    def test_update_ignores_unknown_opponents(self):
        pool = OpponentPool(pfsp_mode="hard")
        pool.add_snapshot({}, "snap_1")
        pool.update_results({"Random": (10, 10), "Unknown": (5, 5)})
        # snap_1 should be unchanged (still at prior)
        assert pool._snapshot_ema["snap_1"] == 0.5

    def test_update_ignores_zero_game_results(self):
        pool = OpponentPool(pfsp_mode="hard")
        pool.add_snapshot({}, "snap_1")
        pool.update_results({"snap_1": (0, 0)})
        assert pool._snapshot_ema["snap_1"] == 0.5

    def test_multiple_snapshots_updated_independently(self):
        pool = OpponentPool(pfsp_mode="hard", pfsp_ema_alpha=0.5)
        pool.add_snapshot({}, "snap_1")
        pool.add_snapshot({}, "snap_2")
        # Use 10+ games to get full alpha (reference_games=10)
        pool.update_results({"snap_1": (10, 10), "snap_2": (0, 10)})
        # snap_1: 0.5 * 1.0 + 0.5 * 0.5 = 0.75
        assert pytest.approx(pool._snapshot_ema["snap_1"], abs=1e-5) == 0.75
        # snap_2: 0.5 * 0.0 + 0.5 * 0.5 = 0.25
        assert pytest.approx(pool._snapshot_ema["snap_2"], abs=1e-5) == 0.25

    def test_game_count_accumulates(self):
        pool = OpponentPool(pfsp_mode="hard")
        pool.add_snapshot({}, "snap_1")
        pool.update_results({"snap_1": (5, 10)})
        pool.update_results({"snap_1": (3, 8)})
        assert pool._snapshot_games["snap_1"] == 18


class TestSnapshotEviction:
    """Validate that EMA stats are cleaned up when snapshots are evicted."""

    def test_evicted_snapshot_stats_removed(self):
        pool = OpponentPool(pfsp_mode="hard", snapshot_cap=2)
        pool.add_snapshot({}, "snap_1")
        pool.add_snapshot({}, "snap_2")
        pool.update_results({"snap_1": (5, 10)})
        # Adding a third snapshot should evict snap_1
        pool.add_snapshot({}, "snap_3")
        assert "snap_1" not in pool._snapshot_ema
        assert "snap_1" not in pool._snapshot_games
        assert "snap_2" in pool._snapshot_ema
        assert "snap_3" in pool._snapshot_ema

    def test_evicted_snapshot_not_in_pool(self):
        pool = OpponentPool(pfsp_mode="hard", snapshot_cap=2)
        pool.add_snapshot({}, "snap_1")
        pool.add_snapshot({}, "snap_2")
        pool.add_snapshot({}, "snap_3")
        snap_names = [n for n, _, _ in pool._snapshots]
        assert "snap_1" not in snap_names


class TestGeometricEviction:
    """Validate the geometric (log-spaced age) snapshot eviction strategy."""

    def _names(self, pool):
        return [n for n, _, _ in pool._snapshots]

    def test_geometric_keeps_newest_always(self):
        pool = OpponentPool(snapshot_cap=3)
        for upd in range(1, 11):
            pool.add_snapshot({}, f"PPO_{upd}", update=upd)
        # Newest snapshot is always kept because target age 1 picks it.
        assert "PPO_10" in self._names(pool)

    def test_geometric_produces_log_spaced_ages(self):
        pool = OpponentPool(snapshot_cap=4)
        for upd in range(1, 33):
            pool.add_snapshot({}, f"PPO_{upd}", update=upd)
        kept_updates = sorted(pool._snapshot_updates.values(), reverse=True)
        # Endpoints are pinned: newest (32) and oldest (1) always survive.
        assert len(kept_updates) == 4
        assert kept_updates[0] == 32
        assert kept_updates[-1] == 1
        # Intermediate ages spread log-style: gaps from newest grow
        # geometrically (each gap is strictly larger than the previous).
        gaps = [kept_updates[i] - kept_updates[i + 1] for i in range(3)]
        assert all(gaps[i + 1] >= gaps[i] for i in range(2)), (
            f"expected non-decreasing gaps, got {gaps} from {kept_updates}"
        )

    def test_geometric_large_pool_spread(self):
        pool = OpponentPool(snapshot_cap=10)
        for upd in range(1, 1001):
            pool.add_snapshot({}, f"PPO_{upd}", update=upd)
        kept = sorted(pool._snapshot_updates.values(), reverse=True)
        # Endpoints pinned: newest (1000) and oldest-reachable (≈1).
        assert kept[0] == 1000
        assert kept[-1] == 1
        # Ladder spans the full history — ratio of consecutive "ages" from
        # newest is roughly geometric so no two kept updates should be closer
        # than ~1 apart at the top nor more than ~half the remaining history
        # at the bottom.
        assert len(kept) == 10
        # Strictly decreasing and spread across the full range.
        assert all(a > b for a, b in zip(kept, kept[1:]))

    def test_geometric_preserves_stats_for_kept(self):
        pool = OpponentPool(snapshot_cap=3)
        for upd in range(1, 11):
            pool.add_snapshot({}, f"PPO_{upd}", update=upd)
        # After 10 adds the pool has compacted itself several times under
        # geometric eviction; record EMA on a snapshot that is guaranteed to
        # survive the next compaction (the newest one is always kept).
        pool.update_results({"PPO_10": (8, 10)})
        pool.add_snapshot({}, "PPO_11", update=11)
        # Newest (PPO_11) always survives; verify its stat bookkeeping exists.
        assert "PPO_11" in self._names(pool)
        assert "PPO_11" in pool._snapshot_ema

    def test_fifo_fallback_when_updates_missing(self):
        # Eviction transparently falls back to FIFO if any pool member lacks
        # an update — geometric age-spacing requires update numbers.
        pool = OpponentPool(snapshot_cap=2)
        pool.add_snapshot({}, "snap_1")       # no update
        pool.add_snapshot({}, "snap_2")       # no update
        pool.add_snapshot({}, "snap_3")       # no update → FIFO: drop snap_1
        assert self._names(pool) == ["snap_2", "snap_3"]

    def test_geometric_evicts_stats(self):
        pool = OpponentPool(snapshot_cap=3)
        for upd in range(1, 11):
            pool.add_snapshot({}, f"PPO_{upd}", update=upd)
        dropped_names = set(range(1, 11)) - {
            int(n.split("_")[1]) for n in self._names(pool)
        }
        for upd in dropped_names:
            name = f"PPO_{upd}"
            assert name not in pool._snapshot_ema
            assert name not in pool._snapshot_games
            assert name not in pool._snapshot_updates


class TestPFSPSummary:
    """Validate the PFSP summary output for logging."""

    def test_summary_empty_when_no_snapshots(self):
        pool = OpponentPool(pfsp_mode="hard")
        assert pool.get_pfsp_summary() == {}

    def test_summary_contains_all_snapshots(self):
        pool = OpponentPool(pfsp_mode="hard")
        pool.add_snapshot({}, "snap_1")
        pool.add_snapshot({}, "snap_2")
        summary = pool.get_pfsp_summary()
        assert "snap_1" in summary
        assert "snap_2" in summary

    def test_summary_weights_sum_to_one(self):
        pool = OpponentPool(pfsp_mode="hard")
        pool.add_snapshot({}, "snap_1")
        pool.add_snapshot({}, "snap_2")
        pool._snapshot_ema["snap_1"] = 0.8
        pool._snapshot_ema["snap_2"] = 0.3
        summary = pool.get_pfsp_summary()
        total = sum(v["weight"] for v in summary.values())
        assert pytest.approx(total, abs=1e-5) == 1.0

    def test_summary_includes_ema_and_weight(self):
        pool = OpponentPool(pfsp_mode="variance")
        pool.add_snapshot({}, "snap_1")
        summary = pool.get_pfsp_summary()
        assert "ema_win_rate" in summary["snap_1"]
        assert "weight" in summary["snap_1"]


class TestFactoryPFSPIntegration:
    """Validate that make_factory uses PFSP weights for snapshot selection."""

    def test_hard_mode_biases_toward_hard_opponents(self):
        """With extreme win rates, hard mode should strongly prefer the harder opponent."""
        pool = OpponentPool(
            pfsp_mode="hard", self_play_ratio=1.0, snapshot_cap=10,
        )
        pool.add_snapshot({"dummy": 1}, "easy_snap")
        pool.add_snapshot({"dummy": 2}, "hard_snap")
        pool._snapshot_ema["easy_snap"] = 0.95  # agent always wins
        pool._snapshot_ema["hard_snap"] = 0.05  # agent always loses
        pool._snapshot_games["easy_snap"] = 100
        pool._snapshot_games["hard_snap"] = 100

        # Patch _make_ppo_opponent to avoid needing real model infrastructure
        with patch("src.ppo.opponent_pool._make_ppo_opponent") as mock_make:
            from src.ai.random_agent import RandomAgent
            mock_make.side_effect = lambda name, sd, cn, d, registry=None, model_config=None: RandomAgent(name)
            factory = pool.make_factory(card_names=["Scout"])

            selections = {"easy_snap": 0, "hard_snap": 0}
            for _ in range(1000):
                agent = factory()
                if agent.name in selections:
                    selections[agent.name] += 1

        # hard_snap should be selected much more often
        assert selections["hard_snap"] > selections["easy_snap"] * 3

    def test_uniform_mode_selects_evenly(self):
        """Uniform mode should not bias even with skewed EMA values."""
        pool = OpponentPool(
            pfsp_mode="uniform", self_play_ratio=1.0, snapshot_cap=10,
        )
        pool.add_snapshot({"dummy": 1}, "snap_A")
        pool.add_snapshot({"dummy": 2}, "snap_B")
        pool._snapshot_ema["snap_A"] = 0.95
        pool._snapshot_ema["snap_B"] = 0.05

        with patch("src.ppo.opponent_pool._make_ppo_opponent") as mock_make:
            from src.ai.random_agent import RandomAgent
            mock_make.side_effect = lambda name, sd, cn, d, registry=None, model_config=None: RandomAgent(name)
            factory = pool.make_factory(card_names=["Scout"])

            selections = {"snap_A": 0, "snap_B": 0}
            for _ in range(1000):
                agent = factory()
                if agent.name in selections:
                    selections[agent.name] += 1

        # Should be roughly 50/50 (within statistical bounds)
        assert 350 < selections["snap_A"] < 650
        assert 350 < selections["snap_B"] < 650
