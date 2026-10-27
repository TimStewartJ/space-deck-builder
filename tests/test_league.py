"""Held-out league: scripted opponents, matchups, journal resume, and isolation."""
from __future__ import annotations

import json

import pytest
import torch

from src.ai.heuristic_agent import HeuristicAgent
from src.ai.held_out_agents import (
    HELD_OUT_STYLES,
    STYLE_PARAMETERS,
    HeldOutAgent,
    held_out_agent_factory,
)
from src.config import DataConfig, ModelConfig, save_checkpoint
from src.encoding.action_encoder import get_action_space_size
from src.encoding.state_encoder import get_state_size
from src.engine.actions import get_available_actions
from src.ppo import league
from src.ppo.league import LeagueConfig, play_scripted_games, run_league
from src.ppo.ppo_actor_critic import PPOActorCritic


@pytest.fixture(scope="module")
def setup():
    data = DataConfig()
    cards = data.load_cards()
    return cards, data.build_registry(cards)


def _signature(action):
    return (
        action.type,
        id(action.card) if action.card is not None else None,
        action.card_id,
        action.card_source,
        action.target_id,
        id(action.card_effect) if action.card_effect is not None else None,
    )


class LegalityChecked(HeldOutAgent):
    """A held-out agent whose every decision is checked against the legal actions."""

    def make_decision(self, game):
        legal = {_signature(action) for action in get_available_actions(game, game.current_player)}
        action = super().make_decision(game)
        assert _signature(action) in legal, action
        return action


def _checked(style):
    return lambda name: LegalityChecked(name, style)


@pytest.mark.parametrize("style", HELD_OUT_STYLES)
def test_held_out_agents_play_legal_repeatable_games(setup, style):
    cards, registry = setup
    against_heuristic = play_scripted_games(
        cards, registry, _checked(style), HeuristicAgent, 20, seed=5,
    )
    against_itself = play_scripted_games(
        cards, registry, _checked(style), _checked(style), 20, seed=6,
    )
    again = play_scripted_games(
        cards, registry, _checked(style), HeuristicAgent, 20, seed=5,
    )

    assert len(against_heuristic) == len(against_itself) == 20
    assert [(row["a_won"], row["turns"]) for row in against_heuristic] == [
        (row["a_won"], row["turns"]) for row in again
    ]
    assert sum(row["a_first"] for row in against_heuristic) == 10


def test_unknown_style_is_rejected():
    with pytest.raises(ValueError, match="Unknown held-out style"):
        HeldOutAgent("x", "turtle")
    with pytest.raises(ValueError, match="Unknown held-out style"):
        held_out_agent_factory("turtle")


def test_style_parameters_are_read_only():
    with pytest.raises(TypeError):
        STYLE_PARAMETERS["balanced"]["thin"] = (0.0, 0.0, 0.0)
    with pytest.raises(TypeError):
        STYLE_PARAMETERS["scrapper"]["faction_bonus"][2] = 0.0


def test_held_out_agents_are_in_no_training_registry():
    from src.ppo.elo_tournament import BUILTIN_AGENT_TYPES, _AGENT_FACTORIES as ELO
    from src.ppo.mp_sim_worker import _AGENT_FACTORIES as WORKER
    from src.ppo.opponent_pool import AGENT_REGISTRY
    from src.vector_engine.opponents import OPPONENT_CODE

    for registry in (AGENT_REGISTRY, ELO, WORKER, OPPONENT_CODE, BUILTIN_AGENT_TYPES):
        assert not set(HELD_OUT_STYLES) & set(registry)


def test_league_config_validation():
    with pytest.raises(ValueError, match="held-out styles"):
        LeagueConfig(models=("a.pth",), opponents=("heuristic",))
    with pytest.raises(ValueError, match="at least one model"):
        LeagueConfig()
    with pytest.raises(ValueError, match="even"):
        LeagueConfig(models=("a.pth",), games=3)


def _checkpoint(tmp_path, registry):
    torch.manual_seed(0)
    model = PPOActorCritic(
        state_dim=get_state_size(registry.card_names),
        action_dim=get_action_space_size(registry.card_names),
        num_cards=registry.num_cards,
        model_config=ModelConfig(),
    )
    path = tmp_path / "tiny_policy.pth"
    save_checkpoint(str(path), model.state_dict(), model_config=ModelConfig())
    return str(path)


def test_league_plays_exact_seats_repeatably_and_resumes(setup, tmp_path, monkeypatch):
    _, registry = setup
    model = _checkpoint(tmp_path, registry)
    config = LeagueConfig(
        models=(model,), opponents=("balanced", "aggro"), games=4,
        calibration=True, calibration_games=4, workers=1, device="cpu",
    )
    output = tmp_path / "league.json"

    summary = run_league(config, str(output), log_fn=lambda message: None)

    [checkpoint] = summary["checkpoints"]
    assert checkpoint["label"] == "tiny_policy"
    assert checkpoint["pooled"]["games"] == 8
    assert checkpoint["pooled"]["complete"] is True
    for opponent in ("balanced", "aggro"):
        row = checkpoint["opponents"][opponent]
        assert row["games"] == 4 and row["first_seat_games"] == 2
        assert 0 <= row["wins"] <= 4
    assert set(summary["calibration"]) == {"balanced", "aggro"}
    assert json.loads(output.read_text(encoding="utf-8")) == summary

    second = LeagueConfig(
        models=(model,), opponents=("balanced", "aggro"), games=4,
        workers=1, device="cpu",
    )
    repeat = run_league(second, str(tmp_path / "repeat.json"), log_fn=lambda message: None)
    assert repeat["checkpoints"][0]["opponents"] == checkpoint["opponents"]

    def fail(task):
        raise AssertionError(f"journaled matchup replayed: {task}")

    monkeypatch.setattr(league, "run_task", fail)
    resumed = run_league(config, str(output), log_fn=lambda message: None)
    assert resumed == summary
