"""Tests for published training presets and their CLI application."""
import hashlib
import json
from pathlib import Path

import pytest

from src.__main__ import _run_train, build_cli_parser, parse_cli_args
from src.config import (
    CHAMPION_CHECKPOINT,
    RunConfig,
    TRAINING_PRESETS,
    get_training_preset,
)

REPO = Path(__file__).resolve().parents[1]
CHAMPION_SHA256 = "9d10dca5110c2bbe9cc0f8693e11578dee347a8c60e7ed63002af7d82e80a503"


def _train_dests() -> set[str]:
    _, train_parser = build_cli_parser()
    return {action.dest for action in train_parser._actions}


def test_preset_keys_are_train_cli_destinations():
    dests = _train_dests()
    for preset in TRAINING_PRESETS.values():
        unknown = set(preset.train_defaults) - dests
        assert not unknown, f"{preset.name} uses unknown train options: {unknown}"


def test_preset_defaults_are_applied_when_flags_are_omitted():
    args = parse_cli_args(["train", "--preset", "reference-selfplay"])
    preset = get_training_preset("reference-selfplay")

    for dest, value in preset.train_defaults.items():
        assert getattr(args, dest) == value, dest


def test_explicit_flags_override_preset_values():
    args = parse_cli_args([
        "train", "--episodes", "64", "--preset", "reference-selfplay",
        "--updates", "2", "--pfsp", "uniform",
    ])

    assert args.episodes == 64
    assert args.updates == 2
    assert args.pfsp == "uniform"
    assert args.engine == "vec"
    assert args.self_play is True


def test_train_defaults_are_unchanged_without_a_preset():
    args = parse_cli_args(["train"])
    cfg = RunConfig()

    assert args.preset is None
    assert args.engine == cfg.engine
    assert args.episodes == cfg.episodes
    assert args.updates == cfg.updates
    assert args.self_play == cfg.self_play
    assert args.pfsp == cfg.pfsp_mode


def test_unknown_preset_is_rejected():
    with pytest.raises(SystemExit):
        parse_cli_args(["train", "--preset", "missing"])
    with pytest.raises(ValueError, match="Unknown training preset"):
        get_training_preset("missing")


def test_run_train_builds_reference_run_config(monkeypatch):
    captured = {}

    def fake_train(data_cfg, ppo_cfg, run_cfg, dev_cfg, **kwargs):
        captured["run_cfg"] = run_cfg
        captured["seed"] = kwargs.get("seed")

    monkeypatch.setattr("src.ppo.ppo_trainer.train", fake_train)
    args = parse_cli_args(
        ["train", "--preset", "reference-selfplay", "--seed", "2"]
    )
    _run_train(args)

    run_cfg = captured["run_cfg"]
    assert run_cfg.engine == "vec"
    assert run_cfg.self_play is True
    assert run_cfg.self_play_schedule == "constant"
    assert run_cfg.pfsp_mode == "variance"
    assert run_cfg.vec_max_active_snapshots == 3
    assert captured["seed"] == 2


def test_reference_preset_matches_published_baseline_manifest():
    preset = get_training_preset("reference-selfplay")
    manifest = json.loads(
        (REPO / "results" / "optimized_selfplay_baseline" / "manifest.json")
        .read_text(encoding="utf-8-sig")
    )
    config = manifest["config"]
    defaults = preset.train_defaults

    assert defaults["updates"] == config["updates"]
    assert defaults["episodes"] == config["episodes"]
    assert defaults["self_play_ratio"] == config["self_play_ratio"]
    assert defaults["self_play_schedule"] == config["self_play_schedule"]
    assert defaults["pfsp"] == config["pfsp"]
    assert defaults["vec_max_active_snapshots"] == config["active_snapshot_cap"]
    assert defaults["num_workers"] == config["num_workers"]
    assert preset.chunk_size == config["chunk_size"]
    assert preset.interim_eval_games == config["interim_eval_games"]
    assert preset.final_eval_games == config["final_eval_games"]
    assert (REPO / preset.results).is_file()


def test_every_preset_points_at_an_existing_record():
    for preset in TRAINING_PRESETS.values():
        assert (REPO / preset.results).is_file(), preset.name


def test_champion_checkpoint_is_the_published_selfplay_seed0():
    digest = hashlib.sha256((REPO / CHAMPION_CHECKPOINT).read_bytes()).hexdigest()
    assert digest == CHAMPION_SHA256


def test_control_preset_is_the_reference_recipe_from_the_champion():
    reference = get_training_preset("reference-selfplay")
    control = get_training_preset("continue-selfplay-seed0")

    assert control.train_defaults == {
        **reference.train_defaults,
        "model_path": CHAMPION_CHECKPOINT,
    }
    for field in (
        "chunk_size", "interim_eval_games", "final_eval_games", "eval_num_workers",
    ):
        assert getattr(control, field) == getattr(reference, field), field


def test_control_preset_starts_training_from_the_champion(monkeypatch):
    captured = {}

    def fake_train(data_cfg, ppo_cfg, run_cfg, dev_cfg, **kwargs):
        captured.update(kwargs, run_cfg=run_cfg)

    monkeypatch.setattr("src.ppo.ppo_trainer.train", fake_train)
    _run_train(parse_cli_args(
        ["train", "--preset", "continue-selfplay-seed0", "--seed", "1"]
    ))

    assert captured["model_path"] == CHAMPION_CHECKPOINT
    assert captured["seed"] == 1
    assert captured["run_cfg"].self_play is True
    assert captured["run_cfg"].resume is None
