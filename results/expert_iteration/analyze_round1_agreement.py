"""Analyze how the round 1 student adopted search's move changes.

Usage: python results/expert_iteration/analyze_round1_agreement.py <data_dir> <student.pth> [output.json]

Evaluates the student and source checkpoints on the held-out validation games
(index % 20 == 0) of the round 1 self-play shards, and compares distilled
search against source search on the same diagnostic deals.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.config import DataConfig, ModelConfig, load_checkpoint  # noqa: E402
from src.encoding.action_encoder import get_action_space_size  # noqa: E402
from src.encoding.state_encoder import get_state_size  # noqa: E402
from src.ppo.ppo_actor_critic import PPOActorCritic  # noqa: E402
from src.search.selfplay_data import load_selfplay_data  # noqa: E402

SOURCE = REPO / "results" / "optimized_selfplay_baseline" / "checkpoints" / "selfplay_seed0.pth"
DIAGNOSTIC = REPO / "results" / "expert_iteration" / "round1" / "diagnostic_search_eval.json"
MATCHED = REPO / "results" / "batched_search" / "strength_seed0_budget0_64.json"


def load_model(path: Path, registry) -> PPOActorCritic:
    names = registry.card_names
    checkpoint = load_checkpoint(str(path), map_location="cpu")
    model = PPOActorCritic(
        get_state_size(names), get_action_space_size(names), len(names),
        model_config=ModelConfig.from_dict(checkpoint["config"]["model"]), card_registry=registry,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


@torch.no_grad()
def greedy(model, observations: np.ndarray, masks: np.ndarray) -> np.ndarray:
    actions = []
    for start in range(0, len(observations), 16384):
        logits, _ = model(torch.from_numpy(observations[start : start + 16384]))
        legal = torch.from_numpy(masks[start : start + 16384])
        actions.append(logits.masked_fill(~legal, float("-inf")).argmax(dim=1).numpy())
    return np.concatenate(actions)


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n) if n else 1.0


def main() -> None:
    data_dir, student_path = Path(sys.argv[1]), Path(sys.argv[2])
    output = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    torch.set_num_threads(8)
    registry = DataConfig().build_registry(DataConfig().load_cards())
    data = load_selfplay_data(data_dir)
    rows = np.flatnonzero((data.game % 20 == 0) & ~data.forced)
    observations, masks = data.observations[rows], data.masks[rows]
    search = data.action[rows].astype(np.int64)
    source = data.greedy[rows].astype(np.int64)
    target = data.policy[rows].argmax(axis=1)
    student = greedy(load_model(student_path, registry), observations, masks)
    recomputed_source = greedy(load_model(SOURCE, registry), observations, masks)

    changed = search != source
    student_changed = student != source
    result = {
        "validation_non_forced_positions": int(len(rows)),
        "recorded_greedy_matches_source_model": float(np.mean(recomputed_source == source)),
        "student_search_agreement": float(np.mean(student == search)),
        "student_source_agreement": float(np.mean(student == source)),
        "search_source_agreement": float(np.mean(search == source)),
        "search_changed_positions": int(changed.sum()),
        "search_changed_fraction": float(np.mean(changed)),
        "student_plays_search_move_where_search_changed": float(np.mean(student[changed] == search[changed])),
        "student_keeps_greedy_where_search_changed": float(np.mean(student[changed] == source[changed])),
        "student_changed_positions": int(student_changed.sum()),
        "student_changes_where_search_kept_greedy": float(np.mean(~changed[student_changed])),
        "student_changes_matching_target_top": float(np.mean(student[student_changed] == target[student_changed])),
        "target_top_matches_search_move": float(np.mean(target == search)),
    }

    distilled = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    matched = json.loads(MATCHED.read_text(encoding="utf-8"))
    # The matched run's payload predates the search-implementation label.
    # Between the batched search commit and the label, only latency
    # attribution and resume identity changed, and replaying matched deals
    # with the labeled implementation reproduces its games.
    for payload in (distilled, matched):
        config = payload["config"]
        settings = (config["seed"], config["games"], config["max_candidates"], config["baseline"], 64 in config["budgets"])
        if settings != (0, 1000, 8, "greedy", True):
            raise ValueError("diagnostic and matched runs must share deals and search settings")
    if distilled["config"].get("search_implementation") != "batched-count-roots-v1":
        raise ValueError("the diagnostic must use the batched search implementation")
    new = {row["index"]: row["search_won"] for row in distilled["games"] if row["budget"] == 64}
    old = {row["index"]: row["search_won"] for row in matched["games"] if row["budget"] == 64}
    only_new = sum(1 for index in new if new[index] and not old[index])
    only_old = sum(1 for index in new if old[index] and not new[index])
    n = len(new)
    delta = (only_new - only_old) / n
    half = 1.959963984540054 * math.sqrt(((only_new + only_old) - (only_new - only_old) ** 2 / n) / n**2)
    result["search_advantage_comparison"] = {
        "distilled_budget64_win_rate": sum(new.values()) / n,
        "source_budget64_win_rate": sum(old.values()) / n,
        "paired_difference": delta,
        "paired_difference_ci95": [delta - half, delta + half],
        "discordant_distilled_only": only_new,
        "discordant_source_only": only_old,
        "exact_mcnemar_p": exact_mcnemar(only_new, only_old),
    }
    print(json.dumps(result, indent=2))
    if output is not None:
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
